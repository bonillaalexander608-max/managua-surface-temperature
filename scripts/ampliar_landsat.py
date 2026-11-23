"""Build a monthly Landsat thermal inventory over Managua in Earth Engine."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--aoi", required=True, type=Path, help="Municipal geometry or Feature in WGS84 GeoJSON")
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--min-coverage", type=float, default=0.70)
    parser.add_argument("--output", type=Path, default=Path("inventario_landsat.csv"))
    args = parser.parse_args()
    if args.start_year > args.end_year or any(not 1 <= m <= 12 for m in args.months):
        parser.error("Invalid year range or months")
    if not 0 <= args.min_coverage <= 1:
        parser.error("Coverage must be between 0 and 1")

    import ee

    ee.Initialize(project=args.project)
    geojson = json.loads(args.aoi.read_text(encoding="utf-8"))
    aoi = ee.Geometry(geojson["geometry"] if geojson["type"] == "Feature" else geojson)
    land = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").neq(80)
    collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    ).filterBounds(aoi).filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
    excluded_bits = sum(1 << bit for bit in (0, 1, 2, 3, 4, 7))

    def coverage(image):
        valid = (image.select("ST_B10").gt(0)
                 .And(image.select("QA_PIXEL").bitwiseAnd(excluded_bits).eq(0))
                 .unmask(0, False).updateMask(land).rename("valid"))
        fraction = valid.reduceRegion(
            ee.Reducer.mean(), aoi, 90, crs="EPSG:32616", maxPixels=1_000_000
        ).get("valid")
        return image.set("coverage", fraction)

    rows = []
    for year in range(args.start_year, args.end_year + 1):
        for month in sorted(set(args.months)):
            start = ee.Date.fromYMD(year, month, 1)
            images = collection.filterDate(start, start.advance(1, "month")).map(coverage)
            rows.extend(images.reduceColumns(
                ee.Reducer.toList(3), ["system:id", "system:time_start", "coverage"]
            ).get("list").getInfo())
    inventory = pd.DataFrame(rows, columns=["id_escena", "timestamp_ms", "fraccion_terrestre"])
    inventory["fecha"] = pd.to_datetime(inventory.timestamp_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    inventory = inventory.sort_values(["fecha", "fraccion_terrestre", "id_escena"], ascending=[True, False, True])
    inventory["seleccionada"] = (
        ~inventory.fecha.duplicated() & inventory.fraccion_terrestre.ge(args.min_coverage)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.output, index=False)
    print(f"{len(inventory)} candidates, {inventory.seleccionada.sum()} selected dates")


if __name__ == "__main__":
    main()
