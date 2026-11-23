"""Build a reusable Landsat inventory and processed thermal panel."""
import argparse
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
import requests

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--proyecto-ee", required=True)
parser.add_argument("--desde", type=int, default=2013)
parser.add_argument("--hasta", required=True, help="Fecha final exclusiva YYYY-MM-DD.")
parser.add_argument("--salida", type=Path, default=Path("data/serie_ampliada"))
parser.add_argument("--solo-inventario", action="store_true")
args = parser.parse_args()
salida = args.salida
hasta = pd.Timestamp(args.hasta).normalize()
anio_inicio, anio_fin = args.desde, hasta.year
if anio_inicio < 2013 or hasta <= pd.Timestamp(anio_inicio, 1, 1):
    parser.error("Indica un rango válido a partir de 2013.")

ee.Initialize(project=args.proyecto_ee)
bits_excluir = sum(1 << bit for bit in (0, 1, 2, 3, 4, 7))

import geopandas as gpd
from shapely.geometry import Point, mapping

url_limites = "https://www.geoboundaries.org/api/current/gbOpen/NIC/ADM2/"
respuesta = requests.get(url_limites, timeout=60)
respuesta.raise_for_status()

municipios = gpd.read_file(
    respuesta.json()["gjDownloadURL"]
).to_crs("EPSG:4326")

punto_centro = Point(-86.2730, 12.1364)
managua = municipios.loc[municipios.geometry.covers(punto_centro)].copy()

if len(managua) != 1:
    raise ValueError(f"El punto coincidió con {len(managua)} municipios.")

area_managua = ee.Geometry(mapping(managua.geometry.iloc[0]))

print("Nombre en la fuente:", managua["shapeName"].iloc[0])
print("Límites:", managua.total_bounds.round(4))
landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(
    ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
)

termicas_managua = (
    landsat.filterBounds(area_managua)
    .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
)


cobertura_suelo = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
tierra = cobertura_suelo.neq(80)

# Reutilizamos las 16 consultas mensuales y guardamos los resultados
# en otra carpeta, porque cambió el denominador.
carpeta_tierra = salida / "inventario"
carpeta_tierra.mkdir(parents=True, exist_ok=True)


def agregar_cobertura_terrestre(imagen):
    termica = imagen.select("ST_B10")
    qa = imagen.select("QA_PIXEL")

    valido = (
        termica.gt(0)
        .And(qa.bitwiseAnd(bits_excluir).eq(0))
        .unmask(0, False)
        .updateMask(tierra)
        .rename("valido")
    )

    cobertura = valido.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=area_managua,
        crs="EPSG:32616",
        scale=90,
        maxPixels=1_000_000,
    ).get("valido")

    return imagen.set("cobertura_terrestre", cobertura)


archivos_tierra = []

for anio in range(anio_inicio, anio_fin + 1):
    for mes in range(1, 13):
        inicio_mes = pd.Timestamp(anio, mes, 1)
        fin_mes = inicio_mes + pd.offsets.MonthBegin(1)

        if inicio_mes >= pd.Timestamp(hasta):
            continue
        fin_consulta = min(fin_mes, hasta)
        archivo = carpeta_tierra / f"cobertura_{anio}_{mes:02d}_hasta_{fin_consulta:%Y%m%d}.csv"
        archivos_tierra.append(archivo)
        if archivo.exists():
            continue

        inicio = ee.Date.fromYMD(anio, mes, 1)
        escenas = (
            termicas_managua.filterDate(inicio, ee.Date(fin_consulta.strftime("%Y-%m-%d")))
            .map(agregar_cobertura_terrestre)
        )

        registros = escenas.reduceColumns(
            reducer=ee.Reducer.toList(3),
            selectors=["system:id", "system:time_start", "cobertura_terrestre"],
        ).get("list").getInfo()

        mensual = pd.DataFrame(
            registros or [],
            columns=["id_escena", "tiempo_ms", "fraccion_terrestre"],
        )
        mensual["anio"] = anio
        mensual["mes"] = mes
        temporal = archivo.with_suffix(".part")
        mensual.to_csv(temporal, index=False)
        temporal.replace(archivo)
        print(f"{anio}-{mes:02d} | {len(mensual)} escenas evaluadas")

inventario_tierra = pd.concat(
    [pd.read_csv(archivo) for archivo in archivos_tierra],
    ignore_index=True,
)

inventario_tierra["fecha"] = pd.to_datetime(
    inventario_tierra["tiempo_ms"], unit="ms", utc=True
).dt.strftime("%Y-%m-%d")

mejor_tierra_por_fecha = (
    inventario_tierra.sort_values("fraccion_terrestre", ascending=False, kind="stable")
    .drop_duplicates("fecha")
)


mejor_tierra_por_fecha.to_csv(salida / "inventario_por_fecha.csv", index=False)
if args.solo_inventario:
    raise SystemExit(0)

predictores = pd.read_csv(
    "resultados/cierre_managua/celdas_e_historial_v1.csv",
    usecols=["id_celda", "fraccion_arboles", "fraccion_construida", "elevacion_media_m"],
)

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from shapely.geometry import mapping

seleccion = mejor_tierra_por_fecha.loc[
    mejor_tierra_por_fecha["fraccion_terrestre"].ge(0.70)
].sort_values("fecha")

if seleccion.empty:
    raise ValueError("No hay escenas con cobertura terrestre ≥ 70 %.")

carpeta = salida / "escenas"
carpeta.mkdir(parents=True, exist_ok=True)

geometria_utm = mapping(managua.to_crs("EPSG:32616").geometry.iloc[0])
region = area_managua.bounds()
bits_excluir = sum(1 << bit for bit in (0, 1, 2, 3, 4, 7))
mascara_tierra = tierra.unmask(0).rename("TIERRA")


def descargar_geotiff(imagen, fecha):
    for intento in range(4):
        try:
            url = imagen.getDownloadURL({
                "region": region,
                "scale": 30,
                "crs": "EPSG:32616",
                "format": "GEO_TIFF",
            })
            respuesta = requests.get(url, timeout=(15, 120))
            respuesta.raise_for_status()
            return respuesta.content

        except requests.HTTPError as error:
            if error.response.status_code not in {429, 500, 502, 503, 504}:
                raise

        except (requests.ConnectionError, requests.Timeout):
            pass

        if intento < 3:
            espera = 2 ** (intento + 1)
            print(f"{fecha} | reintento de descarga en {espera} s")
            time.sleep(espera)

    raise RuntimeError(f"{fecha}: descarga pendiente tras cuatro intentos.")


archivos = []
pendientes = []

for escena in seleccion.itertuples(index=False):
    archivo = carpeta / f"celdas_{escena.id_escena.rsplit('/', 1)[-1]}.csv"

    if archivo.exists():
        archivos.append(archivo)
        print(f"{escena.fecha} | reutilizada")
        continue

    imagen = (
        ee.Image(escena.id_escena)
        .select(["ST_B10", "QA_PIXEL"])
        .addBands(mascara_tierra)
        .toUint16()
    )

    try:
        contenido = descargar_geotiff(imagen, escena.fecha)
    except RuntimeError as error:
        print(error)
        pendientes.append(escena.fecha)
        continue

    with MemoryFile(contenido) as memoria:
        with memoria.open() as src:
            termica, qa, es_tierra = src.read()
            transformacion = src.transform

    dentro = geometry_mask(
        [geometria_utm],
        out_shape=termica.shape,
        transform=transformacion,
        invert=True,
    )

    filas, columnas = np.nonzero(dentro & (es_tierra == 1))
    x, y = transformacion * (columnas + 0.5, filas + 0.5)

    dn = termica[filas, columnas]
    calidad = qa[filas, columnas]
    validos = (dn > 0) & ((calidad & bits_excluir) == 0)

    temperatura = np.full(dn.shape, np.nan, dtype="float32")
    temperatura[validos] = dn[validos] * 0.00341802 + 149.0 - 273.15

    pixeles = pd.DataFrame({
        "col_celda": np.floor(x / 250).astype(int),
        "fila_celda": np.floor(y / 250).astype(int),
        "temperatura_c": temperatura,
    })

    resumen = (
        pixeles.groupby(["col_celda", "fila_celda"], as_index=False)
        .agg(
            temperatura_mediana_c=("temperatura_c", "median"),
            n_pixeles_tierra=("temperatura_c", "size"),
            n_pixeles_validos=("temperatura_c", "count"),
        )
    )

    resumen["fraccion_valida"] = (
        resumen["n_pixeles_validos"] / resumen["n_pixeles_tierra"]
    )
    resumen["id_celda"] = (
        resumen["col_celda"].astype(str) + "_" + resumen["fila_celda"].astype(str)
    )
    resumen["fecha"] = escena.fecha
    resumen["id_escena"] = escena.id_escena
    resumen["anio"] = escena.anio
    resumen["mes"] = escena.mes

    temporal = archivo.with_suffix(".part")
    resumen.to_csv(temporal, index=False)
    temporal.replace(archivo)
    archivos.append(archivo)

    print(f"{escena.fecha} | {validos.mean():.1%} de tierra con datos válidos")

if archivos:
    panel_multianual = pd.concat(
        [pd.read_csv(archivo) for archivo in archivos],
        ignore_index=True,
    )
    ruta_panel = salida / "panel_ampliado.csv.gz"
    panel_multianual = panel_multianual.loc[panel_multianual["fraccion_valida"].ge(0.70)].copy()
    panel_multianual = panel_multianual.merge(
        predictores, on="id_celda", how="left", validate="many_to_one"
    )
    panel_multianual.to_csv(ruta_panel, index=False)
    print(f"\nFechas guardadas: {panel_multianual['fecha'].nunique()} de {len(seleccion)}")
    print("Archivo:", ruta_panel.resolve())

print("Fechas pendientes:", pendientes)
if pendientes:
    raise SystemExit(1)
