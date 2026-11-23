"""Evaluate chronological folds using one scene-level and one spatial model."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "temperatura_mediana_c"
STATIC = ["fraccion_arboles", "fraccion_construida", "elevacion_media_m"]
CALENDAR = ["seno_anual", "coseno_anual"]


def ridge(alpha):
    return make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=alpha))


def run(panel, weather, min_scenes):
    panel = panel.sort_values(["fecha", "id_celda"]).reset_index(drop=True)
    if panel.duplicated(["fecha", "id_celda"]).any():
        raise ValueError("El panel debe tener una observación por fecha y celda.")
    dates = sorted(panel["fecha"].unique())
    if len(dates) <= min_scenes:
        raise ValueError("Se necesitan más fechas que el mínimo de ajuste.")

    angle = 2 * np.pi * (panel["fecha"].dt.dayofyear - 1) / 365.2425
    panel["seno_anual"], panel["coseno_anual"] = np.sin(angle), np.cos(angle)
    if weather is not None:
        if TARGET in weather or "fecha" not in weather:
            raise ValueError("El CSV meteorológico debe incluir fecha y predictores exógenos.")
        columns = list(weather.columns.drop("fecha"))
        if not columns:
            raise ValueError("El CSV meteorológico no contiene predictores.")
        panel = panel.merge(weather, on="fecha", how="left", validate="many_to_one")
        if panel[columns].isna().any().any():
            raise ValueError("Faltan predictores meteorológicos en alguna fecha.")
    else:
        columns = []

    daily = panel.groupby("fecha")[TARGET].agg(["sum", "size"])
    previous_global = (daily["sum"].cumsum() / daily["size"].cumsum()).shift()
    cells = panel.groupby("id_celda", sort=False)[TARGET]
    previous_n = cells.cumcount()
    prior = (cells.cumsum() - panel[TARGET]) / previous_n.where(previous_n.gt(0))
    panel["historial"] = prior.fillna(panel["fecha"].map(previous_global))
    panel["residuo"] = panel[TARGET] - panel["historial"]

    results = []
    for date in dates[min_scenes:]:
        train = panel.loc[panel["fecha"].lt(date) & panel["historial"].notna()].copy()
        test = panel.loc[panel["fecha"].eq(date)].copy()
        daily_train = train.groupby("fecha", as_index=False).agg(
            nivel=("residuo", "mean"),
            **{name: (name, "first") for name in CALENDAR + columns},
        )
        train["patron"] = train["residuo"] - train.groupby("fecha")["residuo"].transform("mean")
        spatial = ridge(1.0).fit(train[STATIC], train["patron"]).predict(test[STATIC])
        predictors = ["historial", *CALENDAR, *STATIC]
        direct = ridge(1.0).fit(train[predictors], train[TARGET]).predict(test[predictors])
        versions = {"historial": test["historial"].to_numpy(), "ridge_directo": direct}
        for label, features in [("componentes_calendario", CALENDAR),
                                ("componentes_meteo", CALENDAR + columns)]:
            if label.endswith("meteo") and not columns:
                continue
            level = ridge(3.0).fit(daily_train[features], daily_train["nivel"])
            correction = level.predict(test[features].iloc[:1])[0]
            versions[label] = test["historial"].to_numpy() + spatial + correction
        for name, predicted in versions.items():
            error = predicted - test[TARGET].to_numpy()
            results.append({
                "fecha": date, "anio": pd.Timestamp(date).year, "modelo": name,
                "celdas": len(test), "mae_c": np.abs(error).mean(),
                "sesgo_nivel_c": error.mean(), "error_nivel_c": abs(error.mean()),
                "mae_patron_c": np.abs(error - error.mean()).mean(),
            })
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/predictores/managua_panel_con_predictores.csv.gz")
    parser.add_argument("--meteo", help="CSV con fecha y predictores previos al sobrevuelo.")
    parser.add_argument("--min-escenas", type=int, default=6)
    parser.add_argument("--salida", default="resultados/componentes_por_escena.csv")
    args = parser.parse_args()
    if args.min_escenas < 3:
        parser.error("--min-escenas debe ser al menos 3.")
    panel = pd.read_csv(args.panel, parse_dates=["fecha"], dtype={"id_celda": "string"})
    weather = pd.read_csv(args.meteo, parse_dates=["fecha"]) if args.meteo else None
    results = run(panel, weather, args.min_escenas)
    output = Path(args.salida)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    measures = ["mae_c", "error_nivel_c", "mae_patron_c"]
    print(results.groupby(["modelo", "anio"])[measures].mean().round(3).to_string())


if __name__ == "__main__":
    main()
