"""Compare historical, direct Ridge and scene-plus-cell Ridge by future year."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "temperatura_mediana_c"
STATIC = ["fraccion_arboles", "fraccion_construida", "elevacion_media_m"]


def historical_features(train, test, target, feature):
    groups = train.groupby("id_celda")[target]
    count = groups.cumcount()
    train = train.assign(**{feature: (groups.cumsum() - train[target]) / count.where(count > 0)})
    test = test.join(groups.mean().rename(feature), on="id_celda")
    return train, test


def ridge_predict(train, test, features, target):
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge())
    return model.fit(train[features], train[target]).predict(test[features])


def evaluate(panel, weather=None, min_training_years=2):
    valid = panel.loc[panel.fraccion_valida.ge(0.70) & panel[TARGET].notna()].copy()
    valid["fecha"] = pd.to_datetime(valid.fecha)
    valid["year"] = valid.fecha.dt.year
    valid["day"] = valid.fecha.dt.dayofyear
    if valid.duplicated(["fecha", "id_celda"]).any():
        raise ValueError("Expected one observation per cell and date.")
    levels = valid.groupby("fecha", as_index=False).agg(
        level=(TARGET, "mean"), day=("day", "first"), year=("year", "first")
    )
    features = ["day"]
    if weather is not None:
        weather = weather.assign(fecha=pd.to_datetime(weather.fecha))
        levels = levels.merge(weather, on="fecha", how="left", validate="one_to_one")
        features += [column for column in weather if column != "fecha"]
        if levels[features].isna().any().any():
            raise ValueError("Weather must cover every evaluated scene.")
    valid = valid.join(levels.set_index("fecha").level, on="fecha")
    valid["anomaly"] = valid[TARGET] - valid.level
    records = []
    for year in sorted(valid.year.unique())[min_training_years:]:
        train = valid.loc[valid.year.lt(year)]
        test = valid.loc[valid.year.eq(year)]
        previous, future = historical_features(train, test, TARGET, "history")
        previous = previous.assign(n_previas=previous.groupby("id_celda").cumcount())
        counts = train.groupby("id_celda").size()
        future = future.assign(n_previas=future.id_celda.map(counts).fillna(0))
        history = future.history.fillna(train[TARGET].mean())
        direct = ridge_predict(previous, future, ["history", "n_previas", "day", *STATIC], TARGET)
        previous, future = historical_features(previous, future, "anomaly", "anomaly_history")
        scene_train = levels.loc[levels.year.lt(year)]
        scene_test = levels.loc[levels.year.eq(year)].copy()
        scene_test["level_pred"] = ridge_predict(scene_train, scene_test, features, "level")
        future = future.join(scene_test.set_index("fecha").level_pred, on="fecha")
        spatial = ridge_predict(previous, future, ["anomaly_history", *STATIC], "anomaly")
        result = future.assign(history_pred=history, direct_pred=direct, spatial_pred=spatial)
        for date, scene in result.groupby("fecha"):
            records.append({
                "fecha": date.date(), "year": year, "n_cells": len(scene),
                "mae_history": (scene[TARGET] - scene.history_pred).abs().mean(),
                "mae_ridge": (scene[TARGET] - scene.direct_pred).abs().mean(),
                "mae_two_components": (scene[TARGET] - scene.level_pred - scene.spatial_pred).abs().mean(),
                "error_scene_level": (scene.level_pred - scene.level).iloc[0],
                "mae_spatial_anomaly": (scene.anomaly - scene.spatial_pred).abs().mean(),
            })
    if not records:
        raise ValueError("No test years remain after the minimum training years.")
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/predictores/managua_panel_con_predictores.csv.gz")
    parser.add_argument("--weather", help="CSV with fecha and numeric features available at prediction time")
    parser.add_argument("--min-training-years", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("metricas_validacion_temporal.csv"))
    args = parser.parse_args()
    if args.min_training_years < 1:
        parser.error("--min-training-years must be positive")
    weather = pd.read_csv(args.weather) if args.weather else None
    metrics = evaluate(pd.read_csv(args.panel), weather, args.min_training_years)
    metrics.to_csv(args.output, index=False)
    print(metrics.round(3).to_string(index=False))
    print(metrics.groupby("year").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
