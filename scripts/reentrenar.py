from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

carpeta = Path("resultados/cierre_managua")
carpeta.mkdir(parents=True, exist_ok=True)

ruta_panel = Path("data/predictores/managua_panel_con_predictores.csv.gz")
datos = pd.read_csv(ruta_panel, parse_dates=["fecha"], dtype={"id_celda": "string"})
objetivo = "temperatura_mediana_c"

datos = (
    datos.loc[
        datos["fecha"].dt.year.between(2023, 2026)
        & datos["fraccion_valida"].ge(0.70)
        & datos[objetivo].notna()
    ]
    .sort_values(["fecha", "id_celda"])
    .reset_index(drop=True)
)

if datos["fecha"].nunique() != 18 or datos.duplicated(["id_celda", "fecha"]).any():
    raise ValueError("Esperaba 18 escenas y una fila por celda y fecha.")

estaticas = ["fraccion_arboles", "fraccion_construida", "elevacion_media_m"]
variables_finales = ["media_previa_c", "n_previas", "dia_del_anio", *estaticas]

por_celda = datos.groupby("id_celda", sort=False)
datos["n_previas"] = por_celda.cumcount()
suma_previa = por_celda[objetivo].cumsum() - datos[objetivo]
datos["media_previa_c"] = suma_previa.div(datos["n_previas"].where(datos["n_previas"].gt(0)))
datos["dia_del_anio"] = datos["fecha"].dt.dayofyear

modelo_final = make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    StandardScaler(),
    Ridge(alpha=1.0),
)
modelo_final.fit(datos[variables_finales], datos[objetivo])

historial_final = por_celda[objetivo].agg(media_previa_c="mean", n_previas="size")
columnas_celda = ["id_celda", "col_celda", "fila_celda", *estaticas]
celdas_finales = (
    datos[columnas_celda]
    .drop_duplicates()
    .merge(historial_final, left_on="id_celda", right_index=True, validate="one_to_one")
    .sort_values("id_celda")
)

version = "ridge_managua_2023_2026_v1"
paquete = {
    "modelo": modelo_final,
    "variables": variables_finales,
    "version": version,
    "sklearn_version": sklearn.__version__,
    "ultima_escena": datos["fecha"].max().date().isoformat(),
}

joblib.dump(paquete, carpeta / "ridge_managua_v1.joblib")
celdas_finales.to_csv(carpeta / "celdas_e_historial_v1.csv", index=False)

print(f"Entrenamiento final: {len(datos):,} filas, {datos['fecha'].nunique()} escenas")
print(f"Celdas con historial: {len(celdas_finales):,}")
print("Última escena usada:", paquete["ultima_escena"])
print("Guardado en:", carpeta.resolve())
