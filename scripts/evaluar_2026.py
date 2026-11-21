import pandas as pd

def display(tabla):
    print(tabla.to_string())

panel_modelo = pd.read_csv("data/predictores/managua_panel_con_predictores.csv.gz")

import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

objetivo = "temperatura_mediana_c"

datos = panel_modelo.copy()
datos["fecha"] = pd.to_datetime(datos["fecha"])
datos = datos.sort_values(["fecha", "id_celda"]).reset_index(drop=True)

if datos.duplicated(["id_celda", "fecha"]).any():
    raise ValueError("Hay más de una fila para alguna celda y fecha.")

datos["dia_del_anio"] = datos["fecha"].dt.dayofyear
train_temp = datos.loc[datos["anio"].lt(2026)].copy()
test_temp = datos.loc[datos["anio"].eq(2026)].copy()

# Para cada fila de entrenamiento, usamos únicamente escenas anteriores.
por_celda = train_temp.groupby("id_celda", sort=False)
train_temp["n_previas"] = por_celda.cumcount()
suma_previa = por_celda[objetivo].cumsum() - train_temp[objetivo]
train_temp["media_previa_c"] = suma_previa.div(
    train_temp["n_previas"].where(train_temp["n_previas"].gt(0))
)

# El historial de prueba queda congelado al cierre de 2025.
historial_temp = train_temp.groupby("id_celda", as_index=False).agg(
    media_previa_c=(objetivo, "mean"),
    n_previas=(objetivo, "size"),
)
test_temp = test_temp.merge(
    historial_temp, on="id_celda", how="left", validate="many_to_one"
)
test_temp["n_previas"] = test_temp["n_previas"].fillna(0)

variables = [
    "media_previa_c",
    "n_previas",
    "dia_del_anio",
    "fraccion_arboles",
    "fraccion_construida",
    "elevacion_media_m",
]

x_train = train_temp[variables]
y_train = train_temp[objetivo]
x_test = test_temp[variables]

# Referencia histórica. Si la celda nunca se observó, usamos la media de entrenamiento.
test_temp["pred_historial_c"] = test_temp["media_previa_c"].fillna(y_train.mean())

ridge = make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    StandardScaler(),
    Ridge(alpha=1.0),
)
ridge.fit(x_train, y_train)
test_temp["pred_ridge_c"] = ridge.predict(x_test)

imputador = SimpleImputer(strategy="median", add_indicator=True)
x_train_ngb = imputador.fit_transform(x_train)
x_test_ngb = imputador.transform(x_test)

ngb = NGBRegressor(
    Dist=Normal,
    Score=LogScore,
    Base=DecisionTreeRegressor(max_depth=2, min_samples_leaf=150),
    n_estimators=300,
    learning_rate=0.03,
    random_state=42,
    verbose=False,
)
ngb.fit(x_train_ngb, y_train.to_numpy())

distribucion = ngb.pred_dist(x_test_ngb)
test_temp["pred_ngboost_c"] = distribucion.params["loc"]
test_temp["desvio_ngboost_c"] = distribucion.params["scale"]
test_temp["limite_inferior_90_c"] = distribucion.ppf(0.05)
test_temp["limite_superior_90_c"] = distribucion.ppf(0.95)

# CRPS de la distribución normal. Menor es mejor.
z = (
    (test_temp[objetivo] - test_temp["pred_ngboost_c"])
    / test_temp["desvio_ngboost_c"]
)
test_temp["crps_ngboost_c"] = test_temp["desvio_ngboost_c"] * (
    z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / (3.141592653589793 ** 0.5)
)

filas = []
for fecha, escena in test_temp.groupby("fecha", sort=True):
    y = escena[objetivo]
    fila = {"fecha": fecha.date(), "celdas": len(escena)}

    for modelo in ("historial", "ridge", "ngboost"):
        pred = escena[f"pred_{modelo}_c"]
        fila[f"MAE_{modelo}_c"] = mean_absolute_error(y, pred)
        fila[f"RMSE_{modelo}_c"] = mean_squared_error(y, pred) ** 0.5

    inferior = escena["limite_inferior_90_c"]
    superior = escena["limite_superior_90_c"]
    fila["cobertura_90_pct"] = 100 * (y.ge(inferior) & y.le(superior)).mean()
    fila["ancho_intervalo_90_c"] = (superior - inferior).mean()
    fila["CRPS_ngboost_c"] = escena["crps_ngboost_c"].mean()
    filas.append(fila)

metricas_temp = pd.DataFrame(filas)

display(metricas_temp.round(3))
display(metricas_temp.drop(columns=["fecha", "celdas"]).mean().round(3))
