import pickle
import pandas as pd

# =========================
# CARGAR MODELO
# =========================
with open(
    "models/logistic_model.pkl",
    "rb"
) as file:

    model = pickle.load(file)

# =========================
# NUEVO ESTUDIANTE
# =========================
student_data = {

    "dificultad": [1],

    "tiempo_respuesta": [40],

    "intentos": [2],

    "tipo_factorizacion_diferencia_cuadrados": [0],

    "tipo_factorizacion_factor_comun": [1],

    "tipo_factorizacion_trinomio": [0],

    "tipo_factorizacion_trinomio_cuadrado_perfecto": [0]
}

# =========================
# DATAFRAME
# =========================
X_new = pd.DataFrame(
    student_data
)

# =========================
# PREDICCIÓN
# =========================
prediction = model.predict(
    X_new
)

print("\nPredicción realizada")

print(prediction)