import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================
# CARGA DEL DATASET
# =========================
df = pd.read_csv(
    "dataset/cocomath_dataset_50000.csv"
)

# =========================
# MOSTRAR INFORMACIÓN
# =========================
print("\nDATASET")
print(df.head())

# =========================
# VARIABLES DE ENTRADA
# =========================
X = df[
    [
        "tipo_factorizacion",
        "dificultad",
        "tiempo_respuesta",
        "intentos"
    ]
]

# =========================
# VARIABLE OBJETIVO
# =========================
y = df["tipo_error"]

# =========================
# CONVERTIR VARIABLES
# =========================
X = pd.get_dummies(
    X,
    columns=["tipo_factorizacion"]
)

# =========================
# CODIFICAR CLASES
# =========================
encoder = LabelEncoder()

y = encoder.fit_transform(y)

# =========================
# TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# INFORMACIÓN
# =========================
print("\nPreprocesamiento completado")

print("\nTrain:")
print(X_train.shape)

print("\nTest:")
print(X_test.shape)