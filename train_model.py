import pickle

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from preprocessing import (
    X_train,
    X_test,
    y_train,
    y_test
)

# =========================
# MODELO
# =========================
model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)

# =========================
# ENTRENAMIENTO
# =========================
print("\nEntrenando modelo...\n")

model.fit(
    X_train,
    y_train
)

# =========================
# PREDICCIONES
# =========================
y_pred = model.predict(X_test)

# =========================
# MÉTRICAS
# =========================
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:")
print(accuracy)

print("\nReporte de clasificación:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================
# GUARDAR MODELO
# =========================
with open(
    "models/logistic_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )

print("\nModelo guardado correctamente")