"""
train_model.py
--------------
Entrena el modelo de regresión logística sobre creditcard.csv
y guarda el artefacto en model/model.pkl.

Uso:
    python model/train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

#rutas
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

#Carga el CSV y aplica la limpieza básica del notebook.
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Class"] = df["Class"].astype(str)          # str para EDA
    df = df.drop_duplicates()
    return df


def prepare_features(df: pd.DataFrame):
    df_model = df.copy()
    df_model["Class"] = df_model["Class"].astype("float64")
    X = df_model.drop("Class", axis=1)
    y = df_model["Class"]
    return X, y

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    param_grid = {
        "C": [10],
        "class_weight": [None, "balanced"],
    }

    logreg = LogisticRegression(max_iter=100_000, solver="saga", n_jobs=1)
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    grid   = GridSearchCV(logreg, param_grid, cv=skf, scoring="f1", n_jobs=1)
    grid.fit(X_train, y_train)

    print(f"Mejores parámetros : {grid.best_params_}")
    print(f"Mejor F1 (CV)      : {grid.best_score_:.4f}")

    # Evaluación sobre test
    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:, 1]

    print("\n── Métricas en test      ──")
    print("\n \n \n Test set score: {:.2f}".format(grid.score(X_test, y_test)))
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1        : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte completo:")
    print(classification_report(y_test, y_pred))

    return grid, X_test, y_test, y_prob


def save_model(estimator, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(estimator, path)
    print(f"\nModelo guardado en: {path}")


if __name__ == "__main__":
    print("Cargando datos…")
    df = load_and_clean(DATA_PATH)
    print(f"Registros tras limpieza: {len(df):,}")
    X, y = prepare_features(df)
    print("Entrenando modelo…")
    estimator, X_test, y_test, y_prob = train(X, y)
    save_model(estimator, MODEL_PATH)
