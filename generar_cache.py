import joblib, json, numpy as np, pandas as pd
from sklearn.metrics import (average_precision_score, roc_curve, auc, precision_recall_curve,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from pathlib import Path

RANDOM_STATE = 42

# 1. Carga el CSV IGUAL que en el notebook
df = pd.read_csv(r"C:\Users\Alejandra\Documents\fraud_dashboard\data\creditcard.csv")

# ✅ CRÍTICO: eliminar duplicados, igual que el notebook hace antes de entrenar
df = df.drop_duplicates()

# ✅ CRÍTICO: Class como int64, igual que el notebook hace antes de train_test_split
df['Class'] = df['Class'].astype('int64')

# 2. Reconstruye el split EXACTAMENTE igual que en el notebook
x = df.drop("Class", axis=1)
y = df["Class"]

_, X_test, _, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# 3. Carga el modelo y predice
pipeline = joblib.load(
    r"C:\Users\Alejandra\Documents\Modelos_Optimizados\optuna_XGBoost_ADASYN.joblib"
)

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# 4. Métricas
fpr, tpr, _   = roc_curve(y_test, y_prob)
pr_p, pr_r, _ = precision_recall_curve(y_test, y_prob)
cm            = confusion_matrix(y_test, y_pred).tolist()
rep           = classification_report(y_test, y_pred, output_dict=True)

# ✅ El paso del XGBoost se llama "modelo" en tu pipeline
xgb        = pipeline.named_steps["modelo"]
feat_names = list(X_test.columns)
feat_imp   = xgb.feature_importances_.tolist()

# Sweep de umbrales
thr_vals = np.arange(0.1, 1.0, 0.1)
t_p, t_r, t_f = [], [], []
for t in thr_vals:
    yp = (y_prob >= t).astype(int)
    r2 = classification_report(y_test, yp, output_dict=True, zero_division=0)
    t_p.append(r2.get("1", {}).get("precision", 0))
    t_r.append(r2.get("1", {}).get("recall",    0))
    t_f.append(r2.get("1", {}).get("f1-score",  0))

# 5. Guarda el JSON
Path(r"C:\Users\Alejandra\Documents\fraud_dashboard\assets").mkdir(exist_ok=True)
with open(r"C:\Users\Alejandra\Documents\fraud_dashboard\assets\metricas_cache.json", "w") as f:
    json.dump({
        "accuracy":           rep["accuracy"],
        "auc":                float(auc(fpr, tpr)),
        "f1":                 rep["1"]["f1-score"],
        "precision":          rep["1"]["precision"],
        "recall":             rep["1"]["recall"],
        "specificity":        rep["0"]["recall"],
        "cm":                 cm,
        "fpr":                fpr.tolist(),
        "tpr":                tpr.tolist(),
        "pr_precision":       pr_p.tolist(),
        "pr_recall":          pr_r.tolist(),
        "ap": float(average_precision_score(y_test, y_prob)),
        "feature_names":      feat_names,
        "feature_importance": feat_imp,
        "thresholds":         thr_vals.tolist(),
        "thresh_precision":   t_p,
        "thresh_recall":      t_r,
        "thresh_f1":          t_f,
    }, f, indent=2)

print("✅ metricas_cache.json generado correctamente.")
print(f"   Test set: {len(X_test):,} filas ({y_test.sum()} fraudes)")
print(f"   Accuracy: {rep['accuracy']:.4f}")
print(f"   F1 (fraude): {rep['1']['f1-score']:.4f}")
print(f"   AUC-ROC: {float(auc(fpr, tpr)):.4f}")
