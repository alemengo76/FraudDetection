# fake_model.py
import joblib

class FakeModel:
    def predict(self, X):
        return [0] * len(X)  # siempre "no fraude"

    def predict_proba(self, X):
        return [[0.99, 0.01] for _ in range(len(X))]

joblib.dump(FakeModel(), "model/model.pkl")