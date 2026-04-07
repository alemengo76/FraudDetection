"""
data_loader.py
--------------
Carga única del dataset con limpieza aplicada.
Expone get_df() con caché en memoria para no leer el CSV en cada callback.
"""

import os
import functools
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")


@functools.lru_cache(maxsize=1)
def get_df() -> pd.DataFrame:
    """
    Devuelve el DataFrame limpio (sin duplicados, Class como str).
    El resultado queda en memoria tras la primera llamada.
    """
    df = pd.read_csv(DATA_PATH)
    df["Class"] = df["Class"].astype(str)
    df = df.drop_duplicates()
    return df
