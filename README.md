# Dashboard · Detección de Fraude Financiero

Dashboard interactivo desarrollado en Python + Dash para el análisis y predicción
de fraude en transacciones con tarjeta de crédito, basado en el dataset Credit Card
Fraud Detection (ULB / Kaggle).

---

## Estructura del proyecto

```
fraud_dashboard/
├── app.py                  ← Punto de entrada de la aplicación
├── requirements.txt        ← Dependencias
├── README.md
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py      ← Carga y caché del dataframe
│   └── creditcard.csv      ← Dataset (debes colocarlo aquí)
│
├── model/
│   ├── __init__.py
│   ├── train_model.py      ← Entrenamiento y serialización del modelo
│   └── model.pkl           ← Generado al ejecutar train_model.py
│
├── tabs/
│   ├── __init__.py
│   ├── introduccion.py
│   ├── contexto.py
│   ├── problema.py
│   ├── objetivos.py
│   ├── marco_teorico.py
│   ├── metodologia.py
│   ├── resultados.py
│   ├── prediccion.py
│   ├── limitaciones.py
│   └── conclusiones.py
│
└── assets/
    └── custom.css          ← Estilos corporativos
```

---

## Requisitos previos

- Python 3.10 o 3.11
- El archivo `creditcard.csv` descargado desde Kaggle:
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Instalación paso a paso

### 1. Clonar o descomprimir el proyecto

```bash
cd fraud_dashboard
```

### 2. Crear entorno virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Colocar el dataset

Copia `creditcard.csv` en la carpeta `data/`:

```
fraud_dashboard/
└── data/
    └── creditcard.csv   ← aquí
```

### 5. Entrenar el modelo

```bash
python model/train_model.py
```

Esto generará `model/model.pkl`. Solo es necesario hacerlo una vez.
El proceso puede tardar 2–5 minutos según la capacidad del equipo.

### 6. Ejecutar el dashboard

```bash
python app.py
```

Abre el navegador en: http://127.0.0.1:8050

---

## Pestañas del dashboard

| Pestaña       | Contenido                                                    |
|---------------|--------------------------------------------------------------|
| Introducción  | Contexto del fraude financiero y descripción del dataset     |
| Contexto      | Impacto empresarial, costos y desbalance de clases           |
| Problema      | Distribuciones de fraude, Amount, Time y análisis por hora   |
| Objetivos     | Objetivo general y específicos del proyecto                  |
| Marco teórico | Fundamentos teóricos y tabla de operacionalización           |
| Metodología   | Pipeline del análisis y decisiones metodológicas             |
| Resultados    | EDA completo + métricas del modelo (ROC, CM, F1, etc.)       |
| Predicción    | Formulario interactivo para predicción en tiempo real        |
| Limitaciones  | Restricciones del análisis y del modelo                      |
| Conclusiones  | Hallazgos, resumen del modelo y mejoras futuras              |

---

## Notas técnicas

- El tab **Resultados** y el tab **Predicción** requieren que `model.pkl` exista.
  Si no existe, muestran un aviso y no fallan.
- El dataframe se carga una sola vez en memoria usando `lru_cache` para evitar
  lecturas repetidas en cada callback.
- El umbral de decisión utilizado en Predicción es **0.13**, que maximiza el
  recall sobre la clase fraude según el análisis del notebook original.
- Todos los gráficos son funciones puras que reciben el dataframe como argumento,
  lo que facilita el testing y la reutilización.

---

## Despliegue en producción

Para desplegar con gunicorn:

```bash
pip install gunicorn
gunicorn app:server -b 0.0.0.0:8050 --workers 2
```

---

## Dependencias principales

| Librería              | Versión  | Uso                              |
|-----------------------|----------|----------------------------------|
| dash                  | 2.17.1   | Framework web                    |
| dash-bootstrap-components | 1.6.0 | Componentes UI                |
| plotly                | 5.22.0   | Visualizaciones interactivas     |
| pandas                | 2.2.2    | Manipulación de datos            |
| scikit-learn          | 1.5.0    | Modelo y métricas                |
| joblib                | 1.4.2    | Serialización del modelo         |
| pingouin              | 0.5.4    | Prueba Mann-Whitney U + RBC      |
| statsmodels           | 0.14.2   | VIF                              |
| scipy                 | 1.13.1   | Estadística                      |
