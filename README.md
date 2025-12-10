# Proyecto ML — Predict [X]

## 🏗 Arquitectura

Este proyecto usa **Deploy Code**, donde todo el pipeline ML se gobierna por código versionado en GitHub.

El flujo completo es:
1. Ingest
2. Preparación
3. Entrenamiento del modelo
4. Logging en MLflow
5. Batch Inference

Todo el código se despliega automáticamente a Databricks usando GitHub Actions.

## 🚀 Cómo reproducir localmente

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py
```

```
my-ml-project/
│
├── README.md
├── requirements.txt
├── configs/
│    └── config.yaml
├── src/
│    ├── ingest.py
│    ├── prep.py
│    ├── train.py
│    ├── predict.py
│    ├── utils/
│    │     ├── io.py
│    │     └── metrics.py
├── notebooks/
│    ├── 00_eda.ipynb
│    └── 01_dev_playground.ipynb
├── jobs/
│    ├── job_train.json
│    └── job_inference.json
├── tests/
│    ├── test_prep.py
│    ├── test_train.py
│    └── test_predict.py
└── .github/
     └── workflows/
         └── ci_cd.yaml

```