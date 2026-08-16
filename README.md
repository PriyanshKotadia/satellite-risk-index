# Satellite Risk Index

Multi-Constellation Collision Risk Score Estimator — a spatial-temporal ML model estimating satellite conjunction risk in LEO.

## Overview
This project aims to estimate collision risks and congestion in Low Earth Orbit (LEO) using machine learning techniques on satellite trajectory data.

## Data Source
Kaggle dataset (link TBD)

## Architecture
- **Data Prep** -> **Feature Engineering** -> **LightGBM Model** -> **Static Predictions** -> **deck.gl 3D Globe Frontend**

## Repo Structure
```
satellite-risk-index/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── models/
├── web/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── assets/
├── api/
│   └── main.py
├── docs/
│   ├── methodology.md
│   ├── data_dictionary.md
│   └── model_card.md
└── .github/
    └── workflows/
        └── ci.yml
```

## Setup Instructions
```bash
pip install -r requirements.txt
```

**Status: in progress**
