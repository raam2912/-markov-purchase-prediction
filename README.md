# Predicting Consumer Purchases with Markov Chains
### Real results on 1M+ rows of e-commerce data · UCI Online Retail II

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/dataset-UCI%20Online%20Retail%20II-orange.svg)](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The question

> *Given a customer's most recent product category, can we predict their next purchase better than "just recommend the most popular items globally"?*

This repo is a complete, reproducible experiment — from raw data download to confidence-interval-backed results.

---

## Results (on held-out test set — last 2 months of data)

| Model | HR@1 | HR@3 | HR@5 |
|-------|------|------|------|
| **Markov Chain (order=1)** | **0.211** | **0.468** | **0.650** |
| Most-Popular Baseline | 0.188 | 0.443 | 0.638 |
| **Lift** | **+2.3pp** | **+2.5pp** | **+1.2pp** |

> 95% bootstrap confidence intervals (1000 samples, seed=42). All lifts are statistically meaningful with non-overlapping CIs at K=1 and K=3.

**Hit Rate @ K** = fraction of test cases where the true next category appeared in the top-K predictions.  
**Baseline** = always predict the globally most-purchased categories (no personalisation).

---

## Dataset

**UCI Online Retail II** — UK-based online retailer, Dec 2009–Dec 2011.

| Metric | Value |
|--------|-------|
| Raw rows | 1,067,371 |
| Clean rows (after pipeline) | 765,822 |
| Unique customers | 5,701 |
| Unique SKUs | 4,623 |
| Date range | 2009-12-01 → 2011-12-09 |
| Product categories | 15 (keyword-rule assigned) |

Download: [https://archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

---

## How it works

### 1. Data Cleaning (8 passes)
- Drop nulls (`CustomerID`, `InvoiceDate`, `StockCode`)
- Remove returns (Invoice starting with `C`, `Quantity < 0`)
- Filter system/test SKUs (`POSTAGE`, `DISCOUNT`, `DOTCOM POSTAGE`, etc.)
- Cap outlier quantities and prices at 99.5th percentile
- Remove near-duplicate transactions (same SKU within 60 seconds)

### 2. Category Assignment
Map 4,623 raw SKU descriptions → 15 product categories using keyword rules:

```
BAGS_STORAGE · CANDLES_LIGHTS · CHILDREN_TOYS · CHRISTMAS
CLOCKS_TIMEPIECE · FRAMES_MIRRORS · GARDEN_OUTDOOR · HOME_DECOR
JEWELLERY_ACCESS · KITCHEN_DINING · SEASONAL_GIFT · SIGNS_PLAQUES
STATIONERY · TEXTILES_SOFT · OTHER
```

This reduces the state space from 4,623 (too sparse) to 15 (fully dense matrix).

### 3. Markov Chain
Build a normalised transition probability matrix:

```
T[i][j] = P(next = j | current = i) = count(i → j) / count(i)
```

Trained on **634,540 transitions** from **5,277 customer sequences**.  
Matrix coverage: **100%** (all 15×15 = 225 cells populated).  
Unseen states fall back to the global unigram distribution.

### 4. Evaluation
- **Temporal split**: train on first 23 months, test on last 2 months (Oct–Nov 2011)
- **Test transitions**: 2,115 (one per customer in both sets)
- **Hit Rate @ K** with 1000-sample bootstrap confidence intervals

---

## Repository structure

```
AntiGravSkills/
├── notebook/
│   └── markov_purchase_prediction.ipynb   ← The publishable artifact
├── backend/
│   ├── models/
│   │   ├── category_encoder.py            ← SKU desc → 15 categories
│   │   ├── markov.py                      ← build_markov() + predict_top_k()
│   │   ├── baseline.py                    ← Most-popular baseline
│   │   └── evaluator.py                   ← HR@K + temporal split + bootstrap CI
│   ├── pipeline/
│   │   ├── ingest.py                      ← Download + load UCI dataset
│   │   ├── validate.py                    ← 10 non-mutating validation checks
│   │   ├── clean.py                       ← 8 ordered cleaning passes
│   │   ├── sequence_builder.py            ← Per-customer sequences
│   │   └── report.py                      ← HTML + JSON quality report
│   ├── run_baseline.py                    ← Week 1: run everything, print results
│   ├── generate_notebook.py               ← Programmatically build the notebook
│   ├── main.py                            ← CLI entry point for data pipeline
│   ├── config.py                          ← Centralised config (Pydantic)
│   └── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/markov-purchase-prediction.git
cd markov-purchase-prediction

# 2. Create venv and install
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r backend/requirements.txt

# 3. Download + clean data (downloads ~45MB UCI zip automatically)
cd backend
python main.py

# 4. Run the full evaluation
python run_baseline.py

# 5. Open the notebook
jupyter notebook ../notebook/markov_purchase_prediction.ipynb
```

---

## Why this approach?

| Choice | Rationale |
|--------|-----------|
| Markov order=1 | Interpretable, fast, strong baseline for sequential recommendation |
| 15 categories | Reduces 4k+ SKUs to a dense, learnable state space |
| Temporal split | No data leakage — future never bleeds into past |
| Hit Rate @ K | Standard metric for top-K recommendation tasks |
| Bootstrap CI | Distinguishes real signal from random variation |

---

## What's next (Phase 2)

- [ ] 2nd-order Markov (state = last 2 categories)
- [ ] Per-segment matrices (RFM customer clusters)
- [ ] XGBoost with Markov probabilities as features
- [ ] Prophet demand forecasting layer
- [ ] REST API (FastAPI) for real-time inference

---

## Citation

Daqing Chen, Sai Liang Sain, and Kun Guo.  
*Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining.*  
Journal of Database Marketing and Customer Strategy Management, 2012.

---

*Part of the AntiGrav Market Prediction Framework.*
