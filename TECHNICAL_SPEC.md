# ⚙️ TECHNICAL SPECIFICATION — AntiGrav Market Prediction Framework
*Reference document. Update when technical decisions are finalized.*

---

## 1. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    BROWSER FRONTEND                          │
│  AntiGrav React App (Vite + TypeScript)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ NN Visualizer│  │ Market Dash  │  │ Forecast View   │   │
│  │ (existing)   │  │ (Phase 4)    │  │ (Phase 4)       │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │ REST API / WebSocket
┌─────────────────────────────▼───────────────────────────────┐
│                    PYTHON BACKEND (FastAPI)                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │ Data Ingest │  │ Markov Engine│  │ Forecast Engine │    │
│  │ /api/load   │  │ /api/markov  │  │ /api/forecast   │    │
│  └─────────────┘  └──────────────┘  └─────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Iterative Update Scheduler                │    │
│  │  Cron job / trigger → re-train → push new weights   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                    DATA LAYER                                │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │ Raw CSV     │  │ SQLite / PG  │  │ Model Artifacts │    │
│  │ Datasets    │  │ (cleaned)    │  │ (.pkl, .pt)     │    │
│  └─────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. DATA PIPELINE SPEC

### 2.1 Preprocessing Steps
```
Raw CSV
  → Load with pandas
  → Drop nulls (CustomerID, InvoiceDate, StockCode, Quantity)
  → Remove returns (Quantity < 0 or InvoiceNo starts with 'C')
  → Normalize timestamps to UTC
  → Sort by (CustomerID ASC, InvoiceDate ASC)
  → Group: per-customer purchase sequences
  → Encode: product category IDs (label encode or hash)
  → Output: cleaned DataFrame + sequence dict
```

### 2.2 Output Schema (per customer sequence)
```json
{
  "customer_id": "12345",
  "segment": "B2B_UK",
  "sequences": [
    {
      "timestamp": "2010-12-01T08:26:00Z",
      "category": "DECORATIVE",
      "sku": "85123A",
      "quantity": 6,
      "price": 2.55
    }
  ]
}
```

---

## 3. MARKOV CHAIN MODEL

### 3.1 First-Order Markov
```python
# Transition matrix: P[state_i][state_j] = count(i→j) / count(i)
T = defaultdict(lambda: defaultdict(int))
for seq in sequences:
    for i in range(len(seq) - 1):
        T[seq[i]][seq[i+1]] += 1
# Normalize rows
for state in T:
    total = sum(T[state].values())
    for next_state in T[state]:
        T[state][next_state] /= total
```

### 3.2 Nth-Order Markov
- State = tuple of last N categories
- Trade-off: recall vs. sparsity
- N=2 recommended starting point

### 3.3 Per-Segment Training
- Segment customers by: recency, frequency, monetary (RFM)
- Train separate transition matrices per segment
- Fallback: global matrix if segment too sparse

---

## 4. ACCURACY EVALUATION

### 4.1 Temporal Split
```
Data sorted by date
  → 80% earliest orders = TRAIN
  → 20% most recent orders = TEST
  (Never shuffle — respect time order!)
```

### 4.2 Hit Rate @ K
```python
def hit_rate_at_k(predictions, actual, k=5):
    top_k = sorted(predictions, key=predictions.get, reverse=True)[:k]
    return 1 if actual in top_k else 0
```

### 4.3 Perplexity
```python
import numpy as np
def perplexity(probs):
    return np.exp(-np.mean(np.log(probs + 1e-9)))
```

### 4.4 Baseline Comparison
- Most-popular-item baseline (always predict top-N most bought)
- Markov must beat this to be meaningful

---

## 5. DEMAND FORECASTING LAYER

### 5.1 Feature Engineering
| Feature | Source | Formula |
|---------|--------|---------|
| transition_prob | Markov matrix | P(next_cat \| current_cat) |
| purchase_velocity | Orders | orders_last_30d / avg_monthly |
| category_affinity | Co-occurrence | P(cat_A \| bought cat_B) |
| seasonality_index | Timestamp | FFT decomposition of weekly sales |
| days_since_last | Order history | now - last_purchase_date |

### 5.2 Prophet (Trend + Seasonal)
```python
from prophet import Prophet
model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
model.fit(df[['ds', 'y']])  # ds=date, y=quantity
forecast = model.predict(future_df)
```

### 5.3 XGBoost (Feature-Based)
```python
import xgboost as xgb
model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05)
model.fit(X_train, y_train)
```

### 5.4 LSTM (Sequential)
```python
# PyTorch: sequence length = 10, predict next quantity
# Input: (batch, seq_len, features)
# Output: (batch, 1) — next period demand
```

### 5.5 Ensemble Strategy
- Weighted average: Prophet(0.3) + XGBoost(0.5) + LSTM(0.2)
- Weights tuned by validation MAPE per SKU category
- Fallback: XGBoost alone if others fail

---

## 6. PRODUCTION OUTPUT DEFINITIONS

### 6.1 Inventory Optimisation
- **Reorder Point** = avg_daily_demand × lead_time_days + safety_stock
- **Safety Stock** = Z × σ_demand × √lead_time  (Z=1.65 for 95% service level)
- **Overstock Alert**: if current_stock > reorder_point × 2.5

### 6.2 Trend Detection
- Rising: 3-week rolling avg demand rising > 15%
- Declining: 3-week rolling avg demand falling > 20%
- Emerging: new SKU with exponential adoption rate

### 6.3 Resource Allocation
- Production hours = Σ(predicted_volume × unit_production_time)
- Schedule: earliest-deadline-first across SKUs
- Supply chain sync: flag when lead_time exceeds demand_horizon

---

## 7. ITERATIVE UPDATE LOOP

### Trigger Conditions
1. **Threshold trigger**: actual vs. predicted MAPE > 15%
2. **Scheduled trigger**: weekly batch re-train (cron)
3. **Manual trigger**: admin dashboard button

### Update Process
```
1. Collect actual_outcomes from last prediction window
2. Compute delta = actual - predicted
3. If delta triggers threshold:
   a. Append new data to training set
   b. Re-fit Markov matrices (online update: T[i][j] = α×new + (1-α)×old)
   c. Re-train XGBoost on rolling 6-month window
   d. Fine-tune LSTM with new sequences (transfer learning)
   e. Update Prophet with new observations
4. Log model version, metrics, timestamp
5. Serve new predictions
```

### Online Markov Update (no full re-train needed)
```python
alpha = 0.1  # learning rate for online update
for new_transition in new_data:
    T[new_transition.from][new_transition.to] = (
        (1 - alpha) * T[new_transition.from][new_transition.to] + alpha
    )
# Re-normalize row
```

---

## 8. API ENDPOINTS (FastAPI)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/data/load` | Upload CSV dataset |
| GET | `/api/v1/data/status` | Pipeline status |
| POST | `/api/v1/markov/train` | Train transition matrix |
| GET | `/api/v1/markov/predict` | Get next-item predictions |
| POST | `/api/v1/forecast/train` | Train forecasting models |
| GET | `/api/v1/forecast/predict` | Get demand forecast |
| GET | `/api/v1/inventory/alerts` | Overstock / stockout alerts |
| GET | `/api/v1/trends` | Category trend signals |
| POST | `/api/v1/update/trigger` | Manual iterative update |
| GET | `/api/v1/metrics` | Model performance metrics |

---

## 9. TECH STACK DECISIONS

| Layer | Technology | Status |
|-------|-----------|--------|
| Frontend | React 19 + TypeScript + Vite | ✅ Exists |
| Styling | Vanilla CSS (dark glassmorphism) | ✅ Exists |
| ML Pipeline | Python 3.11+ | 🔲 Planned |
| API Server | FastAPI + Uvicorn | 🔲 Planned |
| Forecasting | Prophet + XGBoost + PyTorch | 🔲 Planned |
| Data Storage | SQLite (dev) → PostgreSQL (prod) | 🔲 Planned |
| Sequence Model | Markov (custom) + LSTM (PyTorch) | 🔲 Planned |
| Deployment | Vercel (frontend) + Railway/Render (API) | 🔲 TBD |

---

## 10. DIRECTORY STRUCTURE (TARGET — End State)

```
AntiGrav/
├── src/                          # Frontend (existing)
│   ├── components/
│   │   ├── [existing NN viz]
│   │   ├── MarketDashboard.tsx   # Phase 4 NEW
│   │   ├── ForecastChart.tsx     # Phase 4 NEW
│   │   ├── InventoryPanel.tsx    # Phase 4 NEW
│   │   └── TrendDetector.tsx     # Phase 4 NEW
│   ├── engine/
│   │   └── [existing]
│   └── api/
│       └── marketClient.ts       # Phase 4: API calls to backend
│
AntiGravBackend/                  # Phase 1 NEW (Python)
├── main.py                       # FastAPI app
├── pipeline/
│   ├── ingest.py                 # CSV loading + cleaning
│   ├── preprocess.py             # Sequencing + encoding
│   └── split.py                  # Temporal train/test split
├── models/
│   ├── markov.py                 # Markov chain engine
│   ├── forecaster.py             # Prophet + XGBoost + LSTM ensemble
│   └── evaluator.py              # Hit Rate, Perplexity, MAPE
├── production/
│   ├── inventory.py              # Reorder points + safety stock
│   ├── trends.py                 # Trend detection
│   └── allocator.py              # Resource allocation
├── updater/
│   └── iterative.py              # Online update loop
├── requirements.txt
└── data/                         # Raw + processed datasets
    ├── raw/
    └── processed/
```
