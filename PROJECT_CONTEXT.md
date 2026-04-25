# 🧠 AntiGrav — Market Prediction & Resource Optimization Framework
**Context File — Load this at the start of every session**
*Last Updated: 2026-04-17*

---

## 🎯 PROJECT MISSION

Build an end-to-end **market behavior prediction framework** for large production companies that:
1. Predicts market demand using historical transaction data + ML models
2. Iteratively updates predictions with real-world feedback (closed-loop system)
3. Saves resources: reduces overstock, stockout waste, optimizes production scheduling

The system must be **production-ready** — not a demo. Real datasets, real metrics, real ROI.

---

## 🗺️ ARCHITECTURE OVERVIEW (from diagram)

```
DATA INGESTION
  ├── UCI Online Retail II    (541k transactions, 2 years)
  ├── RetailRocket            (2.7M events, clickstream)
  └── Instacart Basket        (3M orders, reorder flag)

PREPROCESSING
  └── Clean · Dedupe · Sort by (CustomerID, Timestamp)
      └── Build per-customer purchase sequences by product category

MARKOV CHAIN (Transition Matrix)
  ├── 1st-order Markov        P(next | last purchase), per customer segment
  └── Nth-order Extension     P(next | last N purchases), higher recall + sparsity

ACCURACY EVALUATION
  ├── Hit Rate @ K            Top-K prediction accuracy vs. most-popular baseline
  ├── Perplexity              Uncertainty of sequence model on held-out data
  └── Temporal Split          Train on older orders, test on recent orders
       ↑
       └── [ITERATIVE UPDATE LOOP] ←── real-world values feed back here

DEMAND FORECASTING LAYER
  ├── Feature Extraction      Transition probs · purchase velocity · category affinity · seasonality
  └── Forecasting Model       Prophet · XGBoost · LSTM  →  SKU-level demand + trend signal

PRODUCTION OUTPUT
  ├── Inventory Optimisation  Reduce overstock and stockout waste
  ├── Trend Detection         Emerging category shifts and declining demand
  └── Resource Allocation     Production scheduling and supply chain sync
```

---

## 📁 EXISTING CODEBASE — AntiGrav Neural Network Visualizer

**Location:** `c:\Users\ADMIN\Desktop\AntiGrav\`
**Stack:** React 19 + TypeScript + Vite 8 + Lucide React

### File Map
```
AntiGrav/
├── src/
│   ├── App.tsx                  # Main app, state management, persistence (localStorage)
│   ├── index.css                # Design system / CSS variables
│   ├── main.tsx                 # Entry point
│   ├── components/
│   │   ├── NetworkGraph.tsx     # Interactive D3-like neural network canvas  [6.2 KB]
│   │   ├── BlockGraph.tsx       # Architecture block diagram viewer          [7.7 KB]
│   │   ├── BlockInspector.tsx   # Sidebar inspector for arch nodes           [2.8 KB]
│   │   ├── InspectorPanel.tsx   # Sidebar inspector for neurons/edges        [4.4 KB]
│   │   ├── Toolbar.tsx          # Top toolbar: inputs, LR, train/forward     [3.6 KB]
│   │   └── CodeModal.tsx        # JSON/arch import modal                     [4.6 KB]
│   └── engine/
│       ├── Network.ts           # NeuralNetwork class: forward/backward/SGD  [5.2 KB]
│       ├── ArchParser.ts        # Parses JSON arch into ArchGraph             [7.4 KB]
│       ├── Exporter.ts          # Exports network state to JSON/Python        [5.4 KB]
│       └── math.ts              # Activation functions (relu, sigmoid, etc)  [0.6 KB]
├── package.json                 # React 19, Vite 8, Lucide React
└── index.html
```

### Current App Capabilities
- ✅ Interactive MLP (multi-layer perceptron) visualizer
- ✅ Add/remove layers and neurons
- ✅ Forward pass + backpropagation + SGD weight update
- ✅ State persistence via localStorage
- ✅ Block/architecture diagram mode (import JSON arch)
- ✅ Inspector sidebar (neuron weights, activations, gradients)
- ✅ Code modal (JSON import/export, Python export)

### Known Issues (from past session fd7e9931)
- Toolbar widget state (inputs, targets, LR) reset on refresh — FIXED in current code
- BlockGraph not pannable/scrollable — partially addressed

---

## 🔮 NEXT PHASE: Market Prediction Engine

This is a NEW module to be built on top of or alongside the AntiGrav visualizer.

### Datasets to Use
| Dataset | Size | Key Fields |
|---------|------|------------|
| UCI Online Retail II | 541k rows, 2 yrs | InvoiceNo, StockCode, CustomerID, InvoiceDate, Quantity, Price |
| RetailRocket | 2.7M events | visitorid, event(view/addtocart/buy), itemid, timestamp |
| Instacart | 3M orders | order_id, user_id, product_id, reordered, days_since_prior_order |

### Models to Implement
1. **Markov Chain** — 1st order & N-th order transition matrices per customer segment
2. **Prophet** — Seasonal demand forecasting per SKU
3. **XGBoost** — Non-linear feature-based demand regression
4. **LSTM** — Sequential purchase pattern learning

### Key Metrics
- **Hit Rate @ K** — Are top-K predictions correct?
- **Perplexity** — Model confidence on held-out sequences
- **MAE / MAPE** — Demand forecast accuracy
- **Inventory Savings %** — Overstock reduction vs. baseline

---

## 🔄 ITERATIVE UPDATE LOOP (Core Differentiator)

The system re-trains or updates model parameters when:
1. Actual sales deviate from predicted by > threshold (e.g. 15% MAPE)
2. Weekly/daily batch ingestion of new transaction data
3. Admin manually triggers re-evaluation

Feedback mechanism:
```
Prediction → Production Action → Actual Outcome → Delta Measurement → Model Update → New Prediction
```

---

## 📐 DESIGN SYSTEM

The existing app uses these CSS variables (from index.css):
- `--bg-primary`, `--bg-secondary`, `--bg-tertiary`
- `--accent-color`, `--accent-glow`
- `--border-color`
- `--font-family`
- Dark mode glassmorphism aesthetic

---

## 🗓️ DEVELOPMENT PHASES

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ DONE | Neural Network Visualizer (AntiGrav base app) |
| Phase 1 | 🔲 NEXT | Data pipeline: ingest + preprocess datasets |
| Phase 2 | 🔲 TODO | Markov chain sequence model + transition matrix |
| Phase 3 | 🔲 TODO | Demand forecasting layer (Prophet + XGBoost + LSTM) |
| Phase 4 | 🔲 TODO | Production output dashboard (inventory, trends, allocation) |
| Phase 5 | 🔲 TODO | Iterative update loop + real-world feedback mechanism |
| Phase 6 | 🔲 TODO | Full UI integration + enterprise dashboard |

---

## 💡 KEY DECISIONS LOG

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-17 | Use existing AntiGrav React/Vite app as UI shell | Avoids rebuilding; already has good architecture |
| 2026-04-17 | Start context/memory files before coding | Prevent hallucinations, maintain continuity across sessions |
| 2026-04-17 | Target large production companies | Resource optimization has clear enterprise ROI |

---

## ⚠️ CRITICAL CONSTRAINTS

1. **No breaking changes** to existing AntiGrav MLP visualizer — new features extend it
2. **Real datasets only** — no synthetic data for final demos
3. **Iterative loop is mandatory** — static predictions are not acceptable
4. **TypeScript for frontend**, Python backend is acceptable for ML pipeline
5. **Memory files must be updated** at the end of every session

---

*See also: `PROGRESS_LOG.md`, `TECHNICAL_SPEC.md`, `SESSION_NOTES.md`*
