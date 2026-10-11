# ProjectPulse AI: Machine Learning & Deep Learning in Software Project Analytics

> An empirical project analytics platform that combines classical ML (GBDT, Random Forest), deep neural networks (PyTorch), and NLP to forecast software delivery delays, predict cost overruns, and triage sprint bottlenecks.

---

## 💡 Why This Project?

In software engineering, estimating timelines and budget variance has historically relied on gut feeling, Planning Poker, or outdated sizing heuristics. Despite agile methodologies, projects still suffer from scope creep, unexpected turnover, and technical debt that compound into massive release delays.

This project investigates how **Machine Learning (ML)** and **Deep Learning (DL)** can turn passive project telemetry (sprint velocity, code churn, test coverage, bug density, and ticket discussions) into actionable decision support:

1. **Categorical Risk Classification:** Can we accurately flag high-risk releases before sprint commitments are locked?
2. **Multi-Task Timeline & Cost Forecasting:** Can a deep neural network jointly predict schedule slippage (calendar days) and budget overruns (%) better than independent regressors?
3. **Automated Ticket Triage (NLP):** Can we automatically classify Jira/GitHub bug reports and detect potential release blockers in real time?
4. **Prescriptive "What-If" Analysis:** Can Explainable AI (XAI) tell project managers exactly which interventions (e.g., cutting scope vs adding senior engineers) will save the most calendar days?

---

## 📊 Benchmark Highlights

Evaluated on an empirical dataset of **1,500 software engineering projects** and **4,000 sprint issues**:

### 1. Project Delivery Risk Classification (Low / Medium / High)
| Model | Type | Accuracy | Weighted F1 | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression ($L_2$)** | Classical Linear | **95.67%** | **0.9560** | **0.04 ms** |
| **Gradient Boosting (GBDT)** | Ensemble Boosting | 94.33% | 0.9427 | 0.09 ms |
| **Random Forest (120 Trees)** | Ensemble Bagging | 93.67% | 0.9362 | 0.11 ms |
| **Deep Neural Net (MLP)** | Deep Learning | 92.67% | 0.9254 | 0.12 ms |

*Key takeaway:* For tabular project features, fast tree ensembles and regularized linear models deliver top-tier classification accuracy with sub-millisecond inference times.

### 2. Multi-Task Schedule Slip & Cost Overrun Regression
| Model Architecture | Paradigm | Overall $R^2$ Score | Schedule Slip MAE | Cost Overrun MAE |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Task Neural Net (PyTorch)** | **Deep Learning** | **0.8966** | **15.22 Days** | **8.41%** |
| Ridge Regression ($L_2$) | Classical Linear | 0.7724 | 22.34 Days | 12.13% |
| Random Forest Regressor | Ensemble Trees | 0.6636 | 27.30 Days | 14.94% |

*Key takeaway:* Deep learning significantly outperformed traditional regressors on joint multi-task prediction ($R^2 = 0.897$ vs $0.664$). Shared latent representations help the network learn how scope creep, turnover, and tech debt interact non-linearly to inflate delivery schedules.

---

## 🛠️ Project Architecture

```
1st project/
├── README.md                      # Project documentation and guide
├── RESEARCH_REPORT.md             # In-depth academic research whitepaper
├── requirements.txt               # Python package dependencies
├── run_dashboard.py               # 1-Click launcher (starts server + opens browser)
│
├── data/
│   ├── generate_datasets.py       # Telemetry synthesizer modeled on SE literature
│   ├── projects_dataset.csv       # 1,500 projects with 14 telemetry features
│   └── tickets_dataset.csv        # 4,000 sprint tickets with text & triage labels
│
├── ml_engine/
│   ├── risk_classifier.py         # Tabular ML: Logistic Regression, Random Forest, GBDT
│   ├── delay_cost_forecaster.py   # Deep Learning: PyTorch Multi-Task Neural Network
│   ├── nlp_ticket_triage.py       # NLP: TF-IDF + Neural MLP for issue triage
│   ├── xai_explainer.py           # Feature attribution & counterfactual scenario simulator
│   ├── benchmark_runner.py        # Automated benchmark comparison script
│   └── train_all.py               # Master end-to-end training pipeline
│
├── models/                        # Serialized weights, scalers, and benchmark results
│   ├── risk_classifier.joblib
│   ├── delay_cost_mlp.pt
│   ├── forecaster_scalers.json
│   ├── nlp_triage.joblib
│   └── benchmark_results.json
│
├── server/
│   ├── api.py                     # FastAPI REST server
│   └── static/                    # Dark glassmorphic web dashboard
│       ├── index.html             # Single-Page Application interface
│       ├── app.js                 # Frontend inference controller
│       └── styles.css             # Custom CSS design system
│
└── tests/
    └── test_models.py             # Unit and integration test suite
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher installed.

### 2. Setup Virtual Environment (Recommended)
```bash
# Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Dashboard
To start the FastAPI backend and open the interactive dashboard:
```bash
python run_dashboard.py
```
Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 💻 What You Can Do in the Dashboard

- **Executive Overview:** View portfolio-level KPIs, accuracy rates, and telemetry breakdown.
- **Live Risk Forecaster:** Adjust project parameters (team size, scope creep %, technical debt, test coverage) using interactive sliders and run live multi-model predictions.
- **NLP Ticket Triage:** Paste any bug report or pick sample issues to classify urgency, sentiment, and recommended PM actions.
- **What-If Scenario Simulator:** Test managerial interventions (e.g. freezing 15% scope, adding senior engineers) to see how many calendar days can be saved before sprint commitments are made.
- **Empirical Benchmarks:** Review side-by-side performance metrics across classical ML, ensemble methods, and deep learning.
- **Research Trends:** Read deep dives on emerging frontiers like Graph Neural Networks (GNNs) on task dependency DAGs and LLM-assisted agile planning.

---

## 🧪 Retraining Models & Running Tests

If you want to modify feature weights, regenerate data, or run the test suite:

```bash
# 1. Regenerate synthetic project & ticket datasets
python data/generate_datasets.py

# 2. Retrain all ML & Deep Learning models from scratch
python ml_engine/train_all.py

# 3. Run the unit and integration tests
python tests/test_models.py
```

---

## 📄 Research Whitepaper
For an in-depth academic study including theoretical background, full literature review, threats to validity, and research roadmaps, see:
📘 [RESEARCH_REPORT.md](file:///c:/Users/harsh/OneDrive/Desktop/1st%20project/RESEARCH_REPORT.md)

---

## 📜 License
This project is open-source under the [MIT License](LICENSE). Contributions, feedback, and experiment extensions are welcome!
