# In-Store Promotion Optimizer — Powered by AI

Executive-grade **promotion optimization engine** built in Streamlit.  
This app enables retail and C-store teams to decide **which SKUs to promote, at what discount depth, and in which region**, while balancing **incremental margin, halo effects, cannibalization, and pull-forward risks**.

---

## 🌟 Features

- **EDA (Exploratory Data Analysis):**  
  Visualize revenue, margin, and promo penetration by region & category. Demand stability vs. intermittency (CV vs ADI).

- **Discount Tuner:**  
  Estimate SKU-level **price elasticity** and simulate **net margin curves** by discount depth. Includes Net-Impact decomposition (Lift + Halo − Cannibalization − Pull-forward).

- **Uplift Modeling (T-learner):**  
  Machine-learning uplift model to estimate **true incremental margin** at SKU level. Includes capture curve and confidence bands.

- **Leaflet Optimizer (ILP):**  
  Linear programming engine to choose **optimal SKU mix under guardrails**: slots, category minimums, brand caps, and markdown budget.

- **Impact Measurement (ITS):**  
  Counterfactual analysis to validate realized impact of promos vs. a trend-based forecast. Visual incremental areas.

- **Quality Assurance:**  
  Auto-scans for data issues (unflagged markdowns, negative margins, missing COGS).

- **Learning Lab (Bandits):**  
  Simulates ε-greedy bandit exploration on discount tiers (10/15/20%) to improve learning week-on-week.

- **Executive Insights & Recommendations:**  
  Auto-generated **PM-style insights and recommendations** for decision-makers. Playbook guidance for High-High SKUs, GM protection, and validation.

---

## 🛠 Tech Stack

- **Frontend:** Streamlit (dark theme UI, executive styling)
- **Visualization:** Plotly (fully dark-themed charts)
- **Modeling:** Scikit-learn (uplift T-learner), Statsmodels (SARIMAX), custom elasticity
- **Optimization:** PuLP (Integer Linear Programming)
- **Data:** Synthetic EPOS dataset (stores × categories × weeks)

---

## 🚀 Getting Started

1. **Clone repo**
   ```bash
   git clone https://github.com/<your-username>/in-store-promo-optimizer.git
   cd in-store-promo-optimizer
