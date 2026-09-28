# Dynamic Pricing AI System

An end-to-end **demand forecasting and price optimization** system. An **XGBoost** model predicts product demand from price, discount, competitor price, customer sentiment and calendar effects. A simulation engine then searches candidate prices to find the **revenue-maximizing price**, and **SHAP** explains what drives each prediction. Everything is served in an interactive **Streamlit** dashboard.

---

## Business Problem

Setting prices by intuition leaves revenue on the table. Price too high and demand collapses; price too low and margin is lost. The right price also shifts with competitor pricing, discounts, customer sentiment, weekends and festive seasons.

This project answers one question for every product:

> *Given today's market conditions, what price maximizes expected revenue, and why?*

---

## How It Works

```
Market inputs ──► Feature engineering ──► XGBoost demand model ──► Price simulation ──► Optimal price
 (price, discount,    (price_diff,            (predicted units)        (50 candidate        + revenue curve
  competitor price,    is_weekend,                                     prices, revenue       + SHAP explanation
  sentiment, day)      encodings)                                      = price × demand)
```

### 1. Dataset
A **simulated retail dataset of 20,000 records** (20 electronics products × 1,000 observations over 2023–2024), generated in `AI-Demand-Prediction.ipynb`. Demand is driven by realistic mechanisms:

| Driver | Effect on demand |
|---|---|
| Price | Product-specific price sensitivity (each product has its own elasticity) |
| Competitor price | Cheaper than competitors → higher demand |
| Discount | 0–20% discount tiers |
| Customer sentiment | Derived from product ratings (3.5–5.0) |
| Weekend | Weekend demand boost |
| Seasonality | Festive-season boost in Oct–Dec |

Using simulated data with known behaviour makes it possible to check whether the model recovers the true demand drivers.

### 2. Feature Engineering
Nine model features: `price`, `discount`, `sentiment_score`, `competitor_price`, `day_of_week`, `product_id`, `is_weekend`, `price_diff` (price − competitor price) and `month`. Categorical fields are label-encoded, and the encoders are saved alongside the model.

### 3. Demand Model
An **XGBoost regressor** is trained to predict units demanded (`models/xgb_model.pkl`).

### 4. Price Optimization
For the selected product and market conditions, the app evaluates **50 candidate prices (₹50–₹500)**. It predicts demand at each price, computes **revenue = price × predicted demand**, and recommends the price with maximum expected revenue. A revenue-vs-price curve shows the trade-off visually.

### 5. Explainability
**SHAP** feature attributions show which factors (price gap vs competitor, sentiment, weekend and so on) push predicted demand up or down, so recommendations are transparent rather than a black box.

---

## Dashboard

| Tab | What it shows |
|---|---|
| **Demand** | Predicted demand for the chosen product, price and market conditions |
| **Optimization** | Optimal price, maximum expected revenue, revenue-vs-price curve, and a raise/lower price recommendation |
| **Insights** | SHAP feature-importance chart explaining the prediction |

**Sidebar controls:** price, discount, sentiment score, competitor price, day of week, product.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & features | Python, Pandas, NumPy |
| Modeling | XGBoost, Scikit-learn (label encoding) |
| Explainability | SHAP |
| Visualization & app | Streamlit, Matplotlib |

---

## Project Structure

```
dynamic-pricing-ai/
├── AI-Demand-Prediction.ipynb   # dataset simulation
├── app/
│   └── streamlit_app.py         # dashboard: prediction, optimization, SHAP
├── data/raw/                    # generated dataset
├── models/
│   ├── xgb_model.pkl            # trained XGBoost demand model
│   └── encoders.pkl             # day & product label encoders
└── requirements.txt
```

---

## Run Locally

```bash
git clone https://github.com/Sargam208/dynamic-pricing-ai.git
cd dynamic-pricing-ai
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Future Improvements

- Add model evaluation (MAE / RMSE / R² vs a baseline) to the notebook
- Optimize for **profit** using per-product cost data, not just revenue
- Estimate explicit **price elasticity** per product with log-log regression
- Validate recommendations with an **A/B pricing experiment**

---

## Contributors

- **Sargam Hemnani**, M.Tech AI, Delhi Technological University ([GitHub](https://github.com/Sargam208))
- **Divyam Jariwal** ([GitHub](https://github.com/divyamjariwal))
