# 🛵 Ola Bike Ride Request Demand Forecasting Engine

An end-to-end Machine Learning web application and REST API built to forecast hourly bike ride demand using environmental and temporal telemetry data. The pipeline is powered by an **XGBoost Regressor** and served via a lightweight **Flask** web server.

---

## 📌 Features

* **High-Performance Modeling:** Trained using XGBoost Gradient Boosting with custom feature engineering and scaled attributes.
* **Dual Interface Support:**
  * **Interactive Web Dashboard:** Clean, high-contrast, cardless interface with embedded Chart.js visualizations.
  * **REST API Endpoint:** Production-ready JSON endpoint for external program access.
* **Dynamic Pipeline Fallback:** Self-healing data pipeline capable of synthetic dataset creation if local data is missing.
* **Global Model Analytics:** Built-in relative feature importance tracking.

---

## 🛠️ Tech Stack & Architecture

* **Machine Learning:** `scikit-learn`, `xgboost`, `pandas`, `numpy`
* **Backend Framework:** Python `Flask`
* **Frontend Analytics:** HTML5, Modern CSS3, `Chart.js`
* **Serialization:** `joblib` / `pickle`

---

## 📁 Repository Structure

```text
├── artifacts/
│   ├── model.pkl            # Serialized XGBoost Regressor
│   └── scaler.pkl           # Scaled preprocessing object
├── static/
│   └── style.css            # Enterprise UI styling
├── templates/
│   └── index.html           # Dashboard interface & analytics
├── app.py                   # Flask server & REST API controller
├── train.py                 # Pipeline processing & model training
├── ola.csv                  # Dataset (Optional / Auto-generated)
└── requirements.txt         # Package dependencies