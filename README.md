# Walmart Sales Prediction App

A Streamlit web app that predicts Walmart weekly sales using a Random Forest regression model trained on store, holiday, weather, and economic data.

## Overview

Walmart needs to forecast product sales across stores to optimize inventory, reduce stockouts and overstock, and improve supply chain efficiency. This project builds a regression model that uses store-level and macroeconomic features to predict weekly sales, then deploys it as an interactive web app.

## Features

- Random Forest Regressor trained on historical Walmart sales data
- Interactive Streamlit interface for entering store and economic conditions
- Real-time weekly sales predictions

## Dataset

The model is trained on a Walmart sales dataset containing:

- **Store** — store identifier
- **Holiday_Flag** — whether the week includes a holiday
- **Temperature** — average temperature for the week
- **Fuel_Price** — fuel price in the region
- **CPI** — Consumer Price Index
- **Unemployment** — regional unemployment rate
- **Weekly_Sales** — target variable

## Model Performance

| Metric | Value |
|---|---|
| R² Score | 0.93 |
| RMSE | ~153,140 |

## Project Structure

```
walmart/
├── app.py              # Streamlit app for predictions
├── train_model.py       # Script to train and save the model
├── model.pkl             # Trained Random Forest model
└── README.md
```

## How to Run Locally

1. Clone the repository
   ```
   git clone https://github.com/julius831/walmart.git
   cd walmart
   ```

2. Install dependencies
   ```
   pip install pandas streamlit joblib scikit-learn matplotlib seaborn
   ```

3. (Optional) Retrain the model
   ```
   python train_model.py
   ```

4. Run the app
   ```
   streamlit run app.py
   ```

5. Open the local URL shown in your terminal (usually `http://localhost:8501`)

## Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn (Random Forest Regressor)
- Streamlit

## Author

Julius — [GitHub](https://github.com/julius831)
