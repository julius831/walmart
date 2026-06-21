import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import joblib

df = pd.read_csv("https://raw.githubusercontent.com/BuhariS/walmart_sales_prediction/refs/heads/main/walmart.csv")

df.drop('Date', axis=1, inplace=True)

x = df.drop('Weekly_Sales', axis=1)
y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

random_forest_regressor = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)

model = random_forest_regressor.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

joblib.dump(model, 'model.pkl')

print("Model saved as model.pkl")
print(X_train.columns)