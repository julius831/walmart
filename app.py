import pandas as pd
import streamlit as st
import joblib

model = joblib.load('model.pkl')

st.title("Walmart Sales Prediction App")
st.write("Enter the store and economic details below to predict weekly sales.")

store = st.number_input("Store Number", min_value=1, max_value=45, value=1, step=1)
holiday_flag = st.selectbox("Holiday Week?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
temperature = st.number_input("Temperature (°F)", value=60.0)
fuel_price = st.number_input("Fuel Price ($)", value=3.50)
cpi = st.number_input("Consumer Price Index (CPI)", value=180.0)
unemployment = st.number_input("Unemployment Rate (%)", value=7.0)

if st.button("Predict Weekly Sales"):
    input_data = pd.DataFrame({
        "Store": [store],
        "Holiday_Flag": [holiday_flag],
        "Temperature": [temperature],
        "Fuel_Price": [fuel_price],
        "CPI": [cpi],
        "Unemployment": [unemployment]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Weekly Sales: ${prediction:,.2f}")

