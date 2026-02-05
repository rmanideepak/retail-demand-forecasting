import streamlit as st
import pandas as pd
import joblib

st.title("Retail Demand Forecast Dashboard")

model = joblib.load("/Users/manideepak/Desktop/PycharmProjects/Retail_Demad_Project/models/model.pkl")

store = st.number_input("Store", 1, 50)
dept = st.number_input("Department", 1, 100)
year = st.number_input("Year", 2012, 2025)
month = st.slider("Month", 1, 12)
week = st.slider("Week", 1, 52)

if st.button("Predict Sales"):
    input_data = pd.DataFrame(
        [[store, dept, year, month, week]],
        columns=['Store', 'Dept', 'year', 'month', 'week']
    )

    prediction = model.predict(input_data)
    st.success(f"Predicted Weekly Sales: {prediction[0]:,.2f}")
