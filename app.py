import streamlit as st
import requests

# FastAPI Backend URL
BACKEND_URL = "http://127.0.0.1:8000"

# Streamlit UI
st.title("💰 AI Finance Assistant")

# Manual Input Section
st.header("📝 Input Your Budget Information")

# Create input fields for all the required information
age = st.number_input("Age", min_value=18, max_value=120, value=25)
income = st.number_input("Monthly Income (USD)", min_value=0, value=3000)
rent = st.number_input("Monthly Rent (USD)", min_value=0, value=1000)
groceries = st.number_input("Monthly Groceries (USD)", min_value=0, value=300)
eating_out = st.number_input("Monthly Eating Out (USD)", min_value=0, value=150)
education = st.number_input("Monthly Education Expenses (USD)", min_value=0, value=200)
investments = st.number_input("Monthly Investments (USD)", min_value=0, value=150)
savings = st.number_input("Monthly Savings (USD)", min_value=0, value=200)
leisure = st.number_input("Monthly Leisure Spending (USD)", min_value=0, value=100)

# Button to process the manual input data
if st.button("Process Budget Information"):
    # Create a dictionary to hold the input data
    input_data = {
        "Age": age,
        "Income (USD)": income,
        "Rent (USD)": rent,
        "Groceries (USD)": groceries,
        "Eating Out (USD)": eating_out,
        "Education (USD)": education,
        "Investments (USD)": investments,
        "Savings (USD)": savings,
        "Leisure (USD)": leisure
    }
    
    # Send the input data to the FastAPI backend for processing
    response = requests.post(f"{BACKEND_URL}/process-manual-input", json=input_data)

    if response.status_code == 200:
        processed_data = response.json()  # Processed data received as JSON
        st.success("✅ Budget Processed Successfully!")
        st.write("📊 Processed Budget Information:")
        st.json(processed_data)  # Display the processed data
    else:
        st.error("❌ Error processing budget information")

