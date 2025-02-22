import streamlit as st
import pandas as pd
import requests

# FastAPI Backend URL
BACKEND_URL = "http://127.0.0.1:8000"

# Streamlit UI
st.title("💰 AI Finance Assistant")

# File Upload Section
st.header("📂 Upload Your Expense CSV File")
uploaded_file = st.file_uploader("Upload genz_money_spends.csv", type=["csv"])

if uploaded_file:
    # Display preview of the uploaded CSV
    df = pd.read_csv(uploaded_file)
    st.write("📊 Preview of Uploaded File:")
    st.dataframe(df.head())  # Show first few rows

    # Convert the file to CSV content for sending to backend
    file_content = uploaded_file.getvalue()

    # Button to process the uploaded CSV file
    if st.button("Process Transactions"):
        # Send CSV content to FastAPI backend for processing
        files = {'file': ('genz_money_spends.csv', file_content, 'text/csv')}
        response = requests.post(f"{BACKEND_URL}/process-csv", files=files)

        if response.status_code == 200:
            processed_data = pd.DataFrame(response.json())  # Convert response to DataFrame
            st.success("✅ Transactions Processed!")
            st.write("📌 Categorized Transactions:")
            st.dataframe(processed_data)  # Display processed data
        else:
            st.error("❌ Error processing CSV file")
