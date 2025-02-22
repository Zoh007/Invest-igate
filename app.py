import streamlit as st
import requests

# FastAPI Backend URL
BACKEND_URL = "http://127.0.0.1:8000"

# Streamlit UI
st.title("💰 AI Finance Assistant")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Categorize Expenses", "Budget Planner", "AI Chatbot"])

# 1️⃣ Categorize Expenses
if page == "Categorize Expenses":
    st.header("🛒 Categorize Your Expenses")
    description = st.text_input("Enter expense description", "")

    if st.button("Categorize"):
        response = requests.post(f"{BACKEND_URL}/categorize", json={"description": description})
        if response.status_code == 200:
            category = response.json()["category"]
            st.success(f"✅ This expense is categorized as: **{category}**")
        else:
            st.error("❌ Error: Unable to categorize expense")

# 2️⃣ Budget Planner
elif page == "Budget Planner":
    st.header("📊 Get Budget Recommendations")
    income = st.number_input("Enter your monthly income", min_value=0, step=100)

    if st.button("Get Budget Plan"):
        response = requests.post(f"{BACKEND_URL}/recommend-budget", json={"income": income})
        if response.status_code == 200:
            budget_plan = response.json()["budget_plan"]
            st.success(f"💡 Recommendation: **{budget_plan}**")
        else:
            st.error("❌ Error: Unable to fetch budget recommendation")

# 3️⃣ AI Chatbot for Financial Advice
elif page == "AI Chatbot":
    st.header("💬 Ask AI for Financial Advice")
    user_input = st.text_input("Ask a finance-related question", "")

    if st.button("Get Advice"):
        response = requests.post(f"{BACKEND_URL}/chat", json={"text": user_input})
        if response.status_code == 200:
            reply = response.json()["reply"]
            st.info(f"🤖 AI says: {reply}")
        else:
            st.error("❌ Error: Unable to get AI response")

# Run Streamlit: `streamlit run app.py`
