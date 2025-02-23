import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import requests

# Helper functions
def generate_savings_recommendations(income):
    return {
        "Leisure": income * 0.6,
        "Savings": income * 0.15,
        "Rent": income * 0.2,
        "Groceries": income * 0.05,
        "Investments": income * 0.1
    }

def plot_spending_distribution(income, recommendations):
    categories = list(recommendations.keys())
    amounts = list(recommendations.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(amounts, labels=categories, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3", len(categories)))
    ax.axis('equal')
    st.pyplot(fig)

# Streamlit UI
st.set_page_config(page_title="AI Finance Assistant", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Spending Recommendations", "Investments", "Debt Management", "Event Budgeting", "Tax Estimator", "Emergency Fund", "Net Worth", "FI Tracker", "Expense Trends"])

# Home Page
if page == "Home":
    st.title("💰 Welcome to AI Finance Assistant")
    st.write("""
        Manage your finances smartly with AI-powered insights. Track income, analyze spending habits, set savings goals, 
        and visualize your financial health in real-time.
    """)
    



# Spending Recommendations Page
elif page == "Spending Recommendations":
    st.title("💸 Spending Recommendations")

    # Slider for financial allocation (real-time update)
    income = st.number_input("Enter Monthly Income (USD)", min_value=0, value=3000)
    
    recommendations = generate_savings_recommendations(income)
    st.write("### Recommended Allocations:")
    st.table(pd.DataFrame(list(recommendations.items()), columns=["Category", "Recommended Amount (USD)"]))
    
    st.write("### Spending Distribution Visualization")
    plot_spending_distribution(income, recommendations)

# Investments Page
elif page == "Investments":
    st.title("📈 Investment Planning")
    
    investment_options = ["Stocks", "Bonds", "Cryptocurrency", "Mutual Funds", "Real Estate"]
    investment_choice = st.selectbox("Select Investment Option", investment_options)
    
    st.write(f"You have selected {investment_choice} for your investments.")
    
    # Simple investment calculator
    investment_amount = st.number_input("Enter amount to invest (USD)", min_value=0, value=500)
    annual_rate_of_return = st.slider("Estimated Annual Rate of Return (%)", min_value=1, max_value=15, value=7)
    years = st.slider("Number of Years", min_value=1, max_value=30, value=10)

    future_value = investment_amount * ((1 + (annual_rate_of_return / 100)) ** years)
    
    st.write(f"Your investment will grow to: ${future_value:,.2f} after {years} years at an annual rate of {annual_rate_of_return}%.")
    
# Debt Management Page
elif page == "Debt Management":
    st.title("💳 Debt Management")

    debt_amount = st.number_input("Enter Total Debt Amount (USD)", min_value=0, value=5000)
    interest_rate = st.slider("Debt Interest Rate (%)", min_value=1, max_value=20, value=5)
    monthly_payment = st.number_input("Enter Monthly Payment (USD)", min_value=0, value=200)

    months_to_pay_off = np.log(monthly_payment / (monthly_payment - debt_amount * (interest_rate / 100) / 12)) / np.log(1 + interest_rate / 100 / 12)
    st.write(f"Remaining months to pay off debt: {int(months_to_pay_off)} months")

# Event Budgeting Page
elif page == "Event Budgeting":
    st.title("🎉 Event Budgeting")

    event_name = st.text_input("Enter Event Name (e.g., Vacation, Wedding)")
    event_budget = st.number_input("Enter Event Budget (USD)", min_value=0, value=1000)
    st.write(f"Event: {event_name} with a budget of ${event_budget}")

# Tax Estimator Page
elif page == "Tax Estimator":
    st.title("💼 Tax Estimator")

    tax_rate = st.slider("Estimated Tax Rate (%)", min_value=0, max_value=40, value=15)
    annual_income = st.number_input("Enter Annual Income (USD)", min_value=0, value=36000)
    estimated_tax = annual_income * (tax_rate / 100)
    st.write(f"Estimated Annual Tax: ${estimated_tax:.2f}")

# Emergency Fund Page
elif page == "Emergency Fund":
    st.title("💼 Emergency Fund")

    emergency_fund_goal = st.number_input("Enter Emergency Fund Goal (USD)", min_value=0, value=3000)
    current_savings = st.number_input("Enter Current Emergency Savings (USD)", min_value=0, value=500)

    remaining_fund = emergency_fund_goal - current_savings
    st.write(f"Remaining Amount to Save for Emergency Fund: ${remaining_fund:.2f}")

# Net Worth Page
elif page == "Net Worth":
    st.title("💸 Net Worth")

    assets = st.number_input("Enter Total Assets (USD)", min_value=0, value=10000)
    liabilities = st.number_input("Enter Total Liabilities (USD)", min_value=0, value=5000)
    net_worth = assets - liabilities
    st.write(f"Your Net Worth is: ${net_worth:.2f}")

# Financial Independence Tracker Page
elif page == "FI Tracker":
    st.title("📈 Financial Independence Tracker")

    annual_expenses = st.number_input("Enter Annual Expenses (USD)", min_value=0, value=18000)
    annual_income = st.number_input("Enter Annual Income (USD)", min_value=0, value=30000)
    fi_ratio = annual_income / annual_expenses
    st.write(f"Your Financial Independence Ratio: {fi_ratio:.2f} (Higher than 1 means financial independence!)")

# Currency Converter Page
# elif page == "Currency Converter":
#     st.title("💱 Currency Converter")

#     currency_from = st.text_input("Enter Currency Code to Convert From (e.g., USD)")
#     currency_to = st.text_input("Enter Currency Code to Convert To (e.g., EUR)")
#     amount = st.number_input("Enter Amount to Convert", min_value=0, value=100)

#     # API call to fetch exchange rates (using a free API for example)
#     api_url = f"https://api.exchangerate-api.com/v4/latest/{currency_from}"
#     response = requests.get(api_url)
#     data = response.json()
#     rate = data['rates'].get(currency_to)

#     if rate:
#         converted_amount = amount * rate
#         st.write(f"{amount} {currency_from} = {converted_amount:.2f} {currency_to}")

# Expense Trends Page
elif page == "Expense Trends":
    st.title("📊 Expense Trends")

    # Use user inputs for expenses in different categories
    rent = st.number_input("Monthly Rent (USD)", min_value=0, value=1000)
    groceries = st.number_input("Monthly Groceries (USD)", min_value=0, value=300)
    eating_out = st.number_input("Monthly Eating Out (USD)", min_value=0, value=150)
    education = st.number_input("Monthly Education Expenses (USD)", min_value=0, value=200)
    investments = st.number_input("Monthly Investments (USD)", min_value=0, value=150)
    savings = st.number_input("Monthly Savings (USD)", min_value=0, value=200)
    leisure = st.number_input("Monthly Leisure Spending (USD)", min_value=0, value=100)

    expenses = {
        "Rent": rent,
        "Groceries": groceries,
        "Eating Out": eating_out,
        "Education": education,
        "Investments": investments,
        "Savings": savings,
        "Leisure": leisure
    }

    # Plotting expense trends
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(expenses.keys(), expenses.values(), color=sns.color_palette("Set2", len(expenses)))
    ax.set_title("Monthly Expense Trends")
    ax.set_ylabel("Amount (USD)")
    ax.set_xlabel("Categories")
    st.pyplot(fig)
