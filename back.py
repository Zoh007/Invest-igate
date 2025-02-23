from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define input data model
class BudgetInfo(BaseModel):
    income: float
    rent: float
    groceries: float
    eating_out: float
    education: float
    investments: float
    savings: float
    leisure: float

# Simulate the spending recommendations based on income
def get_spending_recommendations(income: float):
    recommendations = {
        "Leisure": income * 0.6,
        "Savings": income * 0.15,
        "Rent": income * 0.2,
        "Groceries": income * 0.05,
        "Investments": income * 0.1
    }
    return recommendations

# API Endpoint to get budget recommendations
@app.post("/recommendations")
async def get_budget_recommendations(budget_info: BudgetInfo):
    income = budget_info.income
    recommendations = get_spending_recommendations(income)
    return recommendations
