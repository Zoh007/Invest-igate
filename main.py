from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import joblib

app = FastAPI()

# Load dataset function
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Feature Engineering
    df['Income_on_Rent'] = df['Rent (USD)'] / df['Income (USD)']
    df['Savings'] = df['Savings (USD)'] / df['Income (USD)']
    df['Essentials'] = (df['Rent (USD)'] + df['Groceries (USD)']) / df.iloc[:, 2:].sum(axis=1)
    df['Category'] = df.iloc[:, 2:].sum(axis=1)  # Total spending as a category

    return df

# Budget recommendation function
def suggest_budget_plan(user):
    income = user["Income (USD)"]
    rent = user["Rent (USD)"]
    groceries = user["Groceries (USD)"]
    eating_out = user["Eating Out (USD)"]
    savings = user["Savings (USD)"]
    
    # Ideal budget allocation (50% Needs, 30% Wants, 20% Savings)
    ideal_needs = 0.50 * income
    ideal_wants = 0.30 * income
    ideal_savings = 0.20 * income

    actual_needs = rent + groceries
    actual_wants = eating_out
    actual_savings = savings

    advice = []
    if actual_needs > ideal_needs:
        advice.append("Your essential spending (rent, groceries) is too high. Consider reducing rent or grocery expenses.")
    if actual_wants > ideal_wants:
        advice.append("You're spending too much on non-essentials like eating out. Try reducing entertainment expenses.")
    if actual_savings < ideal_savings:
        advice.append("Your savings rate is low. Aim to save at least 20% of your income by cutting down unnecessary expenses.")

    return advice if advice else ["Your spending is well-balanced. Keep it up!"]

# Endpoint: Upload dataset and train model
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    data_str = StringIO(contents.decode('utf-8'))
    df = pd.read_csv(data_str)

    # Feature Engineering
    df['Income_on_Rent'] = df['Rent (USD)'] / df['Income (USD)']
    df['Savings'] = df['Savings (USD)'] / df['Income (USD)']
    df['Essentials'] = (df['Rent (USD)'] + df['Groceries (USD)']) / df.iloc[:, 2:].sum(axis=1)
    df['Category'] = df.iloc[:, 2:].sum(axis=1)

    # Feature Set and Target
    X = df[['Age', 'Income (USD)', 'Rent (USD)', 'Groceries (USD)', 'Eating Out (USD)', 'Income_on_Rent', 'Savings', 'Essentials']]
    y = df['Category']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Base Models for the Stacking Model
    base_models = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42)),
        ('svr', SVR(kernel='rbf')),
        ('knn', KNeighborsRegressor())
    ]

    # Define the meta-model (Lasso)
    meta_model = Lasso(alpha=0.1)

    # Stacking Regressor
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    stacking_model.fit(X_train, y_train)

    # Save models for later
    joblib.dump(stacking_model, 'stacking_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Evaluate
    stacking_pred = stacking_model.predict(X_test)
    stacking_mse = mean_squared_error(y_test, stacking_pred)
    stacking_r2 = r2_score(y_test, stacking_pred)

    return {"MSE": stacking_mse, "R² Score": stacking_r2}

# Endpoint: Predict category + give budget advice
@app.post("/predict/")
async def predict(user: dict):
    try:
        # Load model & scaler
        model = joblib.load('stacking_model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Prepare input
        user_data = [[
            user["Age"], user["Income (USD)"], user["Rent (USD)"], user["Groceries (USD)"], 
            user["Eating Out (USD)"], user["Rent (USD)"] / user["Income (USD)"], 
            user["Savings (USD)"] / user["Income (USD)"], 
            (user["Rent (USD)"] + user["Groceries (USD)"]) / sum(user.values())
        ]]
        
        user_scaled = scaler.transform(user_data)

        # Make prediction
        category = model.predict(user_scaled)[0]

        # Generate budget advice
        budget_advice = suggest_budget_plan(user)

        return {
            "predicted_spending_category": float(category),
            "budget_advice": budget_advice
        }
    
    except Exception as e:
        return {"error": str(e)}

# Load dataset for initial model training
df = load_data('genz_money_spends.csv')

# Train model on startup
X = df[['Age', 'Income (USD)', 'Rent (USD)', 'Groceries (USD)', 'Eating Out (USD)', 'Income_on_Rent', 'Savings', 'Essentials']]
y = df['Category']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define Stacking Model
base_models = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('gb', GradientBoostingRegressor(random_state=42)),
    ('svr', SVR(kernel='rbf')),
    ('knn', KNeighborsRegressor())
]

meta_model = Lasso(alpha=0.1)
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
stacking_model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(stacking_model, 'stacking_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Initial model evaluation
stacking_pred = stacking_model.predict(X_test)
stacking_mse = mean_squared_error(y_test, stacking_pred)
stacking_r2 = r2_score(y_test, stacking_pred)

print(f"Stacking Model MSE: {stacking_mse}")
print(f"Stacking Model R² Score: {stacking_r2}")
