from fastapi import FastAPI, File, UploadFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import joblib

app = FastAPI()

# Load the dataset (Assuming it's initially loaded from a file)
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Feature Engineering
    df['Income_on_Rent'] = df['Rent (USD)'] / df['Income (USD)']
    df['Savings'] = df['Savings (USD)'] / df['Income (USD)']
    df['Essentials'] = (df['Rent (USD)'] + df['Groceries (USD)']) / df[['Rent (USD)', 'Groceries (USD)', 'Eating Out (USD)', 'Entertainment (USD)',
                         'Subscription Services (USD)', 'Education (USD)', 'Online Shopping (USD)', 
                         'Savings (USD)', 'Investments (USD)', 'Travel (USD)', 'Fitness (USD)', 
                         'Miscellaneous (USD)']].sum(axis=1)
    df['Category'] = df[['Rent (USD)', 'Groceries (USD)', 'Eating Out (USD)', 'Entertainment (USD)', 
                         'Subscription Services (USD)', 'Education (USD)', 'Online Shopping (USD)', 
                         'Savings (USD)', 'Investments (USD)', 'Travel (USD)', 'Fitness (USD)', 
                         'Miscellaneous (USD)']].sum(axis=1)

    return df

# FastAPI endpoint to upload and use custom datasets
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    data_str = StringIO(contents.decode('utf-8'))
    df = pd.read_csv(data_str)

    # Feature Engineering
    df['Income_on_Rent'] = df['Rent (USD)'] / df['Income (USD)']
    df['Savings'] = df['Savings (USD)'] / df['Income (USD)']
    df['Essentials'] = (df['Rent (USD)'] + df['Groceries (USD)']) / df[['Rent (USD)', 'Groceries (USD)', 'Eating Out (USD)', 'Entertainment (USD)',
                         'Subscription Services (USD)', 'Education (USD)', 'Online Shopping (USD)', 
                         'Savings (USD)', 'Investments (USD)', 'Travel (USD)', 'Fitness (USD)', 
                         'Miscellaneous (USD)']].sum(axis=1)
    df['Category'] = df[['Rent (USD)', 'Groceries (USD)', 'Eating Out (USD)', 'Entertainment (USD)', 
                         'Subscription Services (USD)', 'Education (USD)', 'Online Shopping (USD)', 
                         'Savings (USD)', 'Investments (USD)', 'Travel (USD)', 'Fitness (USD)', 
                         'Miscellaneous (USD)']].sum(axis=1)

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

    # Define the meta-model (Lasso in this case)
    meta_model = Lasso(alpha=0.1)

    # Initialize the Stacking Regressor with Lasso as the meta-model
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

    # Fit the Stacking Model
    stacking_model.fit(X_train, y_train)

    # Save the model for future predictions
    joblib.dump(stacking_model, 'stacking_model.pkl')

    # Evaluate the model
    stacking_pred = stacking_model.predict(X_test)
    stacking_mse = mean_squared_error(y_test, stacking_pred)
    stacking_r2 = r2_score(y_test, stacking_pred)

    # Return the evaluation metrics
    return {"MSE": stacking_mse, "R² Score": stacking_r2}

# Load model from the saved file for prediction
def load_model():
    return joblib.load('stacking_model.pkl')

# Scenario Analysis function (already implemented)
def run_scenario_analysis(model, scaler):
    new_data_scenarios = [
        {'Age': 25, 'Income (USD)': 50000, 'Rent (USD)': 1000, 'Groceries (USD)': 300, 'Eating Out (USD)': 200, 
         'Income_on_Rent': 0.02, 'Savings': 0.1, 'Essentials': 0.4},  # baseline scenario
        {'Age': 25, 'Income (USD)': 50000, 'Rent (USD)': 1200, 'Groceries (USD)': 250, 'Eating Out (USD)': 200, 
         'Income_on_Rent': 0.024, 'Savings': 0.1, 'Essentials': 0.42},  # Increased rent
        {'Age': 25, 'Income (USD)': 50000, 'Rent (USD)': 800, 'Groceries (USD)': 500, 'Eating Out (USD)': 200, 
         'Income_on_Rent': 0.016, 'Savings': 0.1, 'Essentials': 0.45}   # Increased groceries
    ]

    # Convert the simulated data to a DataFrame and scale it
    new_data_scenarios_df = pd.DataFrame(new_data_scenarios)
    new_data_scaled = scaler.transform(new_data_scenarios_df)

    # Predict for each scenario
    stacking_predictions = model.predict(new_data_scaled)

    for i, prediction in enumerate(stacking_predictions):
        print(f"Prediction for Scenario {i + 1}: {prediction}")

# Load dataset and train model (This part happens when running the app)
df = load_data('genz_money_spends.csv')

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

# Define the meta-model (Lasso in this case)
meta_model = Lasso(alpha=0.1)

# Initialize the Stacking Regressor with Lasso as the meta-model
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Fit the Stacking Model
stacking_model.fit(X_train, y_train)

# Evaluate the model
stacking_pred = stacking_model.predict(X_test)
stacking_mse = mean_squared_error(y_test, stacking_pred)
stacking_r2 = r2_score(y_test, stacking_pred)

# Print the Evaluation Metrics
print(f"Stacking Model MSE: {stacking_mse}")
print(f"Stacking Model R² Score: {stacking_r2}")

# Scenario Analysis for trained model
run_scenario_analysis(stacking_model, scaler)
