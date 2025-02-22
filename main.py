from fastapi import FastAPI, File, UploadFile
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import joblib

app = FastAPI()

# Load the dataset
df = pd.read_csv('genz_money_spends.csv')

# Feature Engineering
df['Investments'] = df['Investments (USD)'] / df['Income (USD)']
df['Income_on_Rent'] = df['Rent (USD)'] / df['Income (USD)']
df['Savings'] = df['Savings (USD)'] / df['Income (USD)']
df['Education'] = df['Education (USD)'] / df['Income (USD)']
df['Leisure'] = df[['Eating Out (USD)', 'Entertainment (USD)', 'Subscription Services (USD)',
                     'Online Shopping (USD)', 'Travel (USD)', 'Fitness (USD)', 'Miscellaneous (USD)']].sum(axis=1) / df['Income (USD)']

# Define Features and Target
X = df[['Age', 'Investments', 'Rent (USD)', 'Groceries (USD)', 'Eating Out (USD)', 'Income_on_Rent', 'Savings', 'Leisure']]
y = df['Income (USD)']

# Select top 5 features for better performance
selector = SelectKBest(score_func=f_regression, k=5)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

# Base Models for the Stacking Model
base_models = [
    ('rf', RandomForestRegressor(n_jobs=-1, random_state=42)),
    ('gb', GradientBoostingRegressor(random_state=42))
]

# Define the meta-model (Lasso in this case)
meta_model = Lasso(alpha=0.1)

# Initialize the Stacking Regressor
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Fit the Stacking Model
stacking_model.fit(X_train, y_train)
joblib.dump(stacking_model, 'stacking_model.pkl')

# Evaluate the model
stacking_pred = stacking_model.predict(X_test)
stacking_mse = mean_squared_error(y_test, stacking_pred)
stacking_r2 = r2_score(y_test, stacking_pred)

# Print Evaluation Metrics
print(f"Stacking Model MSE: {stacking_mse}")
print(f"Stacking Model R² Score: {stacking_r2}")

# Feature Importance using RandomForest
rf_model = RandomForestRegressor(n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
feature_importance = rf_model.feature_importances_

# Visualize Feature Importance
plt.barh(selected_features, feature_importance)
plt.xlabel("Feature Importance")
plt.title("Feature Importance for Random Forest Model")
plt.show()

# Sensitivity Analysis - Simulate different income distributions
new_data_scenarios = pd.DataFrame([
    {'Age': 25, 'Investments': 0.05, 'Rent (USD)': 1000, 'Groceries (USD)': 300, 'Eating Out (USD)': 200, 'Income_on_Rent': 0.02, 'Savings': 0.1, 'Leisure': 0.4},
    {'Age': 25, 'Investments': 0.06, 'Rent (USD)': 1200, 'Groceries (USD)': 250, 'Eating Out (USD)': 200, 'Income_on_Rent': 0.024, 'Savings': 0.1, 'Leisure': 0.42},
    {'Age': 25, 'Investments': 0.04, 'Rent (USD)': 800, 'Groceries (USD)': 500, 'Eating Out (USD)': 200, 'Income_on_Rent': 0.016, 'Savings': 0.1, 'Leisure': 0.45}
])

# Select only relevant features and scale
test_scenarios = new_data_scenarios[selected_features]
test_scenarios_scaled = scaler.transform(test_scenarios)

# Predict for each scenario
stacking_predictions = stacking_model.predict(test_scenarios_scaled)
for i, prediction in enumerate(stacking_predictions):
    print(f"Prediction for Scenario {i + 1}: {prediction}")
