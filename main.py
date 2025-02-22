from fastapi import FastAPI, File, UploadFile
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

app = FastAPI()

# Load the dataset
df = pd.read_csv('genz_money_spends.csv')
# Investment as a proportion of total income
df['Investments'] = df['Investments (USD)'] / df['Income (USD)']

# Income out of Rent
df['Income_on_Rent'] = df['Rent (USD)'] / df['Income (USD)']  # Use df['Savings (USD)'] directly

# Savings out of income
df['Savings'] = df['Savings (USD)'] / df['Income (USD)']  # Use df['Savings (USD)'] directly

# Education expenses as a proportion of total income
df['Education'] = df['Education (USD)'] / df['Income (USD)']

# Leisure as everything else (excluding Rent, Groceries, Investments, and Education)
df['Leisure'] = df[['Eating Out (USD)', 'Entertainment (USD)', 'Subscription Services (USD)',
                     'Online Shopping (USD)', 'Travel (USD)', 'Fitness (USD)', 'Miscellaneous (USD)']].sum(axis=1) / df['Income (USD)']

# Total Spending Category (optional if still needed)
df['Category'] = df[['Rent (USD)', 'Groceries (USD)', 'Eating Out (USD)', 'Entertainment (USD)', 
                     'Subscription Services (USD)', 'Online Shopping (USD)', 
                     'Savings (USD)', 'Travel (USD)', 'Fitness (USD)', 
                     'Miscellaneous (USD)']].sum(axis=1)

# After feature engineering, select the appropriate columns
X = df[['Age', 'Investments','Income (USD)', 'Rent (USD)', 'Groceries (USD)', 'Eating Out (USD)', 'Income_on_Rent', 'Savings', 'Leisure']]  # Add more features
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

# Print Evaluation Metrics for Stacking Model
print(f"Stacking Model MSE: {stacking_mse}")
print(f"Stacking Model R² Score: {stacking_r2}")

# Feature Importance using RandomForest (as it gives more meaningful feature importance)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importance
feature_importance = rf_model.feature_importances_

# Visualize Feature Importance
features = X.columns
plt.barh(features, feature_importance)
plt.xlabel("Feature Importance")
plt.title("Feature Importance for Random Forest Model")
plt.show()


    
# Sensitivity Analysis - Simulate different expense distributions and observe their impact
new_data_scenarios = [
    {'Age': 25, 'Income (USD)': 50000, 'Rent (USD)': 1000, 'Groceries (USD)': 300, 'Eating Out (USD)': 200, 
     'Income_on_Rent': 0.02, 'Savings': 0.1, 'Leisure': 0.4},  # baseline scenario
    {'Age': 25, 'Income (USD)': 50000, 'Rent (USD)': 1200, 'Groceries (USD)': 250, 'Eating Out (USD)': 200, 
     'Income_on_Rent': 0.024, 'Savings': 0.1, 'Leisure': 0.42},  # Increased rent
    {'Age': 25, 'Income (USD)': 50000, 'Rent (USD)': 800, 'Groceries (USD)': 500, 'Eating Out (USD)': 200, 
     'Income_on_Rent': 0.016, 'Savings': 0.1, 'Leisure': 0.45}   # Increased groceries
]

# Convert the simulated data to a DataFrame and scale it
new_data_scenarios_df = pd.DataFrame(new_data_scenarios)
new_data_scaled = scaler.transform(new_data_scenarios_df)

# Predict for each scenario
stacking_predictions = stacking_model.predict(new_data_scaled)

for i, prediction in enumerate(stacking_predictions):
    print(f"Prediction for Scenario {i + 1}: {prediction}")
