from fastapi import FastAPI, File, UploadFile
import pandas as pd
from io import StringIO
from typing import List
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load the dataset
df = pd.read_csv('genz_money_spends.csv')

# Print the columns of the DataFrame to verify their names
print("Columns in the dataset:", df.columns)

# Create a 'Category' column by aggregating existing columns
df['Category'] = df[['Rent (USD)', 'Groceries (USD)', 'Eating Out (USD)', 'Entertainment (USD)', 
                     'Subscription Services (USD)', 'Education (USD)', 'Online Shopping (USD)', 
                     'Savings (USD)', 'Investments (USD)', 'Travel (USD)', 'Fitness (USD)', 
                     'Miscellaneous (USD)']].sum(axis=1)

# Convert 'Category' to categorical by binning (optional)
df['Category'] = pd.cut(df['Category'], bins=5, labels=False)

# Feature Engineering
X = df[['Age', 'Income (USD)', 'Rent (USD)', 'Groceries (USD)', 'Eating Out (USD)']]  # Add more features
y = df['Category']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Selection and Training with increased max_iter and solver
model = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced')  # Adding class_weight
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Additional Evaluation Metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Prediction (example: predicting for a new data point)
new_data = pd.DataFrame({'Age': [25], 'Income (USD)': [50000], 'Rent (USD)': [1000], 'Groceries (USD)': [300], 'Eating Out (USD)': [200]})
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

print(f"Prediction: {prediction}")
