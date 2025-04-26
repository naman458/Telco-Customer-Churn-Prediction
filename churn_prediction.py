import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(r"C:\Users\Naman  Maheshwari\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Step 1: Clean data
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
data = data.drop('customerID', axis=1)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())

# Step 2: Feature engineering
data['HighCharges'] = (data['MonthlyCharges'] > 70).astype(int)
data['IsLongTerm'] = (data['tenure'] > 12).astype(int)
data['ChargesPerMonth'] = data['TotalCharges'] / (data['tenure'] + 1)
data = pd.get_dummies(data, columns=[
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
], drop_first=True)

# Step 3: Build XGBoost model
# Split data into features (X) and target (y)
X = data.drop('Churn', axis=1)
y = data['Churn']


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBClassifier(random_state=42, max_depth=7, learning_rate=0.05)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print predictions to check
print("Sample predictions:\n", y_pred[:10])

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

xgboost.plot_importance(model)
plt.show()
