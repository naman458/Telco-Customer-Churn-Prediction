# Telco Customer Churn Prediction

Overview
This project develops a machine learning model to predict customer churn for a telecommunications company using the Telco Customer Churn dataset.
The goal is to identify customers likely to leave and improve retention forecast accuracy by 18% compared to a baseline (~70% accuracy).

Model: XGBoost (Gradient Boosting)

Dataset Size: 7,043 customers, 21 features

Target: Churn (0 = stay, 1 = churn)

Methodology
1. Data Cleaning
Converted Churn column to binary (0/1).

Dropped customerID (non-predictive).

Converted TotalCharges to numeric; filled missing values with the mean.

2. Feature Engineering
Created new features:

HighCharges (MonthlyCharges > 70 → 1, else 0)

IsLongTerm (tenure > 12 → 1, else 0)

ChargesPerMonth (TotalCharges / (tenure + 1))

Applied one-hot encoding to categorical variables using pd.get_dummies().

3. Model Building
Model: XGBoostClassifier with parameters:

max_depth=7

learning_rate=0.05

random_state=42

Data split: 80% training, 20% testing.

4. Evaluation
Metrics: Accuracy, Precision, Recall, F1-score

Feature Importance: Visualized key drivers of churn.

Outcomes

Metric	Result
Accuracy	80.8% (1,138/1,409 correct predictions)
Baseline Comparison	~15.4% improvement over 70% baseline
Goal Progress	Approaching the 85% target
Classification Report

Class	Precision	Recall	F1-Score	Support
0 (Stay)	0.85	0.90	0.87	1036
1 (Churn)	0.67	0.55	0.60	373
Class 0 (Stay): High performance with strong precision and recall.

Class 1 (Churn): Moderate performance — opportunity for improvement.

Feature Importance Highlights
Top drivers: Tenure, MonthlyCharges, Contract_Two year

Insights:

Short tenure and high monthly charges increase churn likelihood.

Long-term contracts decrease churn probability.

Requirements
Python: 3.8+

Libraries:

nginx
Copy
Edit
pip install pandas numpy scikit-learn xgboost matplotlib
How to Run
Download the dataset:
Get WA_Fn-UseC_-Telco-Customer-Churn.csv from Kaggle.

Place the dataset:
Save it in the same folder as churn_prediction.py (or update the file path inside the script).

Run the script:

nginx
Copy
Edit
python churn_prediction.py
Outputs:

Sample predictions

Accuracy and classification report

Feature importance plot

Improvements
Model Enhancements:

Tune scale_pos_weight (e.g., 3) to boost recall for churners.

Engineer additional features (e.g., tenure bins).

Better Visualization: Replace the default XGBoost plot with a cleaner custom plot:

python
Copy
Edit
importance = model.get_booster().get_score(importance_type='gain')
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
features = [x[0] for x in importance]
scores = [x[1] for x in importance]

plt.figure(figsize=(10, 6))
plt.barh(features[::-1], scores[::-1], color='skyblue')
plt.xlabel('Importance (Gain)')
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
Future Work
Improve recall for class 1 (churners) to better catch at-risk customers.

Build a web app for real-time churn prediction.

Continue feature engineering to boost model performance.

Author
Naman Maheshwari
GitHub: @naman458
