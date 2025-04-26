# Telco-Customer-Churn-Prediction
verview

This project develops a machine learning model to predict customer churn for a telecommunications company using the Telco Customer Churn dataset. The goal is to identify customers likely to leave (churn) and achieve an 18% improvement in retention forecasts compared to a baseline (~70% accuracy). The model uses XGBoost, a powerful gradient-boosting algorithm, with feature engineering and evaluation to guide retention strategies.

Dataset





Source: Telco Customer Churn dataset (not included in this repository due to size).



Size: 7,043 customers, 21 features (e.g., tenure, MonthlyCharges, Contract, Churn).



Target: Churn (0 = stay, 1 = churn).

Methodology

The project follows four steps:





Data Cleaning:





Converted Churn to binary (0/1).



Dropped customerID (non-predictive).



Converted TotalCharges to numeric, filled missing values with mean.



Feature Engineering:





Created HighCharges (1 if MonthlyCharges > 70, else 0).



Created IsLongTerm (1 if tenure > 12, else 0).



Created ChargesPerMonth (TotalCharges / (tenure + 1)).



Encoded categorical features (e.g., gender, Contract) using one-hot encoding (pd.get_dummies).



Model Building:





Used XGBoost with parameters: max_depth=7, learning_rate=0.05, random_state=42.



Split data: 80% training, 20% testing (random_state=42).



Trained model to predict Churn.



Evaluation:





Calculated accuracy and classification report.



Visualized feature importance to identify key churn drivers.

Outcomes

The model was evaluated on the test set (1,409 samples):





Accuracy: 80.8% (1,138/1,409 correct predictions), ~15.4% better than a 70% baseline, approaching the 18% retention forecast boost target (aiming for ~85%).



Classification Report:

               precision    recall  f1-score   support
           0       0.85      0.90      0.87      1036
           1       0.67      0.55      0.60       373
    accuracy                           0.81      1409
   macro avg       0.76      0.72      0.74      1409
weighted avg       0.80      0.81      0.80      1409





Class 0 (stay): High precision (85%) and recall (90%), reflecting strong performance for non-churners (73.6% of test set).



Class 1 (churn): Moderate precision (67%) and recall (55%), indicating room to improve churner identification (26.4% of test set).



Feature Importance:





Visualized using xgboost.plot_importance, showing features like tenure, MonthlyCharges, and Contract_Two year as top churn predictors.



Insight: Short tenure and high charges increase churn likelihood; long-term contracts reduce it.

Requirements





Python 3.8+



Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib



Install dependencies:

pip install pandas numpy scikit-learn xgboost matplotlib

How to Run





Download the dataset:





Get WA_Fn-UseC_-Telco-Customer-Churn.csv from Kaggle.



Place it in the same folder as churn_prediction.py or update the path in the code.



Run the script:

python churn_prediction.py



Output:





Prints sample predictions, accuracy (80.8%), and classification report.



Displays a feature importance plot (note: default plot may be cluttered; see improvements below).

Improvements





Model: Current accuracy (80.8%) is close to the 85% target. Try:





Adjusting scale_pos_weight (e.g., 3) to boost recall for churners.



Adding features (e.g., tenure bins).



Plot: The default xgboost.plot_importance plot is cluttered. A custom Matplotlib plot (top 10 features, gain importance) could improve clarity:

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





Enhance recall for class 1 (churn) to catch more churners, improving retention.



Create a custom feature importance plot for better visualization.



Deploy the model in a web app for real-time predictions.

Author





Naman Maheshwari






GitHub: @naman458
