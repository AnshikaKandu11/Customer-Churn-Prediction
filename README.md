# Customer Churn Prediction

This project predicts whether a telecom customer will leave the company (Churn) or not.

### What I have done:
- Data Cleaning and One-Hot Encoding
- Created 4 new features - AvgMonthlySpend, TenureGroup, NumServices, IsLongTermContract
- Trained 3 Models - Logistic Regression, Random Forest, Gradient Boosting
- Compared models using AUC-ROC and Precision-Recall
- Created a REST API using FastAPI with /predict endpoint

### How to run:
pip install -r requirements.txt
python train_model.py
uvicorn api:app --reload
