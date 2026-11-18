
import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Customer Churn Prediction API")

# Load trained model
saved_data = joblib.load("model.pkl")
model = saved_data["model"]
threshold = saved_data["threshold"]


# Customer input details
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Customer Churn API is running!"}


@app.post("/predict")
def predict_churn(customer: Customer):

    data = pd.DataFrame([customer.model_dump()])

    # Feature Engineering
    tenure = data["tenure"].iloc[0]

    data["TenureGroup"] = pd.cut(
        [tenure],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Medium", "Long", "Loyal"]
    )

    data["AverageMonthlySpend"] = (
        data["TotalCharges"] / max(tenure, 1)
    )

    service_columns = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]

    data["TotalServices"] = (
        data[service_columns] == "Yes"
    ).sum(axis=1)

    # Prediction
    probability = model.predict_proba(data)[0][1]

    prediction = "Yes" if probability >= threshold else "No"

    return {
        "churn_probability": round(float(probability), 4),
        "churn_prediction": prediction
    }
