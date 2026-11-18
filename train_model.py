import pandas as pd

df = pd.read_csv("dataset/customer_churn.csv")

print("Original Dataset Shape:", df.shape)

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

df.drop_duplicates(inplace=True)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)

print("\nData Cleaning Completed Successfully!")


# Feature 1: Tenure Group
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["New", "Medium", "Long", "Loyal"]
)

# Feature 2: Average Monthly Spend
df["AverageMonthlySpend"] = (
    df["TotalCharges"] / df["tenure"].replace(0, 1)
)

# Feature 3: Total Services
service_columns = [
    "PhoneService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies"
]

df["TotalServices"] = (
    df[service_columns] == "Yes"
).sum(axis=1)

print("\nNew Features Created:")
print(df[["TenureGroup", "AverageMonthlySpend",
          "TotalServices"]].head())

print("\nFeature Engineering Completed!")


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# Identify numerical and categorical columns
numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)
print("Preprocessing Setup Completed!")


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# Define 3 machine learning models
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"model__C": [0.1, 1, 10]}
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "model__max_depth": [5, 10, None],
            "model__min_samples_split": [2, 10]
        }
    )
}

results = {}

# Train and tune each model
for name, (model, params) in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    grid = GridSearchCV(
        pipeline,
        params,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    probability = grid.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probability)

    results[name] = (grid.best_estimator_, auc)

    print("\nModel:", name)
    print("Best Parameters:", grid.best_params_)
    print("AUC Score:", round(auc, 4))

print("\nAll Models Trained Successfully!")


from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

print("\nMODEL EVALUATION RESULTS")

for name, (model, auc) in results.items():

    # Predict churn probability
    probabilities = model.predict_proba(X_test)[:, 1]

    # Convert probability into Yes/No prediction
    predictions = (probabilities >= 0.5).astype(int)

    # Calculate evaluation metrics
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    pr_auc = average_precision_score(y_test, probabilities)

    print("\nModel:", name)
    print("ROC-AUC:", round(auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("PR-AUC:", round(pr_auc, 4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

print("\nModel Evaluation Completed!")


import joblib

best_name = "Decision Tree"
best_model = results[best_name][0]

joblib.dump(
    {
        "model": best_model,
        "threshold": 0.5,
        "model_name": best_name
    },
    "model.pkl"
)

print("\nSelected Model:", best_name)
print("Model Saved Successfully!")
