"""
Clinical Risk Assessment Model for Cardiovascular Disease

Purpose:
To stratify cardiovascular disease risk using demographic, clinical,
and functional indicators, supporting early detection and preventive care.

Context:
This model is designed as a core component of an integrated
Clinical Risk Stratification and Decision Support System.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
data = pd.read_csv("heart_data.csv")

print(data.head())
print(data.columns)
diagnostic_variables = [
    "vessels_colored_by_flourosopy",
    "thalassemia"
]

X = data.drop(["target"] + diagnostic_variables, axis=1)
y = data["target"]
categorical_features = [
    "sex",
    "chest_pain_type",
    "fasting_blood_sugar",
    "rest_ecg",
    "exercise_induced_angina",
    "slope"
]

numerical_features = [
    "age",
    "resting_blood_pressure",
    "cholestoral",
    "Max_heart_rate",
    "oldpeak"
]
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
print(X.columns.tolist())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
log_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("Logistic Regression Results")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
print("Accuracy:", accuracy_score(y_test, y_pred_log))
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        min_samples_split=20,
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Results")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# Get trained Random Forest model
rf = rf_model.named_steps["classifier"]

# Get feature names after preprocessing
feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()

# Create feature importance table
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importance.head(15))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape, X_test.shape)
overlap = set(X_train.index).intersection(set(X_test.index))
print(len(overlap))
for col in X.select_dtypes(include="object"):
    print(col, X[col].unique())

joblib.dump(log_model, "heart_logistic_model.pkl")
joblib.dump(rf_model, "heart_random_forest_model.pkl")