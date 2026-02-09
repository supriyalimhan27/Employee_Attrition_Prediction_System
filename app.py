from fastapi import FastAPI, UploadFile, File
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score

emp_app = FastAPI(title="Employee Attrition Backend")

MODEL_PATH = "attrition_model.pkl"
TARGET = "Attrition"

NUM_COLS = [
    'Age','DistanceFromHome','Education','JobLevel','MonthlyIncome',
    'NumCompaniesWorked','PercentSalaryHike','JobSatisfaction',
    'WorkLifeBalance','TotalWorkingYears','YearsAtCompany',
    'YearsWithCurrManager'
]

CAT_COLS = [
    'BusinessTravel','Department','EducationField','Gender',
    'JobRole','MaritalStatus','OverTime'
]


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
        ]
    )

    model = RandomForestClassifier(
    n_estimators=260,
    max_depth=9,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


@emp_app.post("/train")
async def train_model(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    return {
        "message": "Model trained successfully",
        "accuracy": round(acc, 4)
    }


@emp_app.post("/test")
async def test_model(file: UploadFile = File(...)):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(file.file)

    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET].map({"Yes": 1, "No": 0})

    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    cm = confusion_matrix(y, y_pred)

    return {
        "metrics": {
            "accuracy": round(acc, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3)
        },
        "confusion_matrix": {
            "actual_no": {
                "predicted_no": int(cm[0][0]),
                "predicted_yes": int(cm[0][1])
            },
            "actual_yes": {
                "predicted_no": int(cm[1][0]),
                "predicted_yes": int(cm[1][1])
            }
        }
    }


@emp_app.post("/predict")
async def predict_attrition(data: dict):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    input_df = pd.DataFrame([data])
    input_df = input_df[NUM_COLS + CAT_COLS]

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    return {
        "prediction": "Yes" if pred == 1 else "No",
        "probability": round(prob, 4)
    }
