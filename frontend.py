import streamlit as st
import pandas as pd
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")
st.title("👨‍💼 Employee Attrition Prediction System")

# TRAIN DATA
st.header("Train Model")

train_file = st.file_uploader("Upload Training CSV", type=["csv"], key="train")

if st.button("Train Model"):
    if train_file is not None:
        files = {"file": train_file.getvalue()}
        res = requests.post(f"{BACKEND_URL}/train", files={"file": train_file})
        st.success(res.json()["message"])
    else:
        st.warning("Please upload training dataset")

# TEST DATA
st.header(" Test Model")

test_file = st.file_uploader("Upload Testing CSV", type=["csv"], key="test")

if st.button("Test Model"):
    if test_file is not None:
        res = requests.post(f"{BACKEND_URL}/test", files={"file": test_file})
        result = res.json()

        metrics = result["metrics"]
        cm = result["confusion_matrix"]

        st.subheader("Model Performance")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", metrics["accuracy"])
        col2.metric("Precision", metrics["precision"])
        col3.metric("Recall", metrics["recall"])
        col4.metric("F1 Score", metrics["f1_score"])

        st.subheader("Confusion Matrix")

        cm_df = pd.DataFrame(
            [
                [cm["actual_no"]["predicted_no"], cm["actual_no"]["predicted_yes"]],
                [cm["actual_yes"]["predicted_no"], cm["actual_yes"]["predicted_yes"]],
            ],
            columns=["Predicted No", "Predicted Yes"],
            index=["Actual No", "Actual Yes"]
        )

        st.table(cm_df)

    else:
        st.warning("Please upload testing dataset")

# SINGLE PREDICTION
st.header("🔮 Predict Employee Attrition")

with st.form("prediction_form"):

    Age = st.number_input("Age", 18, 60, 30)
    DistanceFromHome = st.number_input("Distance From Home", 1, 30, 10)

    Education = st.selectbox("Education",
        ["Below College", "College", "Bachelor", "Master", "Doctor"])

    JobLevel = st.selectbox("Job Level",
        ["Entry", "Junior", "Mid", "Senior", "Executive"])

    MonthlyIncome = st.number_input("Monthly Income", 1000, 50000, 5000)
    NumCompaniesWorked = st.number_input("Companies Worked", 0, 10, 1)
    PercentSalaryHike = st.slider("Percent Salary Hike", 10, 30, 15)

    JobSatisfaction = st.selectbox("Job Satisfaction",
        ["Very Bad", "Bad", "Good", "Excellent"])

    WorkLifeBalance = st.selectbox("Work Life Balance",
        ["Bad", "Average", "Good", "Excellent"])

    TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 5)
    YearsAtCompany = st.number_input("Years at Company", 0, 40, 3)
    YearsWithCurrManager = st.number_input("Years With Current Manager", 0, 20, 2)

    BusinessTravel = st.selectbox("Business Travel",
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

    Department = st.selectbox("Department",
        ["Sales", "Research & Development", "Human Resources"])

    EducationField = st.selectbox("Education Field",
        ["Life Sciences", "Medical", "Marketing", "Technical Degree",
         "Human Resources", "Other"])

    Gender = st.selectbox("Gender", ["Male", "Female"])

    JobRole = st.selectbox("Job Role",
        ["Sales Executive", "Research Scientist", "Laboratory Technician",
         "Manufacturing Director", "Healthcare Representative",
         "Manager", "Sales Representative", "Research Director",
         "Human Resources"])

    MaritalStatus = st.selectbox("Marital Status",
        ["Single", "Married", "Divorced"])

    OverTime = st.selectbox("Over Time", ["Yes", "No"])

    submit = st.form_submit_button("Predict Attrition")

# STRING → INT MAPPING
if submit:

    payload = {
        "Age": Age,
        "DistanceFromHome": DistanceFromHome,
        "Education": ["Below College","College","Bachelor","Master","Doctor"].index(Education) + 1,
        "JobLevel": ["Entry","Junior","Mid","Senior","Executive"].index(JobLevel) + 1,
        "MonthlyIncome": MonthlyIncome,
        "NumCompaniesWorked": NumCompaniesWorked,
        "PercentSalaryHike": PercentSalaryHike,
        "JobSatisfaction": ["Very Bad","Bad","Good","Excellent"].index(JobSatisfaction) + 1,
        "WorkLifeBalance": ["Bad","Average","Good","Excellent"].index(WorkLifeBalance) + 1,
        "TotalWorkingYears": TotalWorkingYears,
        "YearsAtCompany": YearsAtCompany,
        "YearsWithCurrManager": YearsWithCurrManager,
        "BusinessTravel": BusinessTravel,
        "Department": Department,
        "EducationField": EducationField,
        "Gender": Gender,
        "JobRole": JobRole,
        "MaritalStatus": MaritalStatus,
        "OverTime": OverTime
    }

    response = requests.post(f"{BACKEND_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()["prediction"]
        if result == 1:
            st.error("⚠️ Employee is likely to leave the company")
        else:
            st.success("✅ Employee is likely to stay")
