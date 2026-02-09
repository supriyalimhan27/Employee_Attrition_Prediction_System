Employee Attrition Prediction System

An end-to-end Machine Learning application that predicts whether an employee is likely to leave the company. This project covers the entire ML lifecycle — from data preprocessing and model training to API deployment and interactive frontend prediction.

---

Project Overview

Employee attrition can have a huge impact on business productivity and cost. This system identifies employees at risk of leaving, helping HR teams take proactive measures for retention.

Key features:

-Train a machine learning model using historical employee data
-REST API built with FastAPI
-Interactive Streamlit frontend for uploading data and single employee prediction
-Evaluate performance using standard ML metrics
-Handles both numerical and categorical employee features automatically

---

Tech Stack

-Python
-Pandas, NumPy (for data manipulation)
-Scikit-learn (for machine learning and pipelines)
-FastAPI (backend API)
-Streamlit (frontend interface)
-Pickle (model serialization)
-Uvicorn (server to run FastAPI)

---

Model Details

-Algorithm:- Random Forest Classifier
-Preprocessing:-
  MinMaxScaler for numerical features
  OneHotEncoder for categorical features
-Features Used:-
  Numerical:- Age, DistanceFromHome, Education, JobLevel, MonthlyIncome, NumCompaniesWorked, PercentSalaryHike, JobSatisfaction, WorkLifeBalance, TotalWorkingYears, YearsAtCompany, YearsWithCurrManager
  Categorical:- BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime
-Target Variable:- Attrition (Yes / No)

---

API Endpoints

-POST /train (Upload training CSV and train the model )
-POST /test (Upload testing CSV and evaluate the model )
-POST /predict (Predict attrition for a single employee)

---

Frontend Features (Streamlit)

-Upload training & testing datasets
-Fill interactive form to predict single employee attrition
-Displays prediction result (Yes / No)
-Shows model metrics: Accuracy, Precision, Recall, F1 Score
-Confusion matrix visualization for testing data
-Handles categorical features as strings for easy user input

---

How to Run the Project

-Start FastAPI backend uvicorn app:emp_app --reload
-Start Streamlit frontend streamlit run frontend.py
-Access in browser: API: http://127.0.0.1:8000/ Frontend: http://localhost:8501/

Key Learnings

-Implemented end-to-end ML workflow for employee attrition prediction
-Built robust preprocessing pipelines with numeric scaling and categorical encoding
-Deployed machine learning models using FastAPI and integrated with Streamlit
-Learned to handle real-world deployment issues like file uploads, single input prediction, and backend-frontend communication
-Built an interactive and user-friendly interface for HR teams
