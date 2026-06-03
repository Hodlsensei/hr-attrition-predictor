import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load saved model, scaler and feature names ──────────────────
model = joblib.load(os.path.join(BASE_DIR, 'models/hr_attrition_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models/scaler.pkl'))
feature_names = joblib.load(os.path.join(BASE_DIR, 'models/feature_names.pkl'))

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

st.title("👥 HR Employee Attrition Predictor")
st.markdown("Fill in the employee details below to predict whether they are at risk of leaving.")
st.divider()

# ── Input Form ───────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal Info")
    age = st.slider("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    distance_from_home = st.slider("Distance From Home (km)", 1, 29, 5)
    education = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                             format_func=lambda x: {1:"Below College", 2:"College",
                                                     3:"Bachelor", 4:"Master", 5:"Doctor"}[x])

with col2:
    st.subheader(" Job Info")
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ])
    job_level = st.slider("Job Level", 1, 5, 2)
    job_involvement = st.slider("Job Involvement", 1, 4, 3)
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

with col3:
    st.subheader("💰 Compensation & Experience")
    monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
    daily_rate = st.number_input("Daily Rate", 100, 1500, 800)
    hourly_rate = st.number_input("Hourly Rate", 30, 100, 65)
    monthly_rate = st.number_input("Monthly Rate", 2000, 27000, 14000)
    percent_salary_hike = st.slider("Percent Salary Hike", 11, 25, 15)
    stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
    total_working_years = st.slider("Total Working Years", 0, 40, 10)
    years_at_company = st.slider("Years At Company", 0, 40, 5)
    years_in_current_role = st.slider("Years In Current Role", 0, 18, 3)
    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
    years_with_curr_manager = st.slider("Years With Current Manager", 0, 17, 3)

st.divider()

# ── More inputs ──────────────────────────────────────────────────
col4, col5 = st.columns(2)

with col4:
    st.subheader("🎓 Education & Training")
    education_field = st.selectbox("Education Field", [
        "Life Sciences", "Medical", "Marketing",
        "Technical Degree", "Human Resources", "Other"
    ])
    num_companies_worked = st.slider("Num Companies Worked", 0, 9, 2)
    training_times_last_year = st.slider("Training Times Last Year", 0, 6, 3)
    performance_rating = st.selectbox("Performance Rating", [3, 4],
                                       format_func=lambda x: {3:"Excellent", 4:"Outstanding"}[x])

with col5:
    st.subheader("😊 Satisfaction & Balance")
    environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
    work_life_balance = st.slider("Work Life Balance", 1, 4, 3)

st.divider()

# ── Predict Button ───────────────────────────────────────────────
if st.button("🔍 Predict Attrition Risk", use_container_width=True):

    # Build input dict with all raw features
    input_dict = {
        'Age': age,
        'DailyRate': daily_rate,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': 1 if gender == "Male" else 0,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLevel': job_level,
        'JobSatisfaction': job_satisfaction,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies_worked,
        'OverTime': 1 if overtime == "Yes" else 0,
        'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager,

        # Business Travel
        'BusinessTravel_Non-Travel': 1 if business_travel == "Non-Travel" else 0,
        'BusinessTravel_Travel_Frequently': 1 if business_travel == "Travel_Frequently" else 0,
        'BusinessTravel_Travel_Rarely': 1 if business_travel == "Travel_Rarely" else 0,

        # Department
        'Department_Human Resources': 1 if department == "Human Resources" else 0,
        'Department_Research & Development': 1 if department == "Research & Development" else 0,
        'Department_Sales': 1 if department == "Sales" else 0,

        # Education Field
        'EducationField_Human Resources': 1 if education_field == "Human Resources" else 0,
        'EducationField_Life Sciences': 1 if education_field == "Life Sciences" else 0,
        'EducationField_Marketing': 1 if education_field == "Marketing" else 0,
        'EducationField_Medical': 1 if education_field == "Medical" else 0,
        'EducationField_Other': 1 if education_field == "Other" else 0,
        'EducationField_Technical Degree': 1 if education_field == "Technical Degree" else 0,

        # Job Role
        'JobRole_Healthcare Representative': 1 if job_role == "Healthcare Representative" else 0,
        'JobRole_Human Resources': 1 if job_role == "Human Resources" else 0,
        'JobRole_Laboratory Technician': 1 if job_role == "Laboratory Technician" else 0,
        'JobRole_Manager': 1 if job_role == "Manager" else 0,
        'JobRole_Manufacturing Director': 1 if job_role == "Manufacturing Director" else 0,
        'JobRole_Research Director': 1 if job_role == "Research Director" else 0,
        'JobRole_Research Scientist': 1 if job_role == "Research Scientist" else 0,
        'JobRole_Sales Executive': 1 if job_role == "Sales Executive" else 0,
        'JobRole_Sales Representative': 1 if job_role == "Sales Representative" else 0,

        # Marital Status
        'MaritalStatus_Divorced': 1 if marital_status == "Divorced" else 0,
        'MaritalStatus_Married': 1 if marital_status == "Married" else 0,
        'MaritalStatus_Single': 1 if marital_status == "Single" else 0,

        # Engineered features
        'IncomePerYear': monthly_income / (total_working_years + 1),
        'LoyaltyScore': years_at_company / (total_working_years + 1),
        'AvgSatisfaction': np.mean([environment_satisfaction, job_satisfaction,
                                     relationship_satisfaction, work_life_balance])
    }

    # Convert to dataframe and align with training feature order
    input_df = pd.DataFrame([input_dict])[feature_names]

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ── Display Result ───────────────────────────────────────────
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK — This employee is likely to leave")
        st.metric("Probability of Leaving", f"{probability*100:.1f}%")
        st.markdown("**Recommended HR Actions:**")
        st.markdown("-  Schedule a one-on-one retention conversation")
        st.markdown("-  Review compensation and benefits")
        st.markdown("-  Discuss career growth opportunities")
        st.markdown("-  Review overtime workload if applicable")
    else:
        st.success(f"✅ LOW RISK — This employee is likely to stay")
        st.metric("Probability of Leaving", f"{probability*100:.1f}%")
        st.markdown("**Keep up the good work:**")
        st.markdown("-  Continue current engagement practices")
        st.markdown("-  Monitor satisfaction scores periodically")