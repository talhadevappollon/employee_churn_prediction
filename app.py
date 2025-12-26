"""
Employee Churn Prediction App
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Employee Churn Prediction",
    page_icon="👔",
    layout="wide"
)

# Title and description
st.title("👔 Employee Churn Prediction System")
st.markdown("""
This application predicts the likelihood of an employee leaving the company based on various HR metrics.
Fill in the employee information below to get a churn prediction.
""")

# Load the model and feature info
@st.cache_resource
def load_model_and_info():
    if not os.path.exists('churn_model.pkl'):
        st.error("Model not found! Please run 'train_model.py' first.")
        st.stop()
    
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    with open('categorical_info.pkl', 'rb') as f:
        cat_info = pickle.load(f)
    
    return model, feature_names, cat_info

model, feature_names, cat_info = load_model_and_info()
df_sample = cat_info['df_sample']

# Create input form
st.header("Employee Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    gender = st.selectbox("Gender", df_sample['Gender'].unique() if 'Gender' in df_sample.columns else ['Male', 'Female'])
    marital_status = st.selectbox("Marital Status", df_sample['MaritalStatus'].unique() if 'MaritalStatus' in df_sample.columns else ['Single', 'Married', 'Divorced'])
    distance_from_home = st.number_input("Distance From Home (miles)", min_value=0, max_value=50, value=10)

with col2:
    st.subheader("Job Info")
    department = st.selectbox("Department", df_sample['Department'].unique() if 'Department' in df_sample.columns else ['Sales', 'Research & Development', 'Human Resources'])
    job_role = st.selectbox("Job Role", df_sample['JobRole'].unique() if 'JobRole' in df_sample.columns else ['Sales Executive', 'Research Scientist', 'Laboratory Technician'])
    job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2)
    years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)
    years_in_current_role = st.number_input("Years In Current Role", min_value=0, max_value=20, value=2)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
    years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=2)

with col3:
    st.subheader("Compensation & Satisfaction")
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
    salary_hike_pct = st.slider("Percent Salary Hike", min_value=0, max_value=25, value=15)
    stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    
    st.subheader("Performance & Work")
    job_satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=3)
    environment_satisfaction = st.select_slider("Environment Satisfaction", options=[1, 2, 3, 4], value=3)
    relationship_satisfaction = st.select_slider("Relationship Satisfaction", options=[1, 2, 3, 4], value=3)
    work_life_balance = st.select_slider("Work Life Balance", options=[1, 2, 3, 4], value=3)
    performance_rating = st.select_slider("Performance Rating", options=[1, 2, 3, 4], value=3)
    
st.subheader("Additional Info")
col4, col5, col6, col7 = st.columns(4)

with col4:
    education = st.selectbox("Education Level", [1, 2, 3, 4, 5], help="1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor")
    education_field = st.selectbox("Education Field", df_sample['EducationField'].unique() if 'EducationField' in df_sample.columns else ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other'])

with col5:
    business_travel = st.selectbox("Business Travel", df_sample['BusinessTravel'].unique() if 'BusinessTravel' in df_sample.columns else ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
    overtime = st.selectbox("Overtime", df_sample['OverTime'].unique() if 'OverTime' in df_sample.columns else ['Yes', 'No'])

with col6:
    num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=8)

with col7:
    training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2)
    daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, value=800)
    hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=100, value=65)
    monthly_rate = st.number_input("Monthly Rate ($)", min_value=2000, max_value=27000, value=14000)

# Predict button
if st.button("🔮 Predict Churn Risk", type="primary"):
    # Create input dataframe
    input_data = {
        'Age': age,
        'DailyRate': daily_rate,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'HourlyRate': hourly_rate,
        'JobLevel': job_level,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies_worked,
        'PercentSalaryHike': salary_hike_pct,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager,
        'EnvironmentSatisfaction': environment_satisfaction,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'BusinessTravel': business_travel,
        'Department': department,
        'EducationField': education_field,
        'Gender': gender,
        'JobRole': job_role,
        'MaritalStatus': marital_status,
        'OverTime': overtime,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Get categorical columns from the sample
    categorical_cols = cat_info['categorical_cols']
    
    # One-hot encode (same as training)
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # Align with training features
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[feature_names]
    
    # Make prediction
    prediction = model.predict(input_encoded)[0]
    prediction_proba = model.predict_proba(input_encoded)[0]
    
    # Display results
    st.markdown("---")
    st.header("Prediction Results")
    
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        if prediction == 1:
            st.error("⚠️ **HIGH RISK**: This employee is likely to leave")
            st.markdown(f"**Churn Probability: {prediction_proba[1]:.1%}**")
        else:
            st.success("✅ **LOW RISK**: This employee is likely to stay")
            st.markdown(f"**Retention Probability: {prediction_proba[0]:.1%}**")
    
    with col_result2:
        st.subheader("Probability Breakdown")
        prob_df = pd.DataFrame({
            'Outcome': ['Will Stay', 'Will Leave'],
            'Probability': [prediction_proba[0], prediction_proba[1]]
        })
        st.bar_chart(prob_df.set_index('Outcome'))
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    if prediction == 1:
        st.markdown("""
        **Actions to reduce churn risk:**
        - Schedule a one-on-one meeting to understand concerns
        - Review compensation and benefits package
        - Discuss career development opportunities
        - Consider flexible work arrangements
        - Improve work-life balance initiatives
        - Provide additional training or mentorship
        """)
    else:
        st.markdown("""
        **Actions to maintain engagement:**
        - Continue regular check-ins
        - Recognize and reward good performance
        - Provide growth opportunities
        - Maintain open communication channels
        """)

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses a Decision Tree Classifier trained on HR data to predict employee churn.
    
    **Model Features:**
    - Max Depth: 4
    - Min Samples per Leaf: 30
    - Algorithm: Decision Tree
    
    **Key Factors:**
    - Job satisfaction levels
    - Compensation metrics
    - Work experience
    - Work-life balance
    - Career progression
    """)
    
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("""
    1. Fill in employee information
    2. Click 'Predict Churn Risk'
    3. Review prediction and recommendations
    """)
