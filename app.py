"""
Employee Churn Prediction App
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

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
    
    # Display results with tabs
    st.markdown("---")
    st.header("Prediction Results")
    
    tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🌳 Decision Path", "💡 Recommendations"])
    
    with tab1:
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
    
    with tab2:
        st.subheader("🌳 Decision Tree Path Explanation")
        st.markdown("This shows how the model arrived at its decision by following the decision tree.")
        
        # Get decision path
        decision_path = model.decision_path(input_encoded)
        node_indicator = decision_path.toarray()[0]
        node_index = np.where(node_indicator == 1)[0]
        
        # Get tree structure
        tree = model.tree_
        feature = tree.feature
        threshold = tree.threshold
        
        st.markdown("### Path Through the Decision Tree:")
        
        # Display the path
        path_data = []
        for node_id in node_index:
            if node_id == node_index[-1]:  # Leaf node
                if tree.value[node_id][0][1] > tree.value[node_id][0][0]:
                    decision = "🔴 Predict: WILL LEAVE"
                    samples_leave = int(tree.value[node_id][0][1])
                    samples_stay = int(tree.value[node_id][0][0])
                else:
                    decision = "🟢 Predict: WILL STAY"
                    samples_leave = int(tree.value[node_id][0][1])
                    samples_stay = int(tree.value[node_id][0][0])
                
                st.success(f"**Final Decision at Leaf Node {node_id}**")
                st.write(f"{decision}")
                st.write(f"- Training samples that stayed: {samples_stay}")
                st.write(f"- Training samples that left: {samples_leave}")
                
            else:
                feature_name = feature_names[feature[node_id]]
                threshold_value = threshold[node_id]
                sample_value = input_encoded.iloc[0][feature_name]
                
                if sample_value <= threshold_value:
                    comparison = "≤"
                    direction = "⬅️ LEFT (YES)"
                else:
                    comparison = ">"
                    direction = "➡️ RIGHT (NO)"
                
                path_data.append({
                    'Node': node_id,
                    'Question': f"{feature_name} ≤ {threshold_value:.2f}?",
                    'Employee Value': f"{sample_value:.2f}",
                    'Decision': direction
                })
        
        if path_data:
            st.markdown("### Decision Steps:")
            for i, step in enumerate(path_data, 1):
                with st.expander(f"Step {i}: Node {step['Node']}", expanded=True):
                    st.markdown(f"**Question:** {step['Question']}")
                    st.markdown(f"**Employee's Value:** {step['Employee Value']}")
                    st.markdown(f"**Path Taken:** {step['Decision']}")
        
        # Visualize the full tree with the path highlighted
        st.markdown("### Complete Decision Tree Visualization")
        st.info("The tree shows all possible decision paths. Nodes in your decision path are highlighted below.")
        
        # Get nodes in the decision path
        path_nodes = set(node_index)
        leaf_node = node_index[-1]
        
        # Create a custom tree visualization using graphviz
        from sklearn.tree import export_graphviz
        import graphviz
        
        # Create DOT data
        dot_data = export_graphviz(
            model,
            feature_names=feature_names,
            class_names=['Will Stay', 'Will Leave'],
            filled=True,
            rounded=True,
            special_characters=True,
            out_file=None
        )
        
        # Modify DOT to highlight path nodes
        dot_lines = dot_data.split('\n')
        modified_lines = []
        
        for line in dot_lines:
            # Check if this line defines a node
            if '->' not in line and '[label=' in line:
                # Extract node number
                node_num = None
                if line.strip().split()[0].isdigit():
                    node_num = int(line.strip().split()[0])
                
                if node_num is not None:
                    if node_num == leaf_node:
                        # Leaf node - bright green with thick border
                        line = line.replace('fillcolor=', 'fillcolor="#00FF00", penwidth=4.0, color="darkgreen", ')
                    elif node_num in path_nodes:
                        # Path nodes - light green with border
                        line = line.replace('fillcolor=', 'fillcolor="#90EE90", penwidth=3.0, color="green", ')
                    else:
                        # Other nodes - grey
                        line = line.replace('fillcolor=', 'fillcolor="#E8E8E8", ')
            
            modified_lines.append(line)
        
        modified_dot = '\n'.join(modified_lines)
        
        # Render with graphviz
        try:
            graph = graphviz.Source(modified_dot)
            st.graphviz_chart(modified_dot)
        except:
            # Fallback to matplotlib if graphviz not available
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(
                model,
                feature_names=feature_names,
                class_names=['Will Stay', 'Will Leave'],
                filled=True,
                rounded=True,
                fontsize=8,
                ax=ax
            )
            plt.tight_layout()
            st.pyplot(fig)
        
        # Add legend explaining the path
        st.success(f"**🟢 Green Path:** Nodes {' → '.join(map(str, node_index))}")
        st.caption(f"The decision goes through {len(node_index)} nodes, ending at leaf node {leaf_node}.")
        
        # Feature importance for this prediction
        st.markdown("### Key Features Influencing This Prediction")
        feature_values = input_encoded.iloc[0]
        feature_importance = model.feature_importances_
        
        # Get top features that were used in the path
        path_features = []
        for node_id in node_index[:-1]:  # Exclude leaf node
            feat_name = feature_names[feature[node_id]]
            path_features.append({
                'Feature': feat_name,
                'Value': feature_values[feat_name],
                'Importance': feature_importance[feature[node_id]]
            })
        
        if path_features:
            path_df = pd.DataFrame(path_features).drop_duplicates(subset=['Feature'])
            path_df = path_df.sort_values('Importance', ascending=False)
            st.dataframe(path_df, use_container_width=True)
    
    with tab3:
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
