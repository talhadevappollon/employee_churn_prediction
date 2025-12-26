# Employee Churn Prediction Streamlit App

A machine learning application that predicts employee churn risk using a Decision Tree Classifier.

## Files

- `app.py` - Main Streamlit application
- `train_model.py` - Script to train and save the model
- `requirements.txt` - Required Python packages
- `Dept_Pred_HR_Attrition.csv` - Training data
- `Churn Predition.ipynb` - Original analysis notebook

## Setup Instructions

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the model:**
```bash
python train_model.py
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

## Features

- Interactive web interface for entering employee data
- Real-time churn prediction with probability scores
- Visual probability breakdown
- Actionable recommendations based on predictions
- Clean, organized input form with multiple categories

## Model Details

- **Algorithm:** Decision Tree Classifier
- **Max Depth:** 4 (for interpretability)
- **Min Samples per Leaf:** 30 (to avoid overfitting)
- **Key Features:** Job satisfaction, income, work experience, work-life balance, and more

## Usage

1. Fill in employee information across all sections
2. Click "Predict Churn Risk" button
3. View prediction results and recommendations
4. Take appropriate action based on risk level
