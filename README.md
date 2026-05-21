# HR Employee Attrition Predictor

A complete end-to-end Machine Learning system that predicts 
whether an employee is at risk of leaving a company.

## Project Structure
- `data/` — IBM HR Analytics dataset from Kaggle
- `notebooks/` — EDA, Feature Engineering, Model Training
- `models/` — Saved model, scaler and feature names
- `app.py` — Streamlit deployment app

## ML Pipeline
1. Exploratory Data Analysis (EDA)
2. Feature Engineering
3. XGBoost Model Training
4. Hyperparameter Optimization (GridSearchCV)
5. Model Evaluation (Recall, AUC-ROC)
6. Streamlit Deployment

## Results
- Accuracy: 80%
- AUC-ROC: 0.7769
- Recall (Leavers): 51%

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
Python, XGBoost, Scikit-learn, Pandas, Streamlit