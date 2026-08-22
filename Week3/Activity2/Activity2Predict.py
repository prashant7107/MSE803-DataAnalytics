import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# 1. Load the dataset (Data Cleaning & Exploration Phase)
df = pd.read_csv('cleaned_dataset.csv')

# Ensure dates are in datetime format for conversion
if 'Join Date' in df.columns:
    df['Join Date Num'] = pd.to_datetime(df['Join Date'], errors='coerce').apply(
        lambda x: x.toordinal() if pd.notnull(x) else np.nan
    )

# --- UTILITY PREDICTION & GRAPHING FUNCTION ---
def execute_regression(target, predictor, missing_condition, entity_name):
    """Handles Supervised Learning (Regression) and Visualizations."""
    df_temp = df.copy()
    
    # Set numeric columns for regression math
    y_col = target + ' Num' if 'Date' in target else target
    x_col = predictor + ' Num' if 'Date' in predictor else predictor
    
    # Isolate training data
    train_data = df_temp.dropna(subset=[x_col, y_col])
    if train_data.empty:
        print(f"Not enough data to train {entity_name}.")
        return
        
    X_train = train_data[[x_col]].values
    y_train = train_data[y_col].values
    
    # Isolate the missing record to predict
    missing_data = df_temp[missing_condition]
    if missing_data.empty or pd.isna(missing_data[x_col].values[0]):
        print(f"Missing predictor variable for {entity_name}. Cannot predict.")
        return
    X_missing = missing_data[[x_col]].values

    # 1. Linear Regression Model
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    pred_lin = lin_model.predict(X_missing)[0]

    # 2. Polynomial Regression Model (Degree = 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    pred_poly = poly_model.predict(poly.transform(X_missing))[0]

    # Format predictions (Convert dates back, round numbers to 2 decimals)
    if 'Date' in target:        
        val_lin = pd.Timestamp.fromordinal(int(pred_lin)).strftime('%Y-%m-%d')
        val_poly = pd.Timestamp.fromordinal(int(pred_poly)).strftime('%Y-%m-%d')

        # 1. ASSIGN THE PREDICTED VALUE BACK TO THE DATAFRAME
        df.loc[missing_condition, target] = val_poly
        df.loc[missing_condition, y_col] = pred_poly

    else:
        val_lin = f"{pred_lin:.2f}"
        val_poly = f"{pred_poly:.2f}"

        # 1. ASSIGN THE PREDICTED VALUE BACK TO THE DATAFRAME
        df.loc[missing_condition, target] = float(val_poly)

    print(f"--- PREDICTION FOR {entity_name.upper()} ---")
    print(f"Target: {target} | Predictor Used: {predictor}")
    print(f"Linear Prediction: {val_lin}")
    print(f"Polynomial (Deg=2) Prediction: {val_poly}\n")
    
    # SAVE TO CSV IMMEDIATELY AFTER PREDICTION
    # Identify and drop any temporary '* Num' columns so the CSV remains clean
    temp_cols = [col for col in df.columns if 'Num' in col]
    df.drop(columns=temp_cols, errors='ignore').to_csv('predicted_data.csv', index=False)
    print(f"-> Successfully saved updated dataset to 'predicted_data.csv'\n")
   
    # Data Visualization (Plotting curves)
    X_plot = np.linspace(X_train.min() * 0.95, X_train.max() * 1.05, 100).reshape(-1, 1)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, color='blue', s=50, label='Historical Data')
    plt.plot(X_plot, lin_model.predict(X_plot), color='green', linestyle='--', label='Linear Fit')
    plt.plot(X_plot, poly_model.predict(poly.transform(X_plot)), color='red', label='Polynomial Fit')
    
    plt.scatter(X_missing, pred_lin, color='lightgreen', marker='s', s=100, label=f'Linear Pred: {val_lin}')
    plt.scatter(X_missing, pred_poly, color='darkred', marker='*', s=150, label=f'Poly Pred: {val_poly}')
    
    plt.title(f'Predicting {target} via {predictor} for {entity_name}')
    plt.xlabel(predictor)
    plt.ylabel(target)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
    
    return float(pred_poly) if 'Date' not in target else pred_poly

# --- SPECIFIC PREDICTION FUNCTIONS ---

def predict_bob_age():
    # Predict Bob (ID 2) missing Age using Join Date
    execute_regression('Age', 'Join Date', (df['ID'] == 2.0) & (df['Age'].isna()), 'Bob (Age)')

def predict_bob_salary():
    # Predict Bob (ID 2) missing Salary using Age
    execute_regression('Salary', 'Age', (df['ID'] == 2.0) & (df['Salary'].isna()), 'Bob (Salary)')

def predict_charlie_join_date():
    # Predict Charlie (ID 4) missing Join Date using Net Worth
    execute_regression('Join Date', 'Net worth', (df['ID'] == 4.0), 'Charlie (Join Date)')

def predict_david_net_worth():
    # Predict David (ID 5) missing Net Worth using Join Date
    execute_regression('Net worth', 'Join Date', (df['ID'] == 5.0), 'David (Net Worth)')

def predict_eve_join_date():
    # Predict Eve (ID NaN) missing Join Date using Salary
    execute_regression('Join Date', 'Salary', (df['Name'] == 'Eve'), 'Eve (Join Date)')

def predict_grace_country():
    # Classification task: Predict Country using Salary & Join Date
    # Because Country is qualitative, we use Logistic Regression (Classification) rather than Polynomial Regression.
    df_temp = df.copy()
    train_data = df_temp.dropna(subset=['Salary', 'Join Date Num', 'Country'])
    
    X_train = train_data[['Salary', 'Join Date Num']].values
    y_train = train_data['Country'].values
    
    grace_data = df_temp[df_temp['Name'] == 'Grace']
    X_missing = grace_data[['Salary', 'Join Date Num']].values
    
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    pred_country = log_model.predict(X_missing)[0]
    
    print(f"--- PREDICTION FOR GRACE (COUNTRY) ---")
    print(f"Task: Classification (Logistic Regression)")
    print(f"Predicted Country: {pred_country}\n")
    # Note: 2D visualizations are not possible here as there are two predictors (requires a 3D plot).

def predict_heidi_pipeline():
    # Sequential Pipeline for Heidi (ID 9)
    print("--- HEIDI PIPELINE INITIATED ---")
    # 1. Predict Age using Join Date
    heidi_age = execute_regression('Age', 'Join Date', (df['Name'] == 'Heidi'), 'Heidi (Age)')
    
    # Temporarily inject Heidi's predicted polynomial age to use as a predictor for the next step
    if heidi_age:
        df.loc[df['Name'] == 'Heidi', 'Age'] = heidi_age
    
    # 2. Predict Salary using the newly predicted Age
    heidi_sal = execute_regression('Salary', 'Age', (df['Name'] == 'Heidi'), 'Heidi (Salary)')
    
    if heidi_sal:
         df.loc[df['Name'] == 'Heidi', 'Salary'] = heidi_sal
         
    # 3. Predict Net worth using Join Date (as requested in prompt)
    execute_regression('Net worth', 'Join Date', (df['Name'] == 'Heidi'), 'Heidi (Net Worth)')


# ==============================================================
# CALL FUNCTIONS ONE BY ONE (Uncomment the one you wish to test)
# ==============================================================

#predict_bob_age()
#predict_bob_salary()
#predict_charlie_join_date()
#predict_david_net_worth()
#predict_eve_join_date()
#predict_grace_country()
predict_heidi_pipeline()