import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# 1. Load and clean real data from Sample_dataset.csv
df = pd.read_csv('cleaned_dataset.csv')


# Filter out rows with missing values in Age or Net worth for training
train_data = df.dropna(subset=['Age', 'Net worth'])
X_train = train_data[['Age']].values
y_train = train_data[['Net worth']].values

# Identify rows where Net worth is missing but Age is available (David)
missing_networth_mask = df['Net worth'].isnull() & df['Age'].notnull()
X_missing = df.loc[missing_networth_mask, ['Age']].values

# ==========================================
# STEP 2: LINEAR REGRESSION IMPUTATION
# ==========================================
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# In-sample predictions and metrics
y_pred_lin = lin_model.predict(X_train)
mse_lin = mean_squared_error(y_train, y_pred_lin)
r2_lin = r2_score(y_train, y_pred_lin)

# Predict missing Net worth using Linear Model
pred_networth_lin = lin_model.predict(X_missing)

# ==========================================
# STEP 3: POLYNOMIAL REGRESSION IMPUTATION (DEGREE = 2)
# ==========================================
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_missing_poly = poly_features.transform(X_missing)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# In-sample predictions and metrics
y_pred_poly = poly_model.predict(X_train_poly)
mse_poly = mean_squared_error(y_train, y_pred_poly)
r2_poly = r2_score(y_train, y_pred_poly)

# Predict missing Net worth using Polynomial Model
pred_networth_poly = poly_model.predict(X_missing_poly)

# ==========================================
# STEP 4: COMPARE PREDICTIONS & IMPUTE
# ==========================================
print("--- MODEL PERFORMANCE EVALUATION ---")
print(f"Linear Model       -> MSE: {mse_lin:.2f}, R2 Score: {r2_lin:.4f}")
print(f"Polynomial (deg=2) -> MSE: {mse_poly:.2f}, R2 Score: {r2_poly:.4f}")
print(f"\nDavid (Age=38) Predicted Net Worth (Linear):     ${pred_networth_lin[0][0]:,.2f}")
print(f"David (Age=38) Predicted Net Worth (Polynomial): ${pred_networth_poly[0][0]:,.2f}")

# Impute David's Net Worth using the polynomial fit
df.loc[missing_networth_mask, 'Net worth'] = pred_networth_poly.flatten()

# For remaining full NaNs (Heidi), fill using dataset medians
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Net worth'] = df['Net worth'].fillna(df['Net worth'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

# Save cleaned output
df.to_csv('cleaned_dataset_generated.csv', index=False)
print("\n--- FINAL CLEANED DATASET ---")
print(df[['ID', 'Name', 'Age', 'Net worth', 'Salary', 'Country']])

# VISUALIZE REGRESSION CURVE
X_plot = np.linspace(X_train.min() - 2, X_train.max() + 2, 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)

plt.figure(figsize=(9, 5))
plt.scatter(X_train, y_train, color='blue', s=60, label='Actual Data Points')
plt.plot(X_plot, lin_model.predict(X_plot), color='green', linestyle='--', label='Linear Fit')
plt.plot(X_plot, poly_model.predict(X_plot_poly), color='red', linewidth=2, label='Polynomial Fit (deg=2)')
plt.scatter(X_missing, pred_networth_poly, color='purple', s=100, marker='*', label='Imputed David (Age=38)')
plt.title('Net Worth vs. Age: Linear vs Polynomial Imputation')
plt.xlabel('Age')
plt.ylabel('Net Worth ($)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()