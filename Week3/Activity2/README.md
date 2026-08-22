# Week 3 - Activity 2: Comprehensive Missing Value Prediction

This activity explores data analysis and data cleaning to predict missing values across multiple records using predictive analytics. By applying learning algorithms (Linear Regression, Polynomial Regression), the script estimates unknown variables.

## Files

- `Activity2Predict.py`: The main Python script containing the predictive regression and classification pipeline.
- `Sample_dataset.csv`: The original raw dataset prior to preprocessing.
- `cleaned_dataset.csv`: The preprocessed and consolidated dataset used as the input training data.
- `predicted_data.csv`: The final dataset, which is continuously updated and saved with newly imputed values after each prediction function executes.
- `bob_age_prediction.png`: Generated linear vs. polynomial comparison chart for Bob's Age.
- `bob_salary_prediction.png`: Generated linear vs. polynomial comparison chart for Bob's Salary.
- `charlie_join date_prediction.png`: Generated linear vs. polynomial comparison chart for Charlie's Join Date.
- `david_networth_prediction.png`: Generated linear vs. polynomial comparison chart for David's Net Worth.
- `eve_joindate_prediction.png`: Generated linear vs. polynomial comparison chart for Eve's Join Date.
- `grace_country_prediction.png`: Generated classification output/chart for Grace's Country.
- `heldi_age_prediction.png`: Generated regression comparison chart for Heidi's Age.
- `heldi_salary_prediction.png`: Generated regression comparison chart for Heidi's Salary.
- `heldi_networth_prediction.png`: Generated regression comparison chart for Heidi's Net Worth.

## Missing Values Predicted

This script implements a sequential prediction pipeline to resolve incomplete data and ensure data quality[cite: 3]:

- **Bob (ID 2.0):** Predicted missing `Age` (using Join Date) and missing `Salary` (using Age).
- **Charlie (ID 4.0):** Predicted missing `Join Date` using `Net worth`.
- **David (ID 5.0):** Predicted missing `Net worth` using `Join Date`.
- **Eve (No ID):** Predicted missing `Join Date` using `Salary`.
- **Grace (ID 8.0):** Predicted missing categorical `Country` using `Salary` and `Join Date` via Logistic Regression.
- **Heidi (ID 9.0):** Executed a chained pipeline predicting `Age` (via Join Date), then `Salary` (via the newly predicted Age), and finally `Net worth` (via Join Date).
