import pandas as pd
import numpy as np

# Load the dataset from the source file
df = pd.read_csv('Sample_dataset.csv')

print("--- ORIGINAL DATA SUMMARY ---")
print(f"Initial shape: {df.shape}")

# 1. Clean 'Net worth': Remove commas and convert to numeric
df['Net worth'] = df['Net worth'].astype(str).str.replace(',', '').str.strip()
df['Net worth'] = pd.to_numeric(df['Net worth'], errors='coerce')

# 2. Clean 'Salary': Map textual numbers to digits and convert to numeric
salary_mapping = {'sixty five thousand': 65000}
df['Salary'] = df['Salary'].replace(salary_mapping)
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

# 3. Clean 'Age': Map textual ages to digits and convert to numeric
age_mapping = {'thirty-eight': 38}
df['Age'] = df['Age'].replace(age_mapping)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# 4. Standardize Country Codes
df['Country'] = df['Country'].replace({'AU': 'AUS'})

# 5. Handle Join Dates (coerce invalid dates like month 13)
df['Join Date'] = pd.to_datetime(df['Join Date'], errors='coerce')


# Handle missing values for Country, Net worth, and Salary
df['Country'] = df['Country'].fillna('Unknown')
df['Net worth'] = df['Net worth'].fillna(df['Net worth'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())


df.to_csv('cleaned_dataset.csv', index=False)
print("\n--- CLEANED DATASET ---")
print(df)

# --- GENERATING STATISTICAL METRICS ---
print("\n--- STATISTICAL RESULTS ---")
stats_summary = df[['Age', 'Net worth', 'Salary']].describe()
print(stats_summary)
stats_summary.to_csv('statistical_summary.csv')
# Additional Specific Statistics
print(f"\nMedian Age: {df['Age'].median()}")
print(f"Median Salary: {df['Salary'].median()}")
print(f"Median Net Worth: {df['Net worth'].median()}")
print(f"Total Net Worth across dataset: {df['Net worth'].sum()}")
print("\nCountry Value Counts:")
print(df['Country'].value_counts(dropna=False))