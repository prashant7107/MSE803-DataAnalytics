import pandas as pd
import numpy as np

# Load the dataset from the source file
df = pd.read_csv('clean_dataset.csv')

print("\n--- CLEANED DATASET ---")
print(f"Initial shape: {df.shape}")


print(df)

# STATISTICAL METRICS
print("\n--- STATISTICAL RESULTS ---")
stats_summary = df[['Age', 'Net worth', 'Salary']].describe()
print(stats_summary)

# Additional Specific Statistics
print(f"\nMedian Age: {df['Age'].median()}")
print(f"Median Salary: {df['Salary'].median()}")
print(f"Median Net Worth: {df['Net worth'].median()}")
print(f"Total Net Worth across dataset: {df['Net worth'].sum()}")
print("\nCountry Value Counts:")
print(df['Country'].value_counts(dropna=False))