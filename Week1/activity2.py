from ucimlrepo import fetch_ucirepo

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# Combine into a single dataframe to check for duplicates easily
df = X.copy()
target_col = y.columns[0]
df[target_col] = y[target_col]

# features and classes available
print("--- Features and Classes ---")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {list(X.columns)}")
print(f"Number of classes: {y[target_col].nunique()}")
print(f"Class names: {list(y[target_col].unique())}\n")

# records in each class
print("--- Records per Class ---")
print(y[target_col].value_counts())
print()

# duplicate records in the dataset
duplicates = df[df.duplicated()]
print("--- Duplicate Records ---")
print(f"Number of duplicate records: {duplicates.shape[0]}")
if duplicates.shape[0] > 0:
  print("Duplicate rows:")
  print(duplicates)