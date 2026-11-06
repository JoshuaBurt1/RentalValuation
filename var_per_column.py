import pandas as pd

# Define column name once
col_name = "mls_number"

# Load the cleaned dataset
df = pd.read_csv("total_cleaned.csv")

# Get unique values
if col_name in df.columns:
    unique_types = df[col_name].dropna().unique()

    print(f"Unique values in '{col_name}' ({len(unique_types)} total):")
    for value in sorted(unique_types):
        print(f"- {value}")
else:
    print(f"Column '{col_name}' not found in total_cleaned.csv")