import pandas as pd

# Load the dataset
df = pd.read_csv("total_cleaned.csv")

# Create boolean mask for non-empty MLS numbers
has_mls = df["mls_number"].notna() & (df["mls_number"].astype(str).str.strip() != "") & (df["mls_number"].astype(str).str.lower() != "nan")

# Convert column values to 'yes' or 'no'
df["mls_number"] = df["mls_number"].where(~has_mls, "yes").where(has_mls, "no")

# Get counts
counts = df["mls_number"].value_counts()
print("MLS Number Conversion Summary:")
print(f"Yes (Has MLS): {counts.get('yes', 0):,}")
print(f"No (No MLS):   {counts.get('no', 0):,}")
print(f"Total Rows:    {len(df):,}")

# Save the updated CSV file
df.to_csv("total_cleaned.csv", index=False)
print("\nSuccessfully updated mls_number column in total_cleaned.csv")