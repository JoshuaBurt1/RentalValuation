import pandas as pd

# Load the original CSV dataset
csv_path = "total_new.csv"
txt_path = "price_under_500.txt"

df = pd.read_csv(csv_path)
initial_count = len(df)

# Read the txt file using ',' as delimiter
# Columns expected: 1st column = url, 2nd column = type/category
urls_to_remove = set()

try:
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(",")
            # Ensure there are at least 2 columns
            if len(parts) >= 2:
                url = parts[0].strip()
                category = parts[1].strip().lower()
                
                # If 2nd column is NOT 'room', mark the URL for removal
                if category != "room":
                    urls_to_remove.add(url)
except FileNotFoundError:
    print(f"Error: {txt_path} not found.")

# Filter out the matching URLs from the DataFrame
df_cleaned = df[~df["url"].astype(str).str.strip().isin(urls_to_remove)]
final_count = len(df_cleaned)

# Save the updated DataFrame to a new CSV (or overwrite total_new.csv)
df_cleaned.to_csv("total_new_cleaned.csv", index=False)

# Print results
print(f"Number of entries BEFORE removal: {initial_count:,}")
print(f"Number of non-room entries removed: {len(urls_to_remove):,}")
print(f"Number of entries AFTER removal:  {final_count:,}")