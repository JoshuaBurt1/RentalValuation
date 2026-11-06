import glob
import pandas as pd

# Find all CSV files starting with "kijiji_rental_ads" (excluding total.csv)
csv_files = [
    f for f in glob.glob("kijiji_rental_ads_canonical_*.csv") if f != "total_new.csv"
]

if not csv_files:
    print("No matching CSV files found.")
else:
    # Read and concatenate matching CSV files
    combined_df = pd.concat(
        [pd.read_csv(f) for f in csv_files], ignore_index=True
    )

    # Export to total.csv
    combined_df.to_csv("total_new.csv", index=False)
    print(
        f"Successfully combined {len(csv_files)} files into total_new.csv"
    )