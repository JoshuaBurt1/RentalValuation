import re
import pandas as pd

# Load the main dataset
df = pd.read_csv("total_cleaned.csv")

# List of target unit types
target_unittypes = [
    "Apartment",
    "Basement",
    "Condo",
    "Duplex/Triplex",
    "House",
    "Townhouse",
]

for unit_type in target_unittypes:
    # Filter the dataframe for the specific unit type
    filtered_df = df[df["unittype"] == unit_type]

    # Clean the unit type string to make a valid filename (e.g., 'Duplex/Triplex' -> 'duplex_triplex')
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", unit_type).lower()
    safe_name = re.sub(r"_+", "_", safe_name).strip("_")

    # Define the output file path
    output_filename = f"total_cleaned_{safe_name}.csv"

    # Export to CSV
    filtered_df.to_csv(output_filename, index=False)
    print(
        f"Saved {len(filtered_df)} records to '{output_filename}' for unittype: {unit_type}"
    )