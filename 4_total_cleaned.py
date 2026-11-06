import numpy as np
import pandas as pd


def calculate_haversine_distance(lat1, lon1, lat2=43.651070, lon2=-79.347015):
    """Calculates Haversine distance in kilometers from downtown Toronto core."""
    R = 6371.0
    dlat = np.radians(lat1 - lat2)
    dlon = np.radians(lon1 - lon2)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat2))
        * np.cos(np.radians(lat1))
        * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# Load the dataset
df = pd.read_csv("total_cleaned.csv")

# Filter 'forrentbyhousing' to keep ONLY 'Professional' or 'Owner'
if "forrentbyhousing" in df.columns:
    df = df[df["forrentbyhousing"].isin(["Professional", "Owner"])]

# Ensure columns are numeric, converting non-numeric values to NaN
df["Price($)"] = pd.to_numeric(df["Price($)"], errors="coerce")
df["areainfeet"] = pd.to_numeric(df["areainfeet"], errors="coerce")

# Drop rows where 'Price($)' is NaN (missing) or equal to 0
df = df[df["Price($)"].notna() & (df["Price($)"] != 0)]

# --- NEW FILTER: Drop rows where Price($) is >= 95,000 ---
df = df[df["Price($)"] < 95000]

# Drop rows where 'areainfeet' is NaN (missing) or equal to 0
df = df[df["areainfeet"].notna() & (df["areainfeet"] != 0)]

# Clean 'numberbedrooms' column (detect 'den' and add +0.5)
if "numberbedrooms" in df.columns:
    has_den = (
        df["numberbedrooms"].astype(str).str.contains("den", case=False, na=False)
    )

    base_beds = (
        df["numberbedrooms"]
        .astype(str)
        .str.replace("Bachelor/Studio", "0", case=False, regex=False)
        .str.extract(r"(\d+)")
    )
    
    # Cast to float64 from the start
    base_numeric = pd.to_numeric(base_beds[0], errors="coerce").fillna(0).astype(float)
    
    # Assign back float values directly without triggering .loc dtype warnings
    df["numberbedrooms"] = base_numeric + np.where(has_den, 0.5, 0.0)

# Clean 'numberparkingspots' column: extract leading digits and fill missing/NaN/empty with 0
if "numberparkingspots" in df.columns:
    df["numberparkingspots"] = (
        df["numberparkingspots"].astype(str).str.extract(r"(\d+)")
    )
    df["numberparkingspots"] = (
        pd.to_numeric(df["numberparkingspots"], errors="coerce").fillna(0)
    )

# Clean 'numberbathrooms' column: extract numeric digits (handles floats like 1.5 or integers like 6+)
if "numberbathrooms" in df.columns:
    df["numberbathrooms"] = (
        df["numberbathrooms"].astype(str).str.extract(r"(\d+\.?\d*)")
    )
    df["numberbathrooms"] = pd.to_numeric(df["numberbathrooms"], errors="coerce")

# Drop rows where 'areainfeet' is less than 50
df = df[df["areainfeet"] >= 50]

# --- Clean and drop missing Walk, Transit, and Bike scores ---
score_cols = ["walkscore", "transit score", "bike score"]
for col in score_cols:
    if col in df.columns:
        # Extract numeric digits if stored as string (e.g., "85 / 100")
        df[col] = pd.to_numeric(
            df[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )

# Drop rows where ANY of the score columns are NaN / empty
existing_score_cols = [col for col in score_cols if col in df.columns]
if existing_score_cols:
    df = df.dropna(subset=existing_score_cols)

# Standardize 'unittype' values to proper capitalized labels
unit_type_mapping = {
    "apartment": "Apartment",
    "basement-apartment": "Basement",
    "condo": "Condo",
    "duplex-triplex": "Duplex/Triplex",
    "house": "House",
    "townhouse": "Townhouse",
}
if "unittype" in df.columns:
    df["unittype"] = df["unittype"].replace(unit_type_mapping)

# Drop specified columns if they exist in the dataframe (including dateavailable)
cols_to_drop = [
    "Title",
    "Address",
    "srplogourl",
    "virtualtour",
    "videochat",
    "termagreement",
    "rentalsvirtualoptions",
    "onlineapp",
    "dateavailable",
    "rentalbadge",
]
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Fill missing/NaN values with "No" for specified amenity columns
amenity_cols = [
    "accessiblewashroomsinsuite",
    "airconditioning",
    "audioprompts",
    "balcony",
    "barrierfreeentrancesandramps",
    "bicycleparking",
    "braillelabels",
    "cabletv",
    "concierge",
    "dishwasher",
    "elevator",
    "fridgefreezer",
    "furnished",
    "gym",
    "heat",
    "hydro",
    "internet",
    "laundryinbuilding",
    "laundryinunit",
    "petsallowed",
    "pool",
    "smokingpermitted",
    "storagelocker",
    "twentyfourhoursecurity",
    "visualaids",
    "water",
    "wheelchairaccessible",
    "yard",
]
existing_amenity_cols = [col for col in amenity_cols if col in df.columns]
df[existing_amenity_cols] = df[existing_amenity_cols].fillna("No")

# Calculate 'citycentredistance' using latitude and longitude in 'LocationCoordinates'
if "LocationCoordinates" in df.columns:
    coords = df["LocationCoordinates"].astype(str).str.split(",", expand=True)
    lat = pd.to_numeric(coords[0], errors="coerce")
    lon = pd.to_numeric(coords[1], errors="coerce")
    df["citycentredistance"] = calculate_haversine_distance(lat, lon)

# Save the cleaned dataframe to a new CSV file
df.to_csv("total_cleaned.csv", index=False)

print("Cleaning complete. Output saved to 'total_cleaned.csv'.")

# ==============================================================================
# MISSING VALUE CHECKER
# ==============================================================================

def check_missing_with_lines(file_path, column_name=None):
    # Load dataset treating empty cells and common N/A strings as missing values
    df_check = pd.read_csv(
        file_path, 
        na_values=['N/A', 'n/a', 'NA', 'na', 'null', 'Null', 'NULL', 'None', '']
    )
    
    def get_lines_for_column(col):
        # Index + 2 accounts for 0-indexing + header row at line 1
        missing_mask = df_check[col].isna()
        line_numbers = (missing_mask[missing_mask].index + 2).tolist()
        return line_numbers

    # Check a single specific column
    if column_name:
        if column_name not in df_check.columns:
            print(f"Error: Column '{column_name}' not found.")
            return
        
        lines = get_lines_for_column(column_name)
        print(f"\n--- Column Analysis: '{column_name}' ---")
        if not lines:
            print("No missing or N/A values found.")
        else:
            print(f"Found {len(lines)} missing value(s) at CSV Line Numbers:")
            print(lines)
        return lines

    # Check all columns
    print("\n--- Dataset Missing Value Line Numbers ---")
    found_any = False
    for col in df_check.columns:
        lines = get_lines_for_column(col)
        if lines:
            found_any = True
            print(f"\nColumn: '{col}' ({len(lines)} missing entries)")
            print(f"  Line numbers: {lines if len(lines) <= 20 else lines[:20] + ['... (' + str(len(lines)-20) + ' more)']}")
            
    if not found_any:
        print("No missing or N/A values found in any column!")


# Execute missing value check on the cleaned CSV output
csv_file = 'total_cleaned.csv'
check_missing_with_lines(csv_file)