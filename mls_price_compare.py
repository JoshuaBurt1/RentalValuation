import numpy as np
import pandas as pd

# Helper function to compute all requested metrics across levels
def calculate_3_levels(series, df_subset):
    # --- LEVEL 1: Original Values (after initial filters) ---
    count_l1 = series.count()
    mean_l1 = series.mean()
    median_l1 = series.median()
    min_l1 = series.min()
    max_l1 = series.max()
    max_url_l1 = df_subset.loc[series.idxmax(), "url"] if count_l1 > 0 else "N/A"
    range_l1 = max_l1 - min_l1
    var_l1 = series.var()
    std_l1 = series.std()
    
    q1_l1 = series.quantile(0.25)
    q3_l1 = series.quantile(0.75)
    iqr_l1 = q3_l1 - q1_l1
    
    # Thresholds
    lower_1_5 = q1_l1 - 1.5 * iqr_l1
    upper_1_5 = q3_l1 + 1.5 * iqr_l1
    lower_3_0 = q1_l1 - 3.0 * iqr_l1
    upper_3_0 = q3_l1 + 3.0 * iqr_l1
    
    low_outliers_1_5 = (series < lower_1_5).sum()
    high_outliers_1_5 = (series > upper_1_5).sum()
    low_outliers_3_0 = (series < lower_3_0).sum()
    high_outliers_3_0 = (series > upper_3_0).sum()
    
    mad_l1 = (series - mean_l1).abs().mean()
    medad_l1 = (series - median_l1).abs().median()

    # --- LEVEL 2: Removing Extreme Outliers [Q1 - 3*IQR, Q3 + 3*IQR] ---
    series_l2 = series[(series >= lower_3_0) & (series <= upper_3_0)]
    count_l2 = series_l2.count()
    mean_l2 = series_l2.mean()
    median_l2 = series_l2.median()
    min_l2 = series_l2.min()
    max_l2 = series_l2.max()
    range_l2 = max_l2 - min_l2
    var_l2 = series_l2.var()
    std_l2 = series_l2.std()
    mad_l2 = (series_l2 - mean_l2).abs().mean()
    medad_l2 = (series_l2 - median_l2).abs().median()

    # --- LEVEL 3: Removing Outliers [Q1 - 1.5*IQR, Q3 + 1.5*IQR] ---
    series_l3 = series[(series >= lower_1_5) & (series <= upper_1_5)]
    count_l3 = series_l3.count()
    mean_l3 = series_l3.mean()
    median_l3 = series_l3.median()
    min_l3 = series_l3.min()
    max_l3 = series_l3.max()
    range_l3 = max_l3 - min_l3
    var_l3 = series_l3.var()
    std_l3 = series_l3.std()
    mad_l3 = (series_l3 - mean_l3).abs().mean()
    medad_l3 = (series_l3 - median_l3).abs().median()

    return {
        "l1": {
            "count": count_l1, "mean": mean_l1, "median": median_l1, "min": min_l1, "max": max_l1,
            "max_url": max_url_l1, "range": range_l1, "var": var_l1, "std": std_l1, "q1": q1_l1, "q3": q3_l1,
            "iqr": iqr_l1, "lower_1_5": lower_1_5, "upper_1_5": upper_1_5, "lower_3_0": lower_3_0,
            "upper_3_0": upper_3_0, "low_1_5": low_outliers_1_5, "high_1_5": high_outliers_1_5,
            "low_3_0": low_outliers_3_0, "high_3_0": high_outliers_3_0, "mad": mad_l1, "medad": medad_l1
        },
        "l2": {
            "count": count_l2, "mean": mean_l2, "median": median_l2, "min": min_l2, "max": max_l2,
            "range": range_l2, "var": var_l2, "std": std_l2, "mad": mad_l2, "medad": medad_l2
        },
        "l3": {
            "count": count_l3, "mean": mean_l3, "median": median_l3, "min": min_l3, "max": max_l3,
            "range": range_l3, "var": var_l3, "std": std_l3, "mad": mad_l3, "medad": medad_l3
        }
    }

# Load the dataset
df = pd.read_csv("total_new_mls_calculations.csv")

# Clean currency symbols/commas and convert to numeric
df["Price($)"] = (
    df["Price($)"]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
)
df["Price($)"] = pd.to_numeric(df["Price($)"], errors="coerce")

# ERRONOUS DATASET CORRECTIONS
# Drop NaNs and zeros
df = df[df["Price($)"].notna() & (df["Price($)"] != 0)]
# FILTER -1: Drop prices >= 95000 because there are full price home and condos in the long-term rental dataset.
df = df[(df["Price($)"] < 95000)]
# FILTER 0: car.py -> ERRONOUS ENTRIES: cars: 1 entry removed
# FILTER 1: price_under_700.py -> Prices under $700: 52 entries removed
# ~ $1 listings
# ~ ERRONOUS ENTRIES IN LONG-TERM RENTALS: garage, recording space, parking spaces, storage locker, weekend only rental
# FILTER 2: looking.py -> looking for a room: 14 entries removed

# Create a mask for non-empty mls_number
mls_series = df["mls_number"].astype(str).str.strip().str.lower()

# Define Non-MLS conditions: missing values, empty strings, 'nan', or 'no'
is_non_mls = (
    df["mls_number"].isna() 
    | (mls_series == "") 
    | (mls_series == "nan") 
    | (mls_series == "no")
)

# MLS is everything else (Yes, MLS IDs, numbers, etc.)
has_mls = ~is_non_mls

# Compute 3 levels of stats for both groups
df_with_mls = df[has_mls]
df_without_mls = df[~has_mls]

with_stats = calculate_3_levels(df_with_mls["Price($)"], df_with_mls)
without_stats = calculate_3_levels(df_without_mls["Price($)"], df_without_mls)

# Format section function for output
def format_section(title, data):
    l1 = data["l1"]
    l2 = data["l2"]
    l3 = data["l3"]
    
    return (
        f"==================================================\n"
        f"--- {title} ---\n"
        f"==================================================\n"
        f"[LEVEL 1: Original Filtered Dataset]\n"
        f"Count:                                 {l1['count']:,}\n"
        f"Average (Mean) Price:                  ${l1['mean']:,.2f}\n"
        f"Median Price:                          ${l1['median']:,.2f}\n"
        f"Minimum Price:                         ${l1['min']:,.2f}\n"
        f"Maximum Price:                         ${l1['max']:,.2f}\n"
        f"Max Price URL:                         {l1['max_url']}\n"
        f"Range:                                 ${l1['range']:,.2f}\n"
        f"Variance:                              ${l1['var']:,.2f}\n"
        f"Standard Deviation:                    ${l1['std']:,.2f}\n"
        f"Q1 (25th Percentile):                  ${l1['q1']:,.2f}\n"
        f"Q3 (75th Percentile):                  ${l1['q3']:,.2f}\n"
        f"IQR (Interquartile Range):             ${l1['iqr']:,.2f}\n"
        f"1.5x IQR Fence [Lower, Upper]:         [${l1['lower_1_5']:,.2f}, ${l1['upper_1_5']:,.2f}]\n"
        f"  - Low Outliers (< Lower):            {l1['low_1_5']:,}\n"
        f"  - High Outliers (> Upper):           {l1['high_1_5']:,}\n"
        f"3.0x IQR Fence [Lower, Upper]:         [${l1['lower_3_0']:,.2f}, ${l1['upper_3_0']:,.2f}]\n"
        f"  - Low Extreme Outliers (< Lower):    {l1['low_3_0']:,}\n"
        f"  - High Extreme Outliers (> Upper):   {l1['high_3_0']:,}\n"
        f"Mean Absolute Deviation (MAD):         ${l1['mad']:,.2f}\n"
        f"Median Absolute Deviation (MedAD):       ${l1['medad']:,.2f}\n\n"
        
        f"[LEVEL 2: Extreme Outliers Removed (3.0x IQR)]\n"
        f"Entries Remaining:                     {l2['count']:,}\n"
        f"Average (Mean) Price:                  ${l2['mean']:,.2f}\n"
        f"Median Price:                          ${l2['median']:,.2f}\n"
        f"Minimum Price:                         ${l2['min']:,.2f}\n"
        f"Maximum Price:                         ${l2['max']:,.2f}\n"
        f"Range:                                 ${l2['range']:,.2f}\n"
        f"Variance:                              ${l2['var']:,.2f}\n"
        f"Standard Deviation:                    ${l2['std']:,.2f}\n"
        f"Mean Absolute Deviation (MAD):         ${l2['mad']:,.2f}\n"
        f"Median Absolute Deviation (MedAD):       ${l2['medad']:,.2f}\n\n"
        
        f"[LEVEL 3: All Outliers Removed (1.5x IQR)]\n"
        f"Entries Remaining:                     {l3['count']:,}\n"
        f"Average (Mean) Price:                  ${l3['mean']:,.2f}\n"
        f"Median Price:                          ${l3['median']:,.2f}\n"
        f"Minimum Price:                         ${l3['min']:,.2f}\n"
        f"Maximum Price:                         ${l3['max']:,.2f}\n"
        f"Range:                                 ${l3['range']:,.2f}\n"
        f"Variance:                              ${l3['var']:,.2f}\n"
        f"Standard Deviation:                    ${l3['std']:,.2f}\n"
        f"Mean Absolute Deviation (MAD):         ${l3['mad']:,.2f}\n"
        f"Median Absolute Deviation (MedAD):       ${l3['medad']:,.2f}\n\n"
    )

# Build full output string
output_text = (
    f"Total valid price entries:             {len(df):,}\n"
    f"Count WITH MLS Number:                 {with_stats['l1']['count']:,}\n"
    f"Count WITHOUT MLS Number:              {without_stats['l1']['count']:,}\n\n"
    + format_section("WITH MLS NUMBER", with_stats)
    + format_section("WITHOUT MLS NUMBER", without_stats)
    + f"==================================================\n"
    f"--- COMPARISON SUMMARY ---\n"
    f"==================================================\n"
    f"Level 1 Mean Diff (With - Without):    ${(with_stats['l1']['mean'] - without_stats['l1']['mean']):,.2f}\n"
    f"Level 2 Mean Diff (With - Without):    ${(with_stats['l2']['mean'] - without_stats['l2']['mean']):,.2f}\n"
    f"Level 3 Mean Diff (With - Without):    ${(with_stats['l3']['mean'] - without_stats['l3']['mean']):,.2f}\n"
    f"Level 1 Median Diff (With - Without):  ${(with_stats['l1']['median'] - without_stats['l1']['median']):,.2f}\n"
)

# Print to console
print(output_text)

# Save exact output to text file
with open("mls_price_compare.txt", "w") as f:
    f.write(output_text)

print("Saved output summary to mls_price_compare.txt")