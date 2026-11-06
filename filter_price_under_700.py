# Filter 1 - prices under $700 -> 52 entries removed
# $1 listings
# ERRONOUS ENTRIES IN LONG-TERM RENTALS: garage, recording space, parking spaces, storage locker, weekend only rental

# Filter 2 - looking for a room -> 14 entries removed

import pandas as pd

# Load the dataset
df = pd.read_csv("total_new.csv")

# Clean currency symbols/commas and convert to numeric
df["Price($)"] = (
    df["Price($)"]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
)
df["Price($)"] = pd.to_numeric(df["Price($)"], errors="coerce")

# Filter rows where Price($) is between 501 and 600 (inclusive)
price_601_700_df = df[df["Price($)"].notna() & (df["Price($)"] >= 601) & (df["Price($)"] <= 700)]

# Extract all URL values
urls_601_700 = price_601_700_df["url"].dropna()

# Display counts and list out each URL
print(f"Total entries with Price($) between 601 and 700: {len(urls_601_700):,}\n")
print("--- URLs with Price between $601 and $700 ---")
for url in urls_601_700:
    print(url)

# Save the list of URLs to a text file
with open("price_under_601_700.txt", "w") as f:
    f.write(f"Total entries with Price($) between 601 and 700: {len(urls_601_700):,}\n\n")
    for url in urls_601_700:
        f.write(f"{url}\n")

print("\nSaved URL list to urls_601_700.txt")