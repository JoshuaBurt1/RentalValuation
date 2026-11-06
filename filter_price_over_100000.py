import pandas as pd

# Load the dataset
df = pd.read_csv("total_new_mls_calculations.csv")

# Clean currency symbols/commas and convert to numeric
df["Price($)"] = (
    df["Price($)"]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
)
df["Price($)"] = pd.to_numeric(df["Price($)"], errors="coerce")

# Filter rows where Price($) is greater than 10,000
high_price_df = df[df["Price($)"].notna() & (df["Price($)"] > 100000)]

# Extract all URL values
urls_over_100000 = high_price_df["url"].dropna()

# Display counts and list each URL
print(f"Total entries with Price($) > 100000: {len(urls_over_100000):,}\n")
print("--- URLs with Price > $100000 ---")
for url in urls_over_100000:
    print(url)

# Save the list of URLs to a text file
with open("urls_over_100000.txt", "w") as f:
    f.write(f"Total entries with Price($) > 100000: {len(urls_over_100000):,}\n\n")
    for url in urls_over_100000:
        f.write(f"{url}\n")

print("\nSaved URL list to urls_over_100000.txt")