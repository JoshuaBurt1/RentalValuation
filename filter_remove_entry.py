import pandas as pd

# Target URL to remove
target_url = "https://www.kijiji.ca/v-apartments-condos/city-of-toronto/king-s-landing-3-bed-den-3-5-bath-two-storey-condo/1745700183"

# Load dataset
df = pd.read_csv("total_new.csv")
initial_count = len(df)

# Filter out the specific row matching the URL
df_cleaned = df[df["url"].astype(str).str.strip() != target_url]
final_count = len(df_cleaned)

# Save to CSV
df_cleaned.to_csv("total_new.csv", index=False)

# Print status
print(f"Initial row count: {initial_count:,}")
print(f"Rows removed:      {initial_count - final_count:,}")
print(f"Final row count:   {final_count:,}")