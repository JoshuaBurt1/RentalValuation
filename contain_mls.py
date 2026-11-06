import pandas as pd

# Load the dataset
df = pd.read_csv("total_cleaned.csv")

# Mask for non-empty mls_number entries
has_mls = df["mls_number"].notna() & (df["mls_number"].astype(str).str.strip() != "")

# Filter dataframe for entries WITH an MLS number
df_with_mls = df[has_mls]

# Extract URLs
urls_with_mls = df_with_mls["url"].dropna().tolist()
total_mls_count = len(urls_with_mls)

# Print total count and URLs to console
print(f"Total entries WITH an MLS number: {total_mls_count:,}\n")
print("--- URLs with MLS Number ---")
for url in urls_with_mls:
    print(url)

# Save output to text file
output_text = f"Total entries WITH an MLS number: {total_mls_count:,}\n\n"
output_text += "\n".join(urls_with_mls)

with open("urls_with_mls.txt", "w") as f:
    f.write(output_text)

print("\nSaved URL list to urls_with_mls.txt")