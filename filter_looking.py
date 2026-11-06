import pandas as pd

# Load dataset
df = pd.read_csv("total_new_cleaned.csv")

# Filter rows where the 'url' column contains the sequence 'look'
matching_df = df[df["url"].astype(str).str.contains("look", case=False, na=False)]

# Extract matching URLs
matching_urls = matching_df["url"].tolist()
total_matches = len(matching_urls)

# Print summary to console
print(f"Total entries containing 'look' in URL: {total_matches:,}\n")
print("--- Matching URLs ---")
for url in matching_urls:
    print(url)

# Format and output to text file
output_text = f"Total entries containing 'look' in URL: {total_matches:,}\n\n"
output_text += "\n".join(matching_urls)

with open("urls_containing_look.txt", "w") as f:
    f.write(output_text)

print("\nSaved matching URLs to urls_containing_look.txt")