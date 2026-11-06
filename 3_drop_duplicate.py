import pandas as pd

# Load total.csv and drop duplicate rows across all columns
df = pd.read_csv("total_new.csv")
df_deduped = df.drop_duplicates()

# Save back to total.csv
df_deduped.to_csv("total_new.csv", index=False)

print(
    f"Removed {len(df) - len(df_deduped)} duplicate rows. Total rows remaining: {len(df_deduped)}"
)