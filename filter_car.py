import pandas as pd

# List of common car brand names to search for in URLs
car_brands = [
    "acura", "audi", "bmw", "buick", "cadillac", "chevrolet", "chevy",
    "chrysler", "dodge", "fiat", "ford", "gmc", "honda", "hyundai",
    "infiniti", "jaguar", "jeep", "kia", "land-rover", "lexus", "lincoln",
    "maserati", "mazda", "mercedes", "mini", "mitsubishi", "nissan",
    "porsche", "ram", "subaru", "tesla", "toyota", "volkswagen", "vw", "volvo"
]

# Create a regex pattern to match any of the brand names as distinct words or within URL paths
pattern = r"|".join([rf"\b{brand}\b" for brand in car_brands])

# Load dataset
df = pd.read_csv("total_new.csv")

# Filter rows where the 'url' column contains any car brand
matching_df = df[df["url"].astype(str).str.contains(pattern, case=False, na=False)]

# Extract matching URLs
urls_with_cars = matching_df["url"].dropna().unique()

# Print summary to console
print(f"Total URLs containing common car brand names: {len(urls_with_cars):,}\n")
print("--- Matching URLs ---")
for url in urls_with_cars:
    print(url)

# Save the list of URLs to a text file
with open("urls_with_car_brands.txt", "w") as f:
    f.write(f"Total URLs containing common car brand names: {len(urls_with_cars):,}\n\n")
    for url in urls_with_cars:
        f.write(f"{url}\n")

print("\nSaved matching URLs to urls_with_car_brands.txt")