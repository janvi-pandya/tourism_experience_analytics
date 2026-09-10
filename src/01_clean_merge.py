"""
01_clean_merge.py
Cleans and merges all Tourism Dataset tables into one consolidated,
model-ready dataset.
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading raw tables...")
transaction = pd.read_excel(f"{DATA_DIR}/Transaction.xlsx")
user = pd.read_excel(f"{DATA_DIR}/User.xlsx")
city = pd.read_excel(f"{DATA_DIR}/City.xlsx")
country = pd.read_excel(f"{DATA_DIR}/Country.xlsx")
region = pd.read_excel(f"{DATA_DIR}/Region.xlsx")
continent = pd.read_excel(f"{DATA_DIR}/Continent.xlsx")
mode = pd.read_excel(f"{DATA_DIR}/Mode.xlsx")
atype = pd.read_excel(f"{DATA_DIR}/Type.xlsx")
# Use the fuller Updated_Item (1698 attractions) instead of the 30-row Item.xlsx
item = pd.read_excel(f"{DATA_DIR}/Updated_Item.xlsx")

# ---------------------------------------------------------------
# 1. Basic cleaning of lookup tables (drop placeholder '-' / 0 rows,
#    dedupe, strip whitespace)
# ---------------------------------------------------------------
def clean_lookup(df, name_col, id_col):
    df = df.copy()
    df[name_col] = df[name_col].astype(str).str.strip()
    df = df[df[name_col] != "-"]
    df = df.drop_duplicates(subset=[id_col])
    return df

city = clean_lookup(city, "CityName", "CityId")
country = clean_lookup(country, "Country", "CountryId")
region = clean_lookup(region, "Region", "RegionId")
continent = clean_lookup(continent, "Continent", "ContinentId")
mode = clean_lookup(mode, "VisitMode", "VisitModeId")
atype = atype.drop_duplicates(subset=["AttractionTypeId"])
item = item.drop_duplicates(subset=["AttractionId"])

print(f"Lookup sizes -> city:{len(city)} country:{len(country)} "
      f"region:{len(region)} continent:{len(continent)} mode:{len(mode)}")

# ---------------------------------------------------------------
# 2. Clean Transaction table (the fact table)
# ---------------------------------------------------------------
transaction = transaction.drop_duplicates(subset=["TransactionId"])
# Ratings should be 1-5; drop rows outside that range or missing
transaction = transaction[transaction["Rating"].between(1, 5)]
# Valid months 1-12, valid years reasonable range
transaction = transaction[transaction["VisitMonth"].between(1, 12)]
transaction = transaction[transaction["VisitYear"].between(2000, 2026)]
# Drop rows missing key ids
transaction = transaction.dropna(subset=["UserId", "AttractionId", "VisitMode"])

print(f"Transaction rows after cleaning: {len(transaction)}")

# ---------------------------------------------------------------
# 3. Clean User table -- handle missing CityId
# ---------------------------------------------------------------
user = user.drop_duplicates(subset=["UserId"])
user["CityId"] = user["CityId"].fillna(-1).astype(int)

# ---------------------------------------------------------------
# 4. Clean Item table -- standardize address, drop rows with no type
# ---------------------------------------------------------------
item["Attraction"] = item["Attraction"].astype(str).str.strip()
item["AttractionAddress"] = item["AttractionAddress"].astype(str).str.strip()
item = item.dropna(subset=["AttractionTypeId"])

# ---------------------------------------------------------------
# 5. Build the geography dimension: city -> country -> region -> continent
# ---------------------------------------------------------------
geo = (
    city.merge(country, on="CountryId", how="left", suffixes=("", "_country"))
        .merge(region, on="RegionId", how="left", suffixes=("", "_region"))
        .merge(continent, on="ContinentId", how="left", suffixes=("", "_continent"))
)
geo = geo.rename(columns={"CityName": "City"})
geo_cols = ["CityId", "City", "CountryId", "Country", "RegionId", "Region",
            "ContinentId", "Continent"]
geo = geo[[c for c in geo_cols if c in geo.columns]]

# ---------------------------------------------------------------
# 6. Attach geography to users
# ---------------------------------------------------------------
user_full = user.merge(geo, on="CityId", how="left", suffixes=("", "_geo"))
# prefer the id-based continent/region/country already on user table when geo join misses
for col in ["ContinentId", "RegionId", "CountryId"]:
    if f"{col}_geo" in user_full.columns:
        user_full[col] = user_full[col].fillna(user_full[f"{col}_geo"])

# ---------------------------------------------------------------
# 7. Attach attraction type + city info to items
# ---------------------------------------------------------------
item_full = item.merge(atype, on="AttractionTypeId", how="left")
item_full = item_full.merge(
    city.rename(columns={"CityId": "AttractionCityId", "CityName": "AttractionCity"}),
    on="AttractionCityId", how="left"
)

# ---------------------------------------------------------------
# 8. Master consolidated dataset: transaction + user + item
# ---------------------------------------------------------------
master = (
    transaction
    .merge(user_full.add_prefix("User_").rename(columns={"User_UserId": "UserId"}),
           on="UserId", how="left")
    .merge(item_full.add_prefix("Attr_").rename(columns={"Attr_AttractionId": "AttractionId"}),
           on="AttractionId", how="left")
    .merge(mode.rename(columns={"VisitModeId": "VisitMode", "VisitMode": "VisitModeName"}),
           on="VisitMode", how="left")
)

# Drop transactions where the attraction wasn't found in the item master
before = len(master)
master = master.dropna(subset=["Attr_Attraction"])
print(f"Dropped {before - len(master)} transactions with unmatched AttractionId")

master.to_csv(f"{OUT_DIR}/master_dataset.csv", index=False)
user_full.to_csv(f"{OUT_DIR}/user_dim.csv", index=False)
item_full.to_csv(f"{OUT_DIR}/item_dim.csv", index=False)

print(f"\nFinal master dataset shape: {master.shape}")
print("Columns:", list(master.columns))
print("\nSaved: outputs/master_dataset.csv, outputs/user_dim.csv, outputs/item_dim.csv")
