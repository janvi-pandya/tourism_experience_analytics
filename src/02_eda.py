"""
02_eda.py
Exploratory Data Analysis on the consolidated tourism dataset.
Saves charts as PNGs to outputs/figures/ and prints key insights.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv("outputs/master_dataset.csv")

# 1. Rating distribution
plt.figure(figsize=(7, 4))
sns.countplot(x="Rating", data=df, palette="viridis")
plt.title("Distribution of Ratings")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/rating_distribution.png", dpi=120)
plt.close()

# 2. Visit mode distribution
plt.figure(figsize=(7, 4))
order = df["VisitModeName"].value_counts().index
sns.countplot(y="VisitModeName", data=df, order=order, palette="mako")
plt.title("Visit Mode Distribution")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/visitmode_distribution.png", dpi=120)
plt.close()

# 3. Users by continent
plt.figure(figsize=(7, 4))
cont_order = df["User_Continent"].value_counts().index
sns.countplot(y="User_Continent", data=df, order=cont_order, palette="crest")
plt.title("Transactions by User Continent")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/users_by_continent.png", dpi=120)
plt.close()

# 4. Top 10 attraction types by popularity
plt.figure(figsize=(8, 5))
top_types = df["Attr_AttractionType"].value_counts().head(10)
sns.barplot(x=top_types.values, y=top_types.index, palette="flare")
plt.title("Top 10 Attraction Types by Visit Count")
plt.xlabel("Number of Visits")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/top_attraction_types.png", dpi=120)
plt.close()

# 5. Average rating by attraction type (top 10 most-visited types)
avg_rating_type = (
    df[df["Attr_AttractionType"].isin(top_types.index)]
    .groupby("Attr_AttractionType")["Rating"].mean().sort_values(ascending=False)
)
plt.figure(figsize=(8, 5))
sns.barplot(x=avg_rating_type.values, y=avg_rating_type.index, palette="rocket")
plt.title("Average Rating by Attraction Type (Top 10 Most Visited)")
plt.xlabel("Average Rating")
plt.xlim(3.5, 5)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/avg_rating_by_type.png", dpi=120)
plt.close()

# 6. VisitMode vs average rating
plt.figure(figsize=(7, 4))
mode_rating = df.groupby("VisitModeName")["Rating"].mean().sort_values(ascending=False)
sns.barplot(x=mode_rating.values, y=mode_rating.index, palette="viridis")
plt.title("Average Rating by Visit Mode")
plt.xlim(3.5, 5)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/rating_by_visitmode.png", dpi=120)
plt.close()

# 7. Visits over year/month (seasonality)
plt.figure(figsize=(9, 4))
monthly = df.groupby(["VisitYear", "VisitMonth"]).size().reset_index(name="count")
monthly["period"] = monthly["VisitYear"].astype(str) + "-" + monthly["VisitMonth"].astype(str).str.zfill(2)
monthly = monthly.sort_values(["VisitYear", "VisitMonth"])
plt.plot(monthly["period"], monthly["count"], marker="o", markersize=3)
plt.xticks(rotation=90, fontsize=6)
plt.title("Number of Visits Over Time")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/visits_over_time.png", dpi=120)
plt.close()

# 8. Correlation heatmap of numeric-encoded features
num_df = df[["VisitYear", "VisitMonth", "VisitMode", "Rating",
             "User_ContinentId", "Attr_AttractionTypeId"]].dropna()
plt.figure(figsize=(6, 5))
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=120)
plt.close()

print("Saved 8 charts to outputs/figures/")

# ---- Print key textual insights ----
print("\n=== KEY INSIGHTS ===")
print(f"Total transactions: {len(df):,}")
print(f"Unique users: {df['UserId'].nunique():,}")
print(f"Unique attractions: {df['AttractionId'].nunique():,}")
print(f"Average rating overall: {df['Rating'].mean():.2f}")
print(f"\nTop attraction type: {top_types.index[0]} ({top_types.iloc[0]:,} visits)")
print(f"Highest avg-rated type (of top 10 visited): {avg_rating_type.index[0]} ({avg_rating_type.iloc[0]:.2f})")
print(f"\nMost common visit mode: {df['VisitModeName'].value_counts().index[0]}")
print(f"Continent with most visits: {df['User_Continent'].value_counts().index[0]}")
