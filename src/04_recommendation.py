"""
04_recommendation.py
Builds a collaborative-filtering recommendation system using a
user-item ratings matrix (SVD-based latent factor model), plus a
content-based fallback for cold-start users.
Evaluates with RMSE and MAP@K.
"""
import pandas as pd
import numpy as np
import joblib
import os
import json
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

MODEL_DIR = "outputs/models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv("outputs/master_dataset.csv")
item_dim = pd.read_csv("outputs/item_dim.csv")

# ---------------------------------------------------------------
# Train/test split of transactions (user-based split for eval)
# ---------------------------------------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Build user-item rating matrix from training data (mean if duplicate)
pivot = train_df.pivot_table(index="UserId", columns="AttractionId",
                              values="Rating", aggfunc="mean")
user_ids = pivot.index.tolist()
item_ids = pivot.columns.tolist()

# Fill missing with user's mean rating (mean-centering), keep track of means
user_means = pivot.mean(axis=1)
matrix_filled = pivot.sub(user_means, axis=0).fillna(0).values

# ---------------------------------------------------------------
# Matrix factorization via truncated SVD
# ---------------------------------------------------------------
k = min(20, min(matrix_filled.shape) - 1)
U, sigma, Vt = svds(matrix_filled, k=k)
sigma = np.diag(sigma)
pred_matrix = np.dot(np.dot(U, sigma), Vt) + user_means.values.reshape(-1, 1)

pred_df = pd.DataFrame(pred_matrix, index=user_ids, columns=item_ids)

# ---------------------------------------------------------------
# Evaluate: RMSE on held-out test ratings (only for users/items seen in training)
# ---------------------------------------------------------------
eval_rows = []
for _, row in test_df.iterrows():
    u, i, actual = row["UserId"], row["AttractionId"], row["Rating"]
    if u in pred_df.index and i in pred_df.columns:
        pred_val = np.clip(pred_df.loc[u, i], 1, 5)
        eval_rows.append((actual, pred_val))

if eval_rows:
    actuals, preds = zip(*eval_rows)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    print(f"Collaborative Filtering RMSE (on {len(eval_rows)} test ratings): {rmse:.4f}")
else:
    rmse = None
    print("No overlapping user/item pairs for RMSE eval (cold-start heavy split).")

# ---------------------------------------------------------------
# MAP@K evaluation: for each test user, did we recommend attractions
# they actually rated highly (>=4)?
# ---------------------------------------------------------------
K = 5
def precision_at_k(user_id, k=K):
    if user_id not in pred_df.index:
        return None
    already_seen = set(train_df[train_df.UserId == user_id]["AttractionId"])
    liked_test = set(test_df[(test_df.UserId == user_id) & (test_df.Rating >= 4)]["AttractionId"])
    if not liked_test:
        return None
    recs = pred_df.loc[user_id].drop(labels=already_seen, errors="ignore").sort_values(ascending=False).head(k).index
    hits = len(set(recs) & liked_test)
    return hits / k

scores = [s for s in (precision_at_k(u) for u in test_df["UserId"].unique()[:2000]) if s is not None]
map_at_k = float(np.mean(scores)) if scores else None
print(f"Precision@{K} (proxy for MAP@{K}) over {len(scores)} users: {map_at_k:.4f}" if map_at_k else "Not enough data for MAP@K")

# ---------------------------------------------------------------
# Content-based fallback: recommend attractions of the same type
# the user has rated highly, for cold-start users (not in training matrix)
# ---------------------------------------------------------------
def content_based_recommend(user_id, full_df, item_dim, n=5):
    user_hist = full_df[full_df.UserId == user_id]
    if user_hist.empty:
        # No history: recommend globally most popular highly-rated attractions
        top = (full_df.groupby(["AttractionId", "Attr_Attraction"])["Rating"]
               .agg(["mean", "count"]).reset_index())
        top = top[top["count"] >= 20].sort_values("mean", ascending=False)
        return top.head(n)[["AttractionId", "Attr_Attraction", "mean"]]
    liked_types = user_hist[user_hist.Rating >= 4]["Attr_AttractionTypeId"].astype(str).unique()
    visited = set(user_hist["AttractionId"])
    candidates = item_dim.copy()
    candidates["AttractionTypeId"] = candidates["AttractionTypeId"].astype(str)
    candidates = candidates[candidates["AttractionTypeId"].isin(liked_types)]
    candidates = candidates[~candidates["AttractionId"].isin(visited)]
    return candidates.head(n)[["AttractionId", "Attraction"]]

# Save the SVD components + means for the Streamlit app
joblib.dump({"pred_df": pred_df, "user_means": user_means}, f"{MODEL_DIR}/cf_model.pkl")

with open("outputs/recommendation_metrics.json", "w") as f:
    json.dump({"rmse": rmse, f"precision_at_{K}": map_at_k}, f, indent=2)

print(f"\nSaved collaborative filtering model to {MODEL_DIR}/cf_model.pkl")
print("Saved metrics to outputs/recommendation_metrics.json")

# Quick demo
demo_user = test_df["UserId"].iloc[0]
print(f"\nDemo recommendations for UserId={demo_user}:")
print(content_based_recommend(demo_user, df, item_dim))
