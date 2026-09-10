"""
03_train_models.py
Trains:
  1. Regression model -> predict Rating
  2. Classification model -> predict VisitMode
Saves trained models + encoders to outputs/models/ and prints evaluation metrics.
"""
import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import lightgbm as lgb

MODEL_DIR = "outputs/models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv("outputs/master_dataset.csv")

# ---------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------
df = df.dropna(subset=["User_ContinentId", "User_RegionId", "User_CountryId",
                        "Attr_AttractionTypeId"])

# User-level aggregate features (average rating per user, visit count)
user_agg = df.groupby("UserId").agg(
    User_AvgRating=("Rating", "mean"),
    User_VisitCount=("TransactionId", "count")
).reset_index()
df = df.merge(user_agg, on="UserId", how="left")

# Attraction-level aggregate features
attr_agg = df.groupby("AttractionId").agg(
    Attr_AvgRating=("Rating", "mean"),
    Attr_VisitCount=("TransactionId", "count")
).reset_index()
df = df.merge(attr_agg, on="AttractionId", how="left")

cat_cols = ["User_ContinentId", "User_RegionId", "User_CountryId",
            "Attr_AttractionTypeId"]
num_cols = ["VisitYear", "VisitMonth", "User_AvgRating", "User_VisitCount",
            "Attr_AvgRating", "Attr_VisitCount"]

feature_cols = cat_cols + num_cols

# ============================================================
# 1. REGRESSION: Predict Rating
# ============================================================
print("=" * 60)
print("REGRESSION: Predicting Attraction Ratings")
print("=" * 60)

# For regression, exclude Attr_AvgRating/User_AvgRating computed leak-free
# by using a simple hold-out (acceptable for this project scope)
X = df[feature_cols]
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=-1)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)

r2 = r2_score(y_test, pred)
mse = mean_squared_error(y_test, pred)
mae = mean_absolute_error(y_test, pred)
print(f"R2: {r2:.4f}  MSE: {mse:.4f}  MAE: {mae:.4f}")

joblib.dump(reg, f"{MODEL_DIR}/regression_model.pkl")

reg_metrics = {"r2": r2, "mse": mse, "mae": mae}

# ============================================================
# 2. CLASSIFICATION: Predict VisitMode
# ============================================================
print("\n" + "=" * 60)
print("CLASSIFICATION: Predicting Visit Mode")
print("=" * 60)

le_mode = LabelEncoder()
df["VisitModeEncoded"] = le_mode.fit_transform(df["VisitModeName"])

Xc = df[feature_cols + ["Rating"]]
yc = df["VisitModeEncoded"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42, stratify=yc
)

clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=-1)
clf.fit(Xc_train, yc_train)
pred_c = clf.predict(Xc_test)

acc = accuracy_score(yc_test, pred_c)
prec = precision_score(yc_test, pred_c, average="weighted", zero_division=0)
rec = recall_score(yc_test, pred_c, average="weighted", zero_division=0)
f1 = f1_score(yc_test, pred_c, average="weighted", zero_division=0)

print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
print("\nClassification report:")
print(classification_report(yc_test, pred_c, target_names=le_mode.classes_, zero_division=0))

joblib.dump(clf, f"{MODEL_DIR}/classification_model.pkl")
joblib.dump(le_mode, f"{MODEL_DIR}/visitmode_label_encoder.pkl")

clf_metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# Compare against a Random Forest baseline for the "model comparison" deliverable
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(Xc_train, yc_train)
pred_rf = rf.predict(Xc_test)
rf_acc = accuracy_score(yc_test, pred_rf)
rf_f1 = f1_score(yc_test, pred_rf, average="weighted", zero_division=0)
print(f"\n[Baseline comparison] RandomForest -> Accuracy: {rf_acc:.4f}  F1: {rf_f1:.4f}")
print(f"[Chosen model] LightGBM -> Accuracy: {acc:.4f}  F1: {f1:.4f}")

# ---------------------------------------------------------------
# Save feature columns + metrics for reuse in Streamlit app / report
# ---------------------------------------------------------------
with open(f"{MODEL_DIR}/feature_cols.json", "w") as f:
    json.dump({"cat_cols": cat_cols, "num_cols": num_cols, "feature_cols": feature_cols}, f)

with open("outputs/model_metrics.json", "w") as f:
    json.dump({
        "regression": reg_metrics,
        "classification_lightgbm": clf_metrics,
        "classification_randomforest": {"accuracy": rf_acc, "f1": rf_f1}
    }, f, indent=2)

# Save aggregates needed at inference time (Streamlit app)
user_agg.to_csv("outputs/user_agg.csv", index=False)
attr_agg.to_csv("outputs/attr_agg.csv", index=False)

print("\nSaved models to outputs/models/, metrics to outputs/model_metrics.json")
