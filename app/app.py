"""
Tourism Experience Analytics -- Streamlit App
Lets users get:
  - A predicted rating for an attraction they're considering
  - A predicted visit mode
  - Personalized attraction recommendations
  - Visualizations of tourism trends
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import os

st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs")


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(OUT, "master_dataset.csv"))
    item_dim = pd.read_csv(os.path.join(OUT, "item_dim.csv"))
    user_agg = pd.read_csv(os.path.join(OUT, "user_agg.csv"))
    attr_agg = pd.read_csv(os.path.join(OUT, "attr_agg.csv"))
    return df, item_dim, user_agg, attr_agg


@st.cache_resource
def load_models():
    reg = joblib.load(os.path.join(OUT, "models", "regression_model.pkl"))
    clf = joblib.load(os.path.join(OUT, "models", "classification_model.pkl"))
    le_mode = joblib.load(os.path.join(OUT, "models", "visitmode_label_encoder.pkl"))
    cf = joblib.load(os.path.join(OUT, "models", "cf_model.pkl"))
    with open(os.path.join(OUT, "models", "feature_cols.json")) as f:
        feat_cfg = json.load(f)
    return reg, clf, le_mode, cf, feat_cfg


df, item_dim, user_agg, attr_agg = load_data()
reg_model, clf_model, le_mode, cf_model, feat_cfg = load_models()
pred_df = cf_model["pred_df"]
user_means = cf_model["user_means"]

st.title("🌍 Tourism Experience Analytics")
st.caption("Personalized recommendations, rating prediction, and visit-mode classification for tourism platforms.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Predict My Trip", "🎯 Recommendations", "📊 Trends & Insights", "ℹ️ About"]
)

# ============================================================
# TAB 1: Prediction (Regression + Classification)
# ============================================================
with tab1:
    st.subheader("Tell us about your trip")

    col1, col2 = st.columns(2)
    with col1:
        continent_opts = df[["User_ContinentId", "User_Continent"]].dropna().drop_duplicates()
        continent_name = st.selectbox("Your continent", sorted(continent_opts["User_Continent"].unique()))
        continent_id = continent_opts[continent_opts.User_Continent == continent_name]["User_ContinentId"].iloc[0]

        region_opts = df[df.User_ContinentId == continent_id][["User_RegionId", "User_Region"]].dropna().drop_duplicates()
        region_name = st.selectbox("Your region", sorted(region_opts["User_Region"].unique()))
        region_id = region_opts[region_opts.User_Region == region_name]["User_RegionId"].iloc[0]

        country_opts = df[df.User_RegionId == region_id][["User_CountryId", "User_Country"]].dropna().drop_duplicates()
        country_name = st.selectbox("Your country", sorted(country_opts["User_Country"].unique()))
        country_id = country_opts[country_opts.User_Country == country_name]["User_CountryId"].iloc[0]

    with col2:
        attr_type_opts = df[["Attr_AttractionTypeId", "Attr_AttractionType"]].dropna().drop_duplicates()
        attr_type_name = st.selectbox("Attraction type you're considering", sorted(attr_type_opts["Attr_AttractionType"].unique()))
        attr_type_id = attr_type_opts[attr_type_opts.Attr_AttractionType == attr_type_name]["Attr_AttractionTypeId"].iloc[0]

        visit_year = st.number_input("Visit year", min_value=2018, max_value=2027, value=2026)
        visit_month = st.selectbox("Visit month", list(range(1, 13)), index=5)

    if st.button("Predict", type="primary"):
        # Build feature row using dataset-wide averages as fallback (new user / new attraction)
        user_avg_rating = df["Rating"].mean()
        user_visit_count = 1
        attr_matches = attr_agg.merge(
            df[["AttractionId", "Attr_AttractionTypeId"]].drop_duplicates(),
            on="AttractionId", how="left"
        )
        type_attrs = attr_matches[attr_matches.Attr_AttractionTypeId == attr_type_id]
        attr_avg_rating = type_attrs["Attr_AvgRating"].mean() if not type_attrs.empty else df["Rating"].mean()
        attr_visit_count = type_attrs["Attr_VisitCount"].mean() if not type_attrs.empty else 1

        features = pd.DataFrame([{
            "User_ContinentId": continent_id,
            "User_RegionId": region_id,
            "User_CountryId": country_id,
            "Attr_AttractionTypeId": attr_type_id,
            "VisitYear": visit_year,
            "VisitMonth": visit_month,
            "User_AvgRating": user_avg_rating,
            "User_VisitCount": user_visit_count,
            "Attr_AvgRating": attr_avg_rating,
            "Attr_VisitCount": attr_visit_count,
        }])

        # --- Regression prediction ---
        reg_features = features[feat_cfg["feature_cols"]]
        pred_rating = np.clip(reg_model.predict(reg_features)[0], 1, 5)

        # --- Classification prediction ---
        clf_features = features[feat_cfg["feature_cols"] + []].copy()
        clf_features["Rating"] = pred_rating
        pred_mode_encoded = clf_model.predict(clf_features[feat_cfg["feature_cols"] + ["Rating"]])[0]
        pred_mode = le_mode.inverse_transform([pred_mode_encoded])[0]

        c1, c2 = st.columns(2)
        c1.metric("⭐ Predicted Rating", f"{pred_rating:.2f} / 5")
        c2.metric("🧳 Predicted Visit Mode", pred_mode)

# ============================================================
# TAB 2: Recommendations
# ============================================================
with tab2:
    st.subheader("Get personalized attraction recommendations")
    user_id_input = st.selectbox(
        "Select a UserId (from historical data) to see recommendations",
        sorted(df["UserId"].unique())[:500]
    )
    n_recs = st.slider("Number of recommendations", 3, 10, 5)

    if st.button("Get Recommendations"):
        if user_id_input in pred_df.index:
            already_seen = set(df[df.UserId == user_id_input]["AttractionId"])
            recs = (pred_df.loc[user_id_input]
                    .drop(labels=already_seen, errors="ignore")
                    .sort_values(ascending=False).head(n_recs))
            rec_df = pd.DataFrame({"AttractionId": recs.index, "PredictedRating": recs.values})
            rec_df = rec_df.merge(item_dim[["AttractionId", "Attraction", "AttractionType"]],
                                   on="AttractionId", how="left")
            st.write("**Collaborative filtering recommendations:**")
            st.dataframe(rec_df[["Attraction", "AttractionType", "PredictedRating"]], hide_index=True)
        else:
            st.info("This user has limited history — showing popular highly-rated attractions instead.")
            top = (df.groupby(["AttractionId", "Attr_Attraction"])["Rating"]
                   .agg(["mean", "count"]).reset_index())
            top = top[top["count"] >= 20].sort_values("mean", ascending=False).head(n_recs)
            st.dataframe(top.rename(columns={"Attr_Attraction": "Attraction", "mean": "AvgRating", "count": "Visits"}),
                         hide_index=True)

# ============================================================
# TAB 3: Trends & Insights
# ============================================================
with tab3:
    st.subheader("Tourism Trends Dashboard")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Rating", title="Rating Distribution", nbins=5)
        st.plotly_chart(fig, use_container_width=True)

        top_types = df["Attr_AttractionType"].value_counts().head(10).reset_index()
        top_types.columns = ["AttractionType", "Visits"]
        fig3 = px.bar(top_types, x="Visits", y="AttractionType", orientation="h",
                      title="Top 10 Attraction Types by Popularity")
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        mode_counts = df["VisitModeName"].value_counts().reset_index()
        mode_counts.columns = ["VisitMode", "Count"]
        fig2 = px.pie(mode_counts, names="VisitMode", values="Count", title="Visit Mode Share")
        st.plotly_chart(fig2, use_container_width=True)

        cont_counts = df["User_Continent"].value_counts().reset_index()
        cont_counts.columns = ["Continent", "Visits"]
        fig4 = px.bar(cont_counts, x="Continent", y="Visits", title="Visits by User Continent")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.write("**Model performance snapshot:**")
    with open(os.path.join(OUT, "model_metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(OUT, "recommendation_metrics.json")) as f:
        rec_metrics = json.load(f)

    m1, m2, m3 = st.columns(3)
    m1.metric("Regression R²", f"{metrics['regression']['r2']:.3f}")
    m2.metric("Classification Accuracy", f"{metrics['classification_lightgbm']['accuracy']:.3f}")
    m3.metric("Recommender RMSE", f"{rec_metrics['rmse']:.3f}")

# ============================================================
# TAB 4: About
# ============================================================
with tab4:
    st.subheader("About this project")
    st.markdown("""
This application was built for the **Tourism Experience Analytics** project, covering:

- **Regression** — predicting the rating a user is likely to give an attraction
- **Classification** — predicting a user's likely visit mode (Business, Family, Couples, Friends, Solo)
- **Recommendation** — a collaborative-filtering system (SVD-based) suggesting attractions,
  with a content-based fallback for new/cold-start users

**Data sources:** Transaction, User, City, Country, Region, Continent, Visit Mode, and
Attraction (Type + Item) tables, joined into a consolidated master dataset of ~53,000 transactions.
    """)
