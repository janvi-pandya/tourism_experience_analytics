# Tourism Experience Analytics — Project Package

## Structure
- `src/` — pipeline scripts, run in order:
  1. `01_clean_merge.py` — cleans & merges all 10 raw tables into a master dataset
  2. `02_eda.py` — generates 8 EDA charts (saved to `figures/`)
  3. `03_train_models.py` — trains regression (rating) & classification (visit mode) models
  4. `04_recommendation.py` — builds the SVD collaborative-filtering recommender + content-based fallback
  5. `05_generate_report.js` — generates the Word documentation report (run with `node`)
- `app/app.py` — the Streamlit application (4 tabs: Predict My Trip, Recommendations, Trends & Insights, About)
- `data_outputs/` — cleaned master dataset, dimension tables, and metrics JSONs
- `models/` — trained model artifacts (.pkl)
- `figures/` — EDA chart images
- `Tourism_Experience_Analytics_Report.docx` — full project documentation report

## Running the app
```
pip install streamlit pandas numpy scikit-learn lightgbm plotly joblib
streamlit run app/app.py
```
(Run from a directory where `outputs/` sits alongside `app/`, or adjust paths — the app expects
`data_outputs/` and `models/` to be under a sibling `outputs/` folder, matching this package's layout
if you rename `data_outputs` -> `outputs` and merge `models/` inside it.)

## Key results
- Regression R²: 0.746
- Classification accuracy (LightGBM): 0.508 (vs. 0.496 Random Forest baseline)
- Recommender RMSE: 1.058
