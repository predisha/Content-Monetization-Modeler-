# ---------------------------------------------------------------
# 🎬 YouTube Ad Revenue Predictor - Content Monetization Modeler
# ---------------------------------------------------------------
# End-to-End Project: Data Cleaning + Model + Streamlit Dashboard
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# ---------------------------------------------------------------
# STEP 1️⃣: LOAD AND CLEAN DATA
# ---------------------------------------------------------------

DATA_PATH = r"C:\Users\predi\Downloads\youtube_ad_revenue_cleaned.xlsx"  # 👉 update path if needed

st.set_page_config(page_title="Content Monetization Modeler", layout="wide")

st.title("🎥 Content Monetization Modeler")
st.markdown("Predict YouTube ad revenue and analyze performance metrics interactively!")

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values safely
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    return df

df = load_data()
st.subheader("📊 Dataset Overview")
st.dataframe(df.head())

# ---------------------------------------------------------------
# STEP 2️⃣: FEATURE ENGINEERING & PREPROCESSING
# ---------------------------------------------------------------

# Engagement Rate Feature
if all(x in df.columns for x in ['likes', 'comments', 'views']):
    df['engagement_rate'] = (df['likes'] + df['comments']) / (df['views'] + 1)

# Extract month from date
if 'date' in df.columns:
    df['month'] = df['date'].dt.month

# Encode categorical columns
cat_cols = ['category', 'device', 'country']
le = LabelEncoder()
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# Feature Scaling
scaler = StandardScaler()
num_cols = ['views', 'likes', 'comments', 'watch_time_minutes',
            'video_length_minutes', 'subscribers', 'engagement_rate']
for col in num_cols:
    if col in df.columns:
        df[col] = scaler.fit_transform(df[[col]])

# ---------------------------------------------------------------
# STEP 3️⃣: MODEL BUILDING
# ---------------------------------------------------------------
X = df.drop(columns=['ad_revenue_usd', 'video_id', 'date'], errors='ignore')
y = df['ad_revenue_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Support Vector Regressor": SVR()
}

results = {}
best_model_name = None
best_score = -1

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    results[name] = [r2, rmse, mae]
    if r2 > best_score:
        best_score = r2
        best_model_name = name
        best_model = model

# ---------------------------------------------------------------
# STEP 4️⃣: MODEL PERFORMANCE DISPLAY
# ---------------------------------------------------------------
st.subheader("📈 Model Performance Comparison")
results_df = pd.DataFrame(results, index=['R2', 'RMSE', 'MAE']).T
st.dataframe(results_df.style.highlight_max(axis=0))

st.success(f"🏆 Best Model Selected: {best_model_name}")

# Save the best model
joblib.dump(best_model, "best_ad_revenue_model.pkl")

# ---------------------------------------------------------------
# STEP 5️⃣: FEATURE IMPORTANCE (if applicable)
# ---------------------------------------------------------------
if hasattr(best_model, "feature_importances_"):
    st.subheader("🔍 Top 10 Important Features")
    importance = pd.Series(best_model.feature_importances_, index=X.columns)
    st.bar_chart(importance.nlargest(10))

# ---------------------------------------------------------------
# STEP 6️⃣: STREAMLIT PREDICTION SECTION
# ---------------------------------------------------------------
st.header("💰 Predict Ad Revenue from Video Metrics")

views = st.number_input("Views", min_value=0)
likes = st.number_input("Likes", min_value=0)
comments = st.number_input("Comments", min_value=0)
watch_time = st.number_input("Watch Time (minutes)", min_value=0)
video_length = st.number_input("Video Length (minutes)", min_value=0)
subscribers = st.number_input("Subscribers", min_value=0)
engagement_rate = (likes + comments) / (views + 1)

input_data = pd.DataFrame([[views, likes, comments, watch_time, video_length, subscribers, engagement_rate]],
                          columns=['views', 'likes', 'comments', 'watch_time_minutes',
                                   'video_length_minutes', 'subscribers', 'engagement_rate'])
input_data[num_cols] = scaler.transform(input_data[num_cols])

if st.button("🚀 Predict Revenue"):
    prediction = best_model.predict(input_data)[0]
    st.success(f"Estimated Ad Revenue: **${prediction:.2f} USD**")

# ---------------------------------------------------------------
# STEP 7️⃣: INSIGHTS SECTION
# ---------------------------------------------------------------
st.header("📊 Insights & Correlation")

corr = df.corr()['ad_revenue_usd'].sort_values(ascending=False)
st.write(corr.head(10))

st.caption("✅ Project Completed Successfully — Developed by Predisha.")
