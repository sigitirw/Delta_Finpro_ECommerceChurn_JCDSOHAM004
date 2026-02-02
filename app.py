import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from pathlib import Path
import xgboost
import imblearn

st.write("APP STARTED")
# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="E-Commerce Customer Churn Prediction",
    page_icon="🛒",
    layout="wide"
)

# =====================================================
# CUSTOM CSS (Bright but not white)
# =====================================================
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(135deg, #e0f2ff, #f3e8ff);
        }
        .stApp {
            background: linear-gradient(135deg, #e0f2ff, #f3e8ff);
        }
        .block-container {
            padding-top: 2rem;
        }
        .metric-card {
            background-color: #ffffffaa;
            padding: 1.2rem;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# CONSTANTS
# =====================================================
DATA_PATH = Path("ecommerce_churn.xlsx")
MODEL_PATH = Path("ECommerceChurn.pkl")

FEATURE_COLUMNS = [
    'Tenure', 'PreferredLoginDevice', 'CityTier',
    'WarehouseToHome', 'PreferredPaymentMode', 'Gender',
    'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'PreferedOrderCat', 'SatisfactionScore',
    'MaritalStatus', 'NumberOfAddress', 'Complain',
    'OrderAmountHikeFromlastYear', 'CouponUsed',
    'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
]

TARGET_COL = "Churn"

# =====================================================
# LOADERS (CACHED)
# =====================================================
@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# =====================================================
# DATA PROFILING
# =====================================================
cat_cols = df[FEATURE_COLUMNS].select_dtypes(include="object").columns.tolist()
num_cols = df[FEATURE_COLUMNS].select_dtypes(exclude="object").columns.tolist()

unique_values = {
    col: sorted(df[col].dropna().unique().tolist())
    for col in cat_cols
}

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("🛒 E-Commerce Churn App")
menu = st.sidebar.radio(
    "Navigation",
    ["📊 Exploratory Data Analysis", "🔮 Predict Customer Churn"]
)

# =====================================================
# EDA PAGE
# =====================================================
if menu == "📊 Exploratory Data Analysis":

    st.title("📊 Exploratory Data Analysis (EDA)")
    st.write("Understand customer behavior and churn patterns")

    # Dataset Overview
    st.subheader("📌 Dataset Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Churn Rate (%)", round(df[TARGET_COL].mean() * 100, 2))

    # Data Types
    st.subheader("📄 Data Types")
    st.dataframe(df.dtypes.astype(str), use_container_width=True)

    # Missing Values
    st.subheader("⚠️ Missing Values")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing Count"]
    st.dataframe(missing_df, use_container_width=True)

    # Churn Distribution
    st.subheader("📉 Churn Distribution")
    fig = px.histogram(
        df,
        x=TARGET_COL,
        color=TARGET_COL,
        title="Churn vs Non-Churn Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Numerical vs Churn
    st.subheader("📈 Numerical Features vs Churn")
    num_feature = st.selectbox("Select Numerical Feature", num_cols)

    fig = px.box(
        df,
        x=TARGET_COL,
        y=num_feature,
        color=TARGET_COL,
        title=f"{num_feature} vs Churn"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Categorical vs Churn
    st.subheader("📊 Categorical Features vs Churn")
    cat_feature = st.selectbox("Select Categorical Feature", cat_cols)

    fig = px.histogram(
        df,
        x=cat_feature,
        color=TARGET_COL,
        barmode="group",
        title=f"{cat_feature} vs Churn"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PREDICTION PAGE
# =====================================================
else:

    st.title("🔮 E-Commerce Customer Churn Prediction")
    st.write("Predict churn risk for individual customers or in bulk")

    tab1, tab2 = st.tabs(["👤 Single Prediction", "📂 Bulk Prediction"])

    # -------------------------------------------------
    # SINGLE CUSTOMER PREDICTION
    # -------------------------------------------------
    with tab1:

        st.subheader("👤 Single Customer Prediction")

        with st.form("single_prediction_form"):
            input_data = {}

            for col in FEATURE_COLUMNS:
                if col in cat_cols:
                    input_data[col] = st.selectbox(
                        col,
                        unique_values[col]
                    )
                else:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())

                    input_data[col] = st.slider(
                        col,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val
                    )

            submitted = st.form_submit_button("Predict Churn")

        if submitted:
            input_df = pd.DataFrame([input_data])

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            if pred == 1:
                st.error(f"🚨 **Churn Predicted**")
                st.metric("Churn Probability", f"{prob*100:.2f}%")
            else:
                st.success(f"✅ **Not Churn**")
                st.metric("Churn Probability", f"{prob*100:.2f}%")

    # -------------------------------------------------
    # BULK PREDICTION
    # -------------------------------------------------
    with tab2:

        st.subheader("📂 Bulk Prediction (File Upload)")

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx"]
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    bulk_df = pd.read_csv(uploaded_file)
                else:
                    bulk_df = pd.read_excel(uploaded_file)

                missing_cols = set(FEATURE_COLUMNS) - set(bulk_df.columns)
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    preds = model.predict(bulk_df[FEATURE_COLUMNS])
                    probs = model.predict_proba(bulk_df[FEATURE_COLUMNS])[:, 1]

                    result_df = bulk_df.copy()
                    result_df["Churn_Prediction"] = preds
                    result_df["Churn_Probability"] = probs

                    st.success("Prediction completed successfully")
                    st.dataframe(result_df.head(), use_container_width=True)

                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Prediction Result",
                        csv,
                        "churn_prediction_result.csv",
                        "text/csv"
                    )

            except Exception as e:
                st.error(f"Error processing file: {e}")
