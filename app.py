import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(
    page_title="E-Commerce Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# ==========================================================
# DATA LOADING
# ==========================================================
@st.cache_data

def load_data():
    BASE_DIR = Path(__file__).parent
    DATA_PATH = BASE_DIR / "EcommerceChurnCleaned.csv"

    df = pd.read_csv(DATA_PATH)

    # Convert numeric columns to int except CashbackAmount
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = [col for col in numeric_cols if col != "CashbackAmount"]

    for col in numeric_cols:
        df[col] = df[col].fillna(0).astype(int)

    if "CashbackAmount" in df.columns:
        df["CashbackAmount"] = df["CashbackAmount"].astype(float)

    return df


# ==========================================================
# MODEL LOADING
# ==========================================================
@st.cache_resource

def load_model():
    BASE_DIR = Path(__file__).parent
    MODEL_PATH = BASE_DIR / "best_model_xgb.pkl"
    return joblib.load(MODEL_PATH)


# ==========================================================
# MAIN APP
# ==========================================================

# ==========================================================

def main():
    st.title("📉 Customer Churn Prediction – E-Commerce")
    st.markdown("""
    Aplikasi ini digunakan untuk **memprediksi kemungkinan customer churn**
    menggunakan model **XGBoost** yang dioptimalkan dengan **F2 Score**.

    Fokus utama: **mendeteksi churn sedini mungkin** untuk mendukung strategi retensi.
    """)


    df = load_data()
    model = load_model()

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.divider()

    st.subheader("📤 Bulk Prediction (Upload CSV)")
    uploaded_file = st.file_uploader(
        "Upload file CSV untuk bulk prediction",
        type=["csv"]
    )

    if uploaded_file is not None:
        bulk_df = pd.read_csv(uploaded_file)

        st.write("Preview data:")
        st.dataframe(bulk_df.head())

        if st.button("🚀 Run Bulk Prediction"):
            bulk_pred = model.predict(bulk_df)
            bulk_prob = model.predict_proba(bulk_df)[:, 1]

            bulk_result = bulk_df.copy()
            bulk_result["churn_prediction"] = bulk_pred
            bulk_result["churn_probability"] = bulk_prob

            st.success("Bulk prediction selesai")
            st.dataframe(bulk_result.head())

            csv = bulk_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Result CSV",
                data=csv,
                file_name="bulk_churn_prediction.csv",
                mime="text/csv"
            )

    st.divider()

    st.subheader("🧾 Input Customer Data")

    input_data = {}

    for col in df.drop(columns=["Churn"]).columns:
        if df[col].dtype == "int64":
            input_data[col] = st.number_input(col, min_value=0, step=1)
        elif df[col].dtype == "float64":
            input_data[col] = st.number_input(col, min_value=0.0, format="%.2f")
        else:
            input_data[col] = st.selectbox(col, df[col].unique())

    input_df = pd.DataFrame([input_data])

    st.divider()

    if st.button("🔍 Predict Churn"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Customer BERISIKO CHURN (Probabilitas: {probability:.2%})")
        else:
            st.success(f"✅ Customer TIDAK churn (Probabilitas churn: {probability:.2%})")

        st.markdown("""
        ### 📉 Customer Churn Prediction – E-Commerce

        Model ini:
        - Dioptimalkan menggunakan **F2 Score**
        - Fokus meminimalkan **False Negative**
        - Digunakan sebagai **early warning system**
        """)


# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    main()
