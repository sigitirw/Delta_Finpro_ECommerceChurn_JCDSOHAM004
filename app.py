import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Online E-Commerce Churn Prediction",
    page_icon="./assets/Logo_Delta.png",
    layout="wide"
)
st.markdown(
    """
    <style>
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)
# ==========================================================
# DATA LOADING
# ==========================================================
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).parent
    DATA_PATH = BASE_DIR / "EcommerceChurnCleaned.csv"

    if not DATA_PATH.exists():
        st.error(f"Data file tidak ditemukan: {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    # Cast numeric columns to int except CashbackAmount
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = [c for c in numeric_cols if c != "CashbackAmount"]

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
    MODEL_PATH = BASE_DIR / "ECommerceChurn.pkl"

    if not MODEL_PATH.exists():
        st.error(f"Model file tidak ditemukan: {MODEL_PATH}")
        st.write("Files:", list(BASE_DIR.iterdir()))
        st.stop()

    return joblib.load(MODEL_PATH)


# ==========================================================
# MAIN APP
# ==========================================================
import base64

def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: top center;
            background-repeat: no-repeat;
        }}

        section.main > div {{
            background-color: rgba(255, 255, 255, 0.90);
            padding: 2rem;
            border-radius: 14px;
            margin-top: 1.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def main():
    BASE_DIR = Path(__file__).parent
    set_background(BASE_DIR / "assets" / "Streamlit.png")

    
    # ======================================================
    # HEADER WITH LOGO
    # ======================================================
    
    st.markdown(
        """
        Model ini digunakan untuk **mengidentifikasi pelanggan berisiko churn**
        menggunakan **machine learning (XGBoost – F2 Optimized)** untuk mendukung
        strategi retensi pelanggan.
        """
    )

    st.divider()

    # ======================================================
    # LOAD DATA & MODEL
    # ======================================================
    df = load_data()
    model = load_model()

    # ======================================================
    # SIDEBAR
    # ======================================================
    with st.sidebar:
        st.sidebar.image("./assets/Logo_Delta.png", use_column_width=True)
        st.markdown("## 🧠 Model Information")
        st.markdown("""
        - Algorithm: **XGBoost**
        - Metric: **F2 Score**
        - Fokus: **Minimalkan False Negative**
        """)
        st.markdown("---")

        mode = st.radio(
            "Mode Prediksi",
            ["Single Prediction", "Bulk Prediction"]
        )

    # ======================================================
    # SINGLE PREDICTION
    # ======================================================
    if mode == "Single Prediction":
        st.subheader("🧾 Single Customer Prediction")

        input_data = {}
        col1, col2 = st.columns(2)

        for i, col in enumerate(df.drop(columns=["Churn"]).columns):

            target_col = col1 if i % 2 == 0 else col2

            with target_col:
                if col == "Complain":
                    input_data[col] = st.selectbox(
                        "Complain",
                        ["No", "Yes"]
                    )

                elif col == "SatisfactionScore":
                    input_data[col] = st.slider(
                        "Satisfaction Score",
                        min_value=1,
                        max_value=5,
                        value=3
                    )

                elif df[col].dtype == "int64":
                    input_data[col] = st.number_input(
                        col,
                        min_value=0,
                        step=1
                    )

                elif df[col].dtype == "float64":
                    input_data[col] = st.number_input(
                        col,
                        min_value=0.0,
                        format="%.2f"
                    )

                else:
                    input_data[col] = st.selectbox(
                        col,
                        df[col].unique()
                    )

        input_df = pd.DataFrame([input_data])

        # Encode Complain Yes/No → 1/0
        if "Complain" in input_df.columns:
            input_df["Complain"] = input_df["Complain"].map({"No": 0, "Yes": 1})

        st.divider()

        if st.button("🔍 Predict Churn", use_container_width=True):
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            colA, colB = st.columns(2)

            with colA:
                st.metric(
                    "Churn Probability",
                    f"{prob:.2%}"
                )

            with colB:
                if pred == 1:
                    st.error("⚠️ Customer BERISIKO CHURN")
                else:
                    st.success("✅ Customer TIDAK churn")

    # ======================================================
    # BULK PREDICTION
    # ======================================================
    else:
        st.subheader("📤 Bulk Prediction")

        uploaded_file = st.file_uploader(
            "Upload CSV file untuk bulk prediction",
            type=["csv"]
        )

        if uploaded_file is not None:
            bulk_df = pd.read_csv(uploaded_file)
            st.write("Preview data:")
            st.dataframe(bulk_df.head())

            if "Complain" in bulk_df.columns:
                bulk_df["Complain"] = bulk_df["Complain"].map(
                    {"No": 0, "Yes": 1}
                )

            if st.button("🚀 Run Bulk Prediction", use_container_width=True):
                bulk_pred = model.predict(bulk_df)
                bulk_prob = model.predict_proba(bulk_df)[:, 1]

                result = bulk_df.copy()
                result["churn_prediction"] = bulk_pred
                result["churn_probability"] = bulk_prob

                st.success("Bulk prediction selesai")
                st.dataframe(result.head())

                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Result CSV",
                    csv,
                    "bulk_churn_prediction.csv",
                    "text/csv",
                    use_container_width=True
                )

    # ======================================================
    # FOOTER
    # ======================================================
    st.divider()
    st.caption(
        "⚠️ Model digunakan sebagai early warning system. "
        "Efektivitas tindakan retensi perlu divalidasi melalui A/B testing."
    )


# ==========================================================
# RUN APP
# ==========================================================
if __name__ == "__main__":
    main()
