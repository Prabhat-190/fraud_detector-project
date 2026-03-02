import os
import time
import requests
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# ================= CONFIG =================
MODEL_FILE = "fraud_model.pkl"
DATA_FILE = "credit_card_fraud_dataset_modified - credit_card_fraud_dataset_modified.csv"

# ================= TRAIN MODEL =================
def build_and_train_pipeline():
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset not found: {DATA_FILE}")
        return None

    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=['TransactionID'], errors='ignore')

    cat_cols = ['Location', 'MerchantCategory']
    num_cols = ['Amount', 'Time', 'CardHolderAge']

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    X = df.drop('IsFraud', axis=1)
    y = df['IsFraud']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    try:
        joblib.dump(pipeline, MODEL_FILE)
    except:
        pass

    return pipeline


@st.cache_resource
def get_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except:
            pass
    return build_and_train_pipeline()

# ================= LIVE CRYPTO (CoinGecko) =================
def fetch_live_crypto_data():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url, timeout=5)
        data = response.json()
        price = data["bitcoin"]["usd"]

        return {
            'Amount': float(price),
            'Time': int(time.time() % 86400),
            'CardHolderAge': 30,
            'Location': 'Online',
            'MerchantCategory': 'Crypto'
        }
    except:
        return None

# ================= STREAMLIT UI =================
def main():
    st.set_page_config(page_title="FRAUD_SHIELD_V3", page_icon="🛡️", layout="wide")
    st.title("🛡️ FraudShield / Audit Pro")
    st.caption("AI-Based Fraud Detection with Live Crypto Demo")

    model = get_model()

    if model is None:
        st.error("Model could not be loaded. Check dataset file.")
        return

    tab1, tab2 = st.tabs(["📊 ANALYST_AUDITOR", "₿ LIVE_CRYPTO_STREAM"])

    # ---------- TAB 1 ----------
    with tab1:
        with st.form("audit_form"):
            amt = st.number_input("TXN_AMOUNT (USD)", value=250.00)
            age_val = st.slider("HOLDER_AGE", 18, 95, 30)
            loc = st.selectbox("LOCATION", ["California", "New York", "London", "Online", "Tokyo"])
            cat = st.selectbox("CATEGORY", ["Retail", "Electronics", "Crypto", "Entertainment"])
            submit = st.form_submit_button("RUN AUDIT")

        if submit:
            input_df = pd.DataFrame([[amt, int(time.time()%86400), age_val, loc, cat]],
                                     columns=['Amount', 'Time', 'CardHolderAge', 'Location', 'MerchantCategory'])

            prob = model.predict_proba(input_df)[0][1]

            if prob > 0.5:
                st.error(f"🚨 HIGH RISK DETECTED ({prob*100:.2f}%)")
            else:
                st.success(f"✅ SAFE TRANSACTION ({prob*100:.2f}%)")

            st.progress(prob)

    # ---------- TAB 2 ----------
    with tab2:
        st.subheader("📡 Live Bitcoin Price (CoinGecko)")
        if st.button("Fetch Latest BTC Price"):
            raw_txn = fetch_live_crypto_data()

            if raw_txn is None:
                st.error("Unable to fetch live data.")
            else:
                input_df = pd.DataFrame([raw_txn])
                prob = model.predict_proba(input_df)[0][1]

                col1, col2, col3 = st.columns(3)
                col1.metric("BTC PRICE", f"${raw_txn['Amount']:,}")
                col2.metric("RISK INDEX", f"{prob*100:.2f}%")
                col3.metric("STATUS", "ANOMALY" if prob > 0.5 else "NOMINAL")

                st.progress(prob)


if __name__ == "__main__":
    main()
