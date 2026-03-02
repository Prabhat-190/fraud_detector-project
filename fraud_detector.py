
import os
import time
import pandas as pd
import streamlit as st


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib


try:
    import ccxt
    CCXT_AVAILABLE = True
except Exception:
    CCXT_AVAILABLE = False

# ==== CONFIG =====
# Consider renaming the CSV in your repo to a simpler name (recommended)
MODEL_FILE = "fraud_model.pkl"
DATA_FILE = "credit_card_fraud_dataset_modified - credit_card_fraud_dataset_modified.csv"
# DATA_FILE = "credit_card_fraud_dataset.csv"   # if you rename file, update this

# ============ TRAINING ============
def build_and_train_pipeline():
    """Train pipeline from CSV and return trained pipeline (do not block UI)."""
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset not found in repo: {DATA_FILE}")
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
    # Optionally save for local reuse
    try:
        joblib.dump(pipeline, MODEL_FILE)
    except Exception:
        # saving may fail on some read-only environments — that's fine
        pass

    return pipeline

# Use Streamlit cached resource so training runs only once per session/deploy
@st.cache_resource
def get_model():
    # Prefer loading pre-saved model if present
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            # If load fails, fallback to training
            pass
    # If no model file or load failed, train
    pipeline = build_and_train_pipeline()
    return pipeline

# ============ LIVE CRYPTO FETCH (safe) ============
def fetch_live_crypto_data(symbol="BTC/USDT"):
    if not CCXT_AVAILABLE:
        return None
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker(symbol)
        return {
            'Amount': float(ticker.get('last', 0.0)),
            'Time': int(time.time() % 86400),
            'CardHolderAge': 30,
            'Location': 'Online',
            'MerchantCategory': 'Crypto'
        }
    except Exception:
        return None

# ============ APP UI ============
def main():
    st.set_page_config(page_title="FRAUD_SHIELD_V3", page_icon="🛡️", layout="wide")
    st.title("🛡️ FraudShield / Audit Pro (Safe Deploy)")
    st.caption("ML pipeline cached; live exchange optional and non-blocking")

    model = get_model()
    if model is None:
        st.error("Model is not available (dataset missing or training failed). Check repo files.")
        st.stop()

    tab1, tab2 = st.tabs(["📊 ANALYST_AUDITOR", "₿ LIVE_CRYPTO_STREAM"])

    # --- Manual Audit tab (safe) ---
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
            try:
                prob = model.predict_proba(input_df)[0][1]
            except Exception as e:
                st.error("Model prediction failed. Check feature consistency. " + str(e))
                prob = None

            if prob is not None:
                if prob > 0.5:
                    st.error(f"🚨 ALERT: HIGH RISK DETECTED ({(prob*100):.1f}%)")
                else:
                    st.success(f"✅ SECURE: Risk Score ({(prob*100):.1f}%)")
                st.progress(prob)

    # --- Live Crypto tab (safe approach: manual fetch or optional lightweight auto-refresh) ---
    with tab2:
        st.subheader("📡 EXCHANGE_LIVE_FEED (BINANCE) — safe mode")
        if not CCXT_AVAILABLE:
            st.warning("Live exchange integration disabled (ccxt not installed). To enable, add 'ccxt' to requirements.txt and reboot.")
        # Manual single-fetch button (safe)
        if st.button("Fetch latest BTC price & score"):
            raw_txn = fetch_live_crypto_data()
            if raw_txn is None:
                st.error("Unable to fetch live data from exchange.")
            else:
                input_df = pd.DataFrame([raw_txn])
                try:
                    prob = model.predict_proba(input_df)[0][1]
                except Exception as e:
                    st.error("Model prediction failed for live input: " + str(e))
                    prob = None

                m1, m2, m3 = st.columns(3)
                m1.metric("LIVE PRICE (BTC)", f"${raw_txn['Amount']:,}")
                m2.metric("RISK INDEX", f"{(prob*100):.2f}%" if prob is not None else "N/A")
                m3.metric("STATUS", "ANOMALY" if (prob is not None and prob > 0.5) else "NOMINAL")

                if prob is not None and prob > 0.5:
                    st.markdown("<div style='background:#4d0000; padding:20px; border-radius:10px; border:1px solid red;'>🚨 CRITICAL: Anomalous Volatility Signature Detected</div>", unsafe_allow_html=True)
                elif prob is not None:
                    st.markdown("<div style='background:#00261a; padding:20px; border-radius:10px; border:1px solid #00ffcc;'>✅ STABLE: Network Traffic Verified Nominal</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.info("Tip: Click 'Fetch latest' to get a fresh sample. If you want auto-refresh, add 'streamlit-autorefresh' to requirements and I can show you that version.")

if __name__ == "__main__":
    main()
