import os
import time
import datetime
import random
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

try:
    import ccxt
    CCXT_AVAILABLE = True
except Exception:
    CCXT_AVAILABLE = False

MODEL_FILE = "fraud_model_final_v12.pkl"

def generate_dynamic_dataset():
    n = 20000
    amounts = np.random.uniform(5.0, 80000.0, size=n)
    times = np.random.randint(0, 86400, size=n)
    ages = np.random.randint(18, 85, size=n)
    locations = np.random.choice(["California", "New York", "London", "Online", "Tokyo"], size=n)
    cats = np.random.choice(["Retail", "Electronics", "Crypto", "Entertainment"], size=n)

    z = (
        -2.5
        + 0.00005 * amounts
        + 0.00000001 * (amounts ** 2)
        + 0.00002 * times
        - 0.015 * ages
    )

    z += np.where(cats == "Crypto", 1.8, 0)
    z += np.where(cats == "Electronics", 0.7, 0)
    z -= np.where(cats == "Retail", 1.2, 0)
    z += np.where(locations == "Online", 1.5, 0)

    noise = np.random.normal(0, 1.2, size=n)
    z += noise

    probs = 1 / (1 + np.exp(-z))
    is_fraud = np.random.binomial(1, probs)

    return pd.DataFrame({
        "Amount": amounts,
        "Time": times,
        "CardHolderAge": ages,
        "Location": locations,
        "MerchantCategory": cats,
        "IsFraud": is_fraud
    })

def build_model():
    df = generate_dynamic_dataset()

    cat_cols = ["Location", "MerchantCategory"]
    num_cols = ["Amount", "Time", "CardHolderAge"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X = df.drop("IsFraud", axis=1)
    y = df["IsFraud"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=500, min_samples_split=5))
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
    return build_model()

def calculate_realtime_risk(input_df, model):
    base_prob = model.predict_proba(input_df)[0][1]

    amount = input_df["Amount"].values[0]
    age = input_df["CardHolderAge"].values[0]
    time_val = input_df["Time"].values[0]
    location = input_df["Location"].values[0]
    category = input_df["MerchantCategory"].values[0]

    amount_score = min(1.0, np.log1p(amount) / 12)
    hour = time_val // 3600
    time_score = 0.7 if hour < 5 or hour > 23 else 0.2
    age_score = max(0, (30 - age) / 30)

    cat_score = 0.8 if category == "Crypto" else 0.4 if category == "Electronics" else 0.1
    loc_score = 0.6 if location == "Online" else 0.2

    noise = np.random.uniform(-0.05, 0.05)

    final = (
        0.4 * base_prob +
        0.2 * amount_score +
        0.1 * time_score +
        0.1 * age_score +
        0.1 * cat_score +
        0.1 * loc_score
    ) + noise

    return np.clip(final, 0, 1)

def fetch_live_trade(symbol="BTC/USD"):
    if not CCXT_AVAILABLE:
        return None
    try:
        exchange = ccxt.kraken()
        trades = exchange.fetch_trades(symbol, limit=1)
        if not trades:
            return None

        t = trades[0]
        usd_value = t["price"] * t["amount"]

        return {
            "Amount": float(usd_value),
            "Time": int(time.time() % 86400),
            "CardHolderAge": random.randint(18, 75),
            "Location": random.choice(["California", "New York", "London", "Online", "Tokyo"]),
            "MerchantCategory": "Crypto",
            "Side": t["side"].upper(),
            "CryptoAmount": t["amount"],
            "Price": t["price"]
        }
    except:
        return None

def main():
    st.set_page_config(page_title="FRAUD_SHIELD_ENTERPRISE", page_icon="🛡️", layout="wide")

    st.markdown("""
    <style>
    .stApp { background-color: #060a11; color: white; }
    .title {
        font-size: 42px;
        font-weight: 900;
        background: linear-gradient(90deg,#00ffcc,#00b3ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right,#00ffcc,#00b3ff);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">🛡️ FRAUD SHIELD ENTERPRISE</div>', unsafe_allow_html=True)
    st.caption("Dynamic Real-Time Risk Intelligence Engine")
    st.divider()

    model = get_model()

    tab1, tab2 = st.tabs(["Manual Scan", "Live Network"])

    with tab1:
        col1, col2 = st.columns([0.4, 0.6])

        with col1:
            with st.form("form"):
                amt = st.number_input("Amount", value=250.0)
                age = st.slider("Age", 18, 95, 30)
                loc = st.selectbox("Location", ["California","New York","London","Online","Tokyo"])
                cat = st.selectbox("Category", ["Retail","Electronics","Crypto","Entertainment"])
                submit = st.form_submit_button("Analyze")

        with col2:
            if submit:
                df_input = pd.DataFrame([[amt, int(time.time()%86400), age, loc, cat]],
                    columns=["Amount","Time","CardHolderAge","Location","MerchantCategory"])
                risk = calculate_realtime_risk(df_input, model)
                st.progress(risk)
                if risk > 0.6:
                    st.error(f"High Risk {risk*100:.2f}%")
                else:
                    st.success(f"Secure {risk*100:.2f}%")

    with tab2:
        active = st.toggle("Activate Live Monitoring")
        if active:
            while active:
                trade = fetch_live_trade()
                if trade:
                    df_input = pd.DataFrame([[trade["Amount"], trade["Time"], trade["CardHolderAge"], trade["Location"], trade["MerchantCategory"]]],
                        columns=["Amount","Time","CardHolderAge","Location","MerchantCategory"])
                    risk = calculate_realtime_risk(df_input, model)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("USD Value", f"${trade['Amount']:,.2f}")
                    col2.metric("BTC Price", f"${trade['Price']:,.2f}")
                    col3.metric("Risk", f"{risk*100:.2f}%")
                    st.progress(risk)
                time.sleep(1.5)

if __name__ == "__main__":
    main()
