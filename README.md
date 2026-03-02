🛡️ Fraud Shield Enterprise

A real-time fraud risk intelligence dashboard built using Machine Learning and live crypto market data.

🌐 Live App:
https://fraudsheild-ai-4ruti3ugr9qcuafvmdxekm.streamlit.app/#anomalous-volatility-signature<img width="1468" height="818" alt="Screenshot 2026-03-02 at 11 10 17 PM" src="https://github.com/user-attachments/assets/7cddf3df-7a04-4e13-a537-5b9065beee73" />
<img width="1467" height="832" alt="Screenshot 2026-03-02 at 11 10 04 PM" src="https://github.com/user-attachments/assets/18f11c47-c75b-4bf4-9f10-5599b25ae057" />


📌 Overview

Fraud Shield Enterprise simulates how modern fintech systems evaluate transaction risk.
It combines a trained Random Forest model with a dynamic risk scoring engine and live BTC/USD market data.

Instead of static rule-based outputs, the system generates a continuously changing risk score (0–100%) based on:

Transaction amount

Time of transaction

Cardholder age

Location

Merchant category

Live crypto trade activity

The result is a real-time risk monitoring interface that behaves like a lightweight fraud detection engine.

⚙️ Features

Machine Learning fraud probability model

Dynamic real-time risk adjustment layer

Live BTC/USD trade integration via Kraken (CCXT)

Auto-updating risk graph

Manual transaction simulation

Enterprise-style dark UI

🛠 Tech Stack

Python • Streamlit • Scikit-learn • Pandas • NumPy • CCXT • Joblib

▶ Run Locally
pip install -r requirements.txt
streamlit run fraud_detector.py
