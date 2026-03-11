# 🛡 Fraud Shield Enterprise

Real-Time Transaction Risk Monitoring Dashboard  

🔗 **Live Demo:**  
https://fraudsheild-ai-4ruti3ugr9qcuafvmdxekm.streamlit.app/<img width="1468" height="780" alt="Screenshot 2026-03-11 at 6 13 55 PM" src="https://github.com/user-attachments/assets/f0fef9d2-b8ea-428b-b833-9a1ca468cebe" />
<img width="1461" height="774" alt="Screenshot 2026-03-11 at 6 14 14 PM" src="https://github.com/user-attachments/assets/595dffa0-03fa-404e-96f5-0c98c045ed21" />
<img width="1463" height="784" alt="Screenshot 2026-03-11 at 6 14 28 PM" src="https://github.com/user-attachments/assets/7fbbec33-d13d-4645-8591-ec5c38854af4" />


---

## About The Project

Fraud Shield Enterprise is a real-time fraud detection dashboard built using Machine Learning and Streamlit.

The goal of this project is to simulate how modern financial security systems monitor transactions and detect suspicious activity instantly. Instead of relying only on static historical data, the system generates dynamic transaction patterns and evaluates risk in real time.

The application can analyze manual transactions or monitor a live data stream to estimate the probability of fraud.

---

## Features

### Manual Transaction Analysis
Users can manually enter transaction details such as:

- Amount
- Cardholder Age
- Location
- Merchant Category

The system immediately calculates a **fraud risk score (0–100%)** using a trained machine learning model.

---

### Live Network Monitoring
The dashboard can also simulate a real-time financial monitoring system.

It connects to a cryptocurrency exchange API and continuously analyzes transaction patterns. If the live API becomes unavailable, the system automatically switches to a fallback simulation to keep the monitoring stream active.

Key capabilities:

- Live BTC price tracking  
- Continuous fraud risk scoring  
- Real-time risk trend chart  
- Automatic fallback simulation  

---

## Machine Learning Model

The fraud detection engine uses a **Random Forest Classifier** trained on dynamically generated transaction data.

Features used for prediction:

- Transaction Amount
- Transaction Time
- Cardholder Age
- Geographic Location
- Merchant Category

Additional heuristics are applied to generate smooth and realistic fraud probability scores.

---

## Tech Stack

**Frontend / Dashboard**
- Streamlit

**Machine Learning**
- Scikit-learn (RandomForestClassifier)

**Data Processing**
- Pandas  
- NumPy  

**Live Data Integration**
- CCXT (Kraken cryptocurrency exchange API)

**Visualization**
- Streamlit Charts

---

## Development Roadmap

The project is designed to evolve into a more advanced fraud monitoring system.

![Fraud Shield Roadmap] https://drive.google.com/file/d/1ETwgxgQCiWfAJuXuNvl-lnXRiHSDSAkR/view?usp=sharing

Future improvements include:

- Unsupervised anomaly detection (Isolation Forest)
- Multi-asset monitoring (BTC, ETH, SOL)
- Geo-IP mapping of transaction sources
- Automated fraud reporting
- FastAPI microservice for scalable ML inference

---

