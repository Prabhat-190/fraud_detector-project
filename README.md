This project demonstrates how a pre-trained machine learning model can be deployed as a real-time fraud detection web service using Flask. The API takes transaction details, predicts if a transaction is fraudulent, and returns both the classification and probability score.

🚀 Features

Loads a pre-trained ML pipeline (fraud_detection_model.joblib)

REST API built with Flask

Endpoint /predict accepts JSON input

Returns fraud prediction (Fraud / Not Fraud) with probability score

Integration-ready for banking systems, payment gateways, or e-commerce platforms

📂 Project Structure
├── app.py                    # Flask API script  
├── fraud_detection_model.joblib  # Pre-trained ML pipeline  
├── requirements.txt          # Dependencies  
├── README.md                 # Project documentation  

⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/fraud-detection-api.git
cd fraud-detection-api


Install dependencies:

pip install -r requirements.txt


Run the Flask app:

python app.py

📡 Usage

Send a POST request to the API:

curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "Amount": 250.50,
           "Location": "New York",
           "MerchantCategory": "Electronics",
           "Time": 123456,
           "CardHolderAge": 32
         }'

✅ Example Response
{
  "prediction": "Fraud",
  "probability": 0.87
}

📊 Model

The fraud detection pipeline was trained on historical credit card transaction data using Random Forest Classifier. Preprocessing steps (scaling, encoding) and training logic are encapsulated inside the saved pipeline.

🔮 Future Improvements

Deploy on cloud (AWS/GCP/Heroku)

Add authentication to the API

Support batch transaction predictions
