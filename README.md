Real-Time Fraud Detection API
This project demonstrates how a pre-trained machine learning model can be deployed as a real-time fraud detection web service using Flask. The API takes transaction details, predicts if a transaction is fraudulent, and returns both the classification and probability score.

🚀 Features
Loads a pre-trained ML pipeline (fraud_detection_model.joblib).

REST API built with Flask to serve predictions.

Endpoint /predict accepts transaction data in JSON format.

Returns a clear fraud prediction ('Fraud' / 'Not Fraud') along with the model's confidence score.

Designed for easy integration with banking systems, payment gateways, or e-commerce platforms.

📂 Project Structure
.
├── app.py                         # The Flask API script that serves the model
├── fraud_detector.py              # The script used to train the model
├── fraud_detection_model.joblib   # The pre-trained ML pipeline file
├── requirements.txt               # A list of all necessary Python libraries
└── README.md                      # Project documentation (this file)

⚙️ Installation & Setup
1. Clone the repository:

git clone https://github.com/Prabhat-190/fraud_detector-project
cd fraud_detector-project

2. Install dependencies:
(Ensure you have Python 3 installed)

python3 -m pip install -r requirements.txt

3. Run the training script (Optional):
If you want to retrain the model yourself, run the training script. This will generate the fraud_detection_model.joblib file.

python3 fraud_detector.py

4. Run the Flask API:
This command starts the web server, which loads the model and waits for prediction requests.

python3 app.py

The server will be running at http://127.0.0.1:5000.

📡 API Usage
To get a prediction, send a POST request to the /predict endpoint with the transaction data in the request body.

Example using curl:

curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
      "Amount": 250.75,
      "Time": 55000,
      "Location": "New York",
      "MerchantCategory": "Electronics",
      "CardHolderAge": 35
    }'

Example Response:

{
  "fraud_probability": 0.62,
  "is_fraud": 1,
  "prediction": "Fraud"
}

📊 Model Performance
The model pipeline was trained on historical credit card transaction data. Based on the last training run, the Logistic Regression model provided a better balance for identifying potential fraud cases compared to the Random Forest model, which failed to identify any.
--------------------------------------
Logistic Regression (Baseline) Report:
--------------------------------------
              precision    recall  f1-score   support

           0       0.96      0.58      0.72        95
           1       0.07      0.60      0.12         5

    accuracy                           0.58       100
   macro avg       0.52      0.59      0.42       100
weighted avg       0.92      0.58      0.69       100

ROC-AUC Score: 0.5432

--------------------------------------
Random Forest (Chosen Model) Report:
--------------------------------------
              precision    recall  f1-score   support

           0       0.95      0.99      0.97        95
           1       0.00      0.00      0.00         5

    accuracy                           0.94       100
   macro avg       0.47      0.49      0.48       100
weighted avg       0.90      0.94      0.92       100

ROC-AUC Score: 0.3905

--- Comparison & Conclusion ---
Logistic Regression ROC-AUC: 0.5432
Random Forest ROC-AUC: 0.3905

📊 Model

The fraud detection pipeline was trained on historical credit card transaction data using Random Forest Classifier. Preprocessing steps (scaling, encoding) and training logic are encapsulated inside the saved pipeline.

Of course. Here is a more professional and detailed "Future Improvements" section that you can use in your `README.md` file.

***

### 🔮 Future Improvements

This project provides a solid foundation for a real-time fraud detection system. The following enhancements could be implemented to further improve its performance, robustness, and scalability:

* **Advanced Model Tuning:**
    * Utilize `GridSearchCV` or `RandomizedSearchCV` to systematically find the optimal hyperparameters for the `RandomForestClassifier`, which could lead to significant gains in precision and recall.
    * Experiment with other state-of-the-art gradient boosting models like **XGBoost** or **LightGBM**, which are often top performers on tabular data and may offer better accuracy.

* **Production-Ready API Enhancements:**
    * **Input Validation:** Implement rigorous validation for incoming JSON data to ensure all required features are present and correctly formatted, preventing errors during prediction.
    * **API Authentication:** Secure the prediction endpoint by requiring an API key for access, ensuring that only authorized services can use the model.
    * **Containerization:** Package the Flask application and its dependencies into a **Docker** container. This will create a portable, consistent, and scalable environment, making it easy to deploy on any cloud platform.

* **Continuous Improvement Pipeline:**
    * **Model Monitoring:** In a live environment, set up a system to monitor the model's predictions for signs of "concept drift," where its performance degrades as fraud patterns change over time.
    * **Automated Retraining:** Build a pipeline to periodically retrain the model on new, incoming transaction data to ensure it stays current and effective against emerging fraud techniques.
