from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle 
import os

# Load trained model, scaler, and label encoders
rf = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
X = pickle.load(open("X_columns.pkl", "rb"))  # Saved training DataFrame with columns

# Define app
app = Flask(__name__)

# Categorical and numerical columns (you can modify as needed)
categorical_cols = [
    "Fraud Type", "Multiple Login Attempts", "Unusual IP Address",
    "Previous Fraud History", "Compromised Credentials Used", "Malicious Link Clicked",
    "Blacklisted Entity Involved", "Dark Web Involvement", "Transaction Time",
    "Unusual Location Detected", "Account Type", "Social Engineering Involvement",
    "Previous Suspicious Activity", "Data Breach Exposure"
]

numerical_cols = [
    "Transaction Amount ($)", "Device Risk Score", "Login Frequency (Per Day)",
    "Transaction Location Risk", "Account Age (Days)"
]

@app.route('/')
def index():
    # Prepare dropdown options for each categorical column
    options = {col: list(label_encoders[col].classes_) for col in categorical_cols}
    return render_template("index.html", options=options)

@app.route('/predict', methods=["POST"])
def predict():
    input_data = {}
    for col in categorical_cols:
        input_data[col] = request.form[col]

    # Default numerical values (can be modified or user-input later)
    input_data.update({
        "Transaction Amount ($)": 5000,
        "Device Risk Score": 50,
        "Login Frequency (Per Day)": 30,
        "Transaction Location Risk": 0.5,
        "Account Age (Days)": 1000
    })

    custom_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in categorical_cols:
        if col in label_encoders:
            if custom_df[col][0] in label_encoders[col].classes_:
                custom_df[col] = label_encoders[col].transform(custom_df[col])
            else:
                custom_df[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])

    # Ensure all expected columns exist
    for col in numerical_cols:
        if col not in custom_df.columns:
            custom_df[col] = 0

    # Ensure column order matches training data
    custom_df = custom_df[X.columns]

    # Separate for preprocessing
    numerical_data = scaler.transform(custom_df[numerical_cols])
    categorical_data = custom_df[categorical_cols].values
    final_input = np.concatenate([numerical_data, categorical_data], axis=1)

    # Make prediction
    prediction = rf.predict(final_input)[0]
    fraud_label = "Yes" if prediction == 1 else "No"

    return render_template("result.html", prediction=fraud_label)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port)
