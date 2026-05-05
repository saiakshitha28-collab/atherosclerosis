import pickle
import numpy as np
import json
from datetime import datetime
from flask import Flask, render_template, request


app = Flask(__name__)

model, scaler = pickle.load(open("heart_model.pkl","rb"))

# Dashboard page
@app.route('/')
def dashboard():
    return render_template("dashboard.html")

# Heart predictor page
@app.route('/predictor')
def predictor():
    return render_template("predictor.html")


@app.route('/predict', methods=['POST'])
def predict():



    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    bp = float(request.form['bp'])
    chol = float(request.form['chol'])
    hr = float(request.form['hr'])

    input_data = [[age, sex, cp, bp, chol, hr]]

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    risk = round(probability[0][1]*100,2)

    if prediction[0] == 1:
        result = "Heart Disease Risk Detected"
    else:
        result = "No Heart Disease Risk"


    # Doctor suggestion logic
    doctor = None

    if risk > 70:
    	doctor = {
        	"name": "Dr. Bharat Vijay Purohit",
        	"specialization": "Interventional Cardiologist",
        	"hospital": "Yashoda Hospitals, Hyderabad",
        	"image": "doctor1.jpg"
    	}

    elif risk > 40:
    	doctor = {
        	"name": "Dr. Kiran Teja Varigonda",
        	"specialization": "Cardiologist",
        	"hospital": "Apollo Hospitals, Jubilee Hills",
        	"image": "doctor2.jpg"
    	}

    else:
    	doctor = {
        	"name": "Dr. Sneha Reddy",
        	"specialization": "General Physician",
        	"hospital": "Yashoda Hospitals, Hyderabad",
        	"image": "doctor3.jpg"
    	}
	
    # SAVE REPORT
    record = {
        "age": age,
        "sex": sex,
        "bp": bp,
        "chol": chol,
        "hr": hr,
        "risk": risk,
        "date": str(datetime.now().date())
    }

    try:
        with open("reports.json","r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(record)

    with open("reports.json","w") as f:
        json.dump(data,f)

    return render_template(
    "predictor.html",
    prediction_text=result,
    risk=risk,
    doctor=doctor
)

@app.route('/reports')
def reports():

    try:
        with open("reports.json","r") as f:
            data = json.load(f)
    except:
        data = []

    return render_template("reports.html", reports=data)

@app.route('/settings')
def settings():
    return render_template("settings.html")

if __name__ == "__main__":
    app.run(debug=True)