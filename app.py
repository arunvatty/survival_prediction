import numpy as np
import pandas as pd
import joblib
import gradio as gr
from xgboost import XGBClassifier

# Load your trained model
model = joblib.load("xgboost-model.pkl")

# Function for prediction
def predict_death_event(features, model=model):
    """
    Predict the probability of death event based on patient features

    Parameters:
    -----------
    features : array-like or pandas DataFrame
        Patient features in the same format and order as used during training
    model : XGBClassifier
        Trained XGBoost model

    Returns:
    --------
    prediction : int
        Predicted class (0 or 1)
    probability : float
        Probability of the positive class (1)
    """
    # Convert dict to DataFrame if necessary
    if isinstance(features, dict):
        features = pd.DataFrame([features])

    # Make prediction (class)
    prediction = model.predict(features)[0]

    # Get probability of positive class (1)
    probability = model.predict_proba(features)[0][1]

    return prediction, probability

def gradio_wrapper(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                  high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                  sex, smoking, time):
    # Format the input as a dictionary
    patient_data = {
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking,
        'time': time
    }

    # Convert to DataFrame for the model
    features = pd.DataFrame([patient_data])

    # Call the original prediction function
    prediction, probability = predict_death_event(features)

    # Format results in a way compatible with Gradio outputs
    result = "Death Event" if prediction == 1 else "Survival"
    prob_percentage = float(probability * 100)

    return result, prob_percentage

# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gr.Interface(
    fn=gradio_wrapper,
    inputs=[
        gr.Slider(minimum=20, maximum=100, step=1, label="Age"),
        gr.Radio(choices=[0, 1], label="Anaemia", info="0 = No, 1 = Yes"),
        gr.Number(label="Creatinine Phosphokinase (CPK)", info="Level of CPK enzyme in the blood (mcg/L)"),
        gr.Radio(choices=[0, 1], label="Diabetes", info="0 = No, 1 = Yes"),
        gr.Slider(minimum=10, maximum=80, step=1, label="Ejection Fraction (%)", info="Percentage of blood leaving the heart at each contraction"),
        gr.Radio(choices=[0, 1], label="High Blood Pressure", info="0 = No, 1 = Yes"),
        gr.Number(label="Platelets", info="Platelets in the blood (kiloplatelets/mL)"),
        gr.Number(label="Serum Creatinine", info="Level of creatinine in the blood (mg/dL)"),
        gr.Number(label="Serum Sodium", info="Level of sodium in the blood (mEq/L)"),
        gr.Radio(choices=[0, 1], label="Sex", info="0 = Female, 1 = Male"),
        gr.Radio(choices=[0, 1], label="Smoking", info="0 = No, 1 = Yes"),
        gr.Number(label="Follow-up Period", info="Follow-up period (days)")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Probability (%)")
    ],
    title=title,
    description=description,
    allow_flagging='never'
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8001)  # Using specific host and port for container deployment