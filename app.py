#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gradio application for heart failure patient survival prediction with Prometheus metrics
"""

import gradio as gr
import joblib
import pandas as pd
import os
from prometheus_client import Counter, Histogram, start_http_server, make_asgi_app
from prometheus_client import Info, Gauge
import time
from functools import wraps

MODEL_PATH = "xgboost-model.pkl"

# Define Prometheus metrics
PREDICTION_COUNT = Counter('heart_failure_predictions_total', 'Total number of predictions made')
PREDICTION_LATENCY = Histogram('heart_failure_prediction_latency_seconds', 'Time spent processing prediction requests')
DEATH_EVENT_PREDICTIONS = Counter('death_event_predictions', 'Number of death event predictions')
SURVIVAL_PREDICTIONS = Counter('survival_predictions', 'Number of survival predictions')
MODEL_VERSION = Info('model_version', 'Version of the currently loaded model')
DEATH_PROBABILITY_GAUGE = Gauge('last_prediction_death_probability', 'Probability of death event for the last prediction')

# Track additional input metrics
AGE_INPUT = Histogram('patient_age', 'Age of patients for prediction', buckets=[20, 30, 40, 50, 60, 70, 80, 90, 100])
EJECTION_FRACTION_INPUT = Histogram('patient_ejection_fraction', 'Ejection fraction of patients', buckets=[10, 20, 30, 40, 50, 60, 70, 80])

def timing_metrics(metric):
    """Decorator to measure function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            metric.observe(elapsed_time)
            return result
        return wrapper
    return decorator

def load_model(model_path=MODEL_PATH):
    """Load the trained model from file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run train_model.py first.")
    model = joblib.load(model_path)
    
    # Set model version info
    MODEL_VERSION.info({'path': model_path, 'file_size': str(os.path.getsize(model_path))})
    return model

@timing_metrics(PREDICTION_LATENCY)
def predict_death_event(features, model):
    """
    Predict the probability of death event based on patient features

    Parameters:
    -----------
    features : array-like or pandas DataFrame
        Patient features in the same format and order as used during training
    model : XGBoost model
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

    # Update metrics
    PREDICTION_COUNT.inc()
    if prediction == 1:
        DEATH_EVENT_PREDICTIONS.inc()
    else:
        SURVIVAL_PREDICTIONS.inc()
    
    DEATH_PROBABILITY_GAUGE.set(probability)

    return prediction, probability

def gradio_wrapper(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                  high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                  sex, smoking, time):
    """Wrapper function for Gradio interface"""
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

    # Load model
    model = load_model()

    # Call the prediction function
    prediction, probability = predict_death_event(features, model)

    # Format results for Gradio outputs
    result = "Death Event" if prediction == 1 else "Survival"
    prob_percentage = float(probability * 100)

    # Record some additional metrics for monitoring
    AGE_INPUT.observe(age)
    EJECTION_FRACTION_INPUT.observe(ejection_fraction)

    return result, prob_percentage

def create_interface():
    """Create and return Gradio interface"""
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
        allow_flagging='never',
        examples=[
            [65, 1, 160, 1, 20, 0, 327000, 2.7, 116, 0, 0, 8]
        ]
    )
    return iface

def main():
    """Main function to launch the Gradio app"""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found. Please run train_model.py first.")
        return
    
    # Start Prometheus metrics server on separate port
    print(f"Starting Prometheus metrics server on port 8000...")
    start_http_server(8000)
    
    # Create and launch interface
    iface = create_interface()
    
    # Get the underlying FastAPI app and mount metrics
    app = iface.app
    app.mount("/metrics", make_asgi_app())
    
    iface.launch(server_name="0.0.0.0", server_port=8001)

if __name__ == "__main__":
    main()