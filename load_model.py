# load_model.py
import joblib


def load_model():
    model = joblib.load("models/logistic_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer
