import os
import joblib


def load_lid_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, "lid_model.pkl"))
    vectorizer = joblib.load(os.path.join(base_dir, "lid_vectorizer.pkl"))
    return model, vectorizer


def predict_language(word, model, vectorizer):
    word = str(word).lower().strip()
    
    if not word:       #Защита от пустых строк
        return "other"

    try:
        vec = vectorizer.transform([word])
        return model.predict(vec)[0]
    except Exception:
        return "other"
