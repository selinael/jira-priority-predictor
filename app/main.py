from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# ---- Load model once at startup ----
model_path = "models/priority_clf.joblib"
pipeline = joblib.load(model_path)

# ---- App setup ----
app = FastAPI(title="Ticket Priority Predictor", version="1.0")

class Ticket(BaseModel):
    title: str
    description: str

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(ticket: Ticket):
    # Join title + description into one text blob
    text = f"{ticket.title} {ticket.description}".strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty ticket text provided")

    # Get predicted label
    pred = pipeline.predict([text])[0]

    # Try to get confidence
    try:
        # Works for LogisticRegression
        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            probs = pipeline.predict_proba([text])[0]
            confidence = float(np.max(probs))
        else:
            # For LinearSVC: decision_function -> softmax-ish
            scores = pipeline.decision_function([text])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            confidence = float(np.max(probs))
    except Exception:
        confidence = None

    # Extract top terms (basic: get top weighted features for class)
    top_terms = []
    try:
        vectorizer = pipeline.named_steps["tfidfvectorizer"]
        clf = pipeline.named_steps["clf"]
        feature_names = np.array(vectorizer.get_feature_names_out())
        class_idx = list(clf.classes_).index(pred)
        coefs = clf.coef_[class_idx]
        top_indices = np.argsort(coefs)[-5:][::-1]
        top_terms = feature_names[top_indices].tolist()
    except Exception:
        pass

    return {
        "priority": pred,
        "confidence": confidence,
        "top_terms": top_terms
    }
