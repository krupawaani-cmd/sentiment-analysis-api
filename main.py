from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment API")

# Load model safely
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME
)

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API running 🚀"}

@app.post("/analyze")
def analyze(data: TextInput):
    if not data.text.strip():
        return {"error": "Empty input"}

    result = sentiment_pipeline(data.text)

    return {
        "input": data.text,
        "prediction": result[0]["label"],
        "confidence": round(result[0]["score"], 4)
    }
