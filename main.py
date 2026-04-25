from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"
)

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/analyze")
def analyze_sentiment(data: TextInput):
    if not data.text.strip():
        return {"error": "Empty input not allowed"}

    result = sentiment_pipeline(data.text)

    return {
        "input": data.text,
        "prediction": result[0]["label"],
        "confidence": round(result[0]["score"], 4)
    }
@app.post("/analyze-batch")
def analyze_batch(texts: list[str]):
    return sentiment_pipeline(texts)
