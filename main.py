from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="Sentiment Analysis API 🚀")

# Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/analyze")
def analyze(data: TextInput):
    if not data.text.strip():
        return {"error": "Empty input not allowed"}

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": data.text}
    )

    if response.status_code != 200:
        return {"error": "Model API failed", "details": response.text}

    result = response.json()

    return {
        "input": data.text,
        "result": result
    }
