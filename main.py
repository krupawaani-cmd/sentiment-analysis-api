from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Working ✅"}

@app.post("/analyze")
def analyze(data: TextInput):
    if not data.text:
        return {"error": "Empty text"}

    res = requests.post(API_URL, headers=headers, json={"inputs": data.text})

    return res.json()
