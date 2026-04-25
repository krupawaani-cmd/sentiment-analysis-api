# sentiment-analysis-api

A REST API that analyzes text sentiment using Hugging Face Transformers.

## Features
- Sentiment classification (Positive/Negative)
- Batch text processing
- FastAPI backend
- Deployed on Render

## Tech Stack
- Python
- FastAPI
- Transformers (Hugging Face)
- Uvicorn

## Live Demo
https://your-app-name.onrender.com/docs

## Example Request
POST /analyze

{
  "text": "I love this project"
}

## Response
{
  "input": "I love this project",
  "prediction": "POSITIVE",
  "confidence": 0.999
}
