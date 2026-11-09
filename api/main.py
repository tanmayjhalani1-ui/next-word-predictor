from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import load_assets, predict_next_words

app = FastAPI(title="Next Word Prediction API", version="1.0.0")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    count: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)


@app.get("/health")
def health():
    return {"status": "ok", "service": "next-word-predictor"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        model, tokenizer, metadata = load_assets()
        suggestions = predict_next_words(
            request.text, model, tokenizer,
            metadata.get("max_sequence_length", 40),
            request.count, request.temperature,
        )
        return {"input": request.text, "suggestions": suggestions}
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
