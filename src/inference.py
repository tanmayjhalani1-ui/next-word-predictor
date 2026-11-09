from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "next_word_lstm.keras"
TOKENIZER_PATH = ROOT / "artifacts" / "tokenizer.json"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"


def load_assets():
    missing = [path.name for path in (MODEL_PATH, TOKENIZER_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model assets: {', '.join(missing)}. Run python -m src.train first.")
    model = keras.models.load_model(MODEL_PATH)
    tokenizer = tokenizer_from_json(TOKENIZER_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    return model, tokenizer, metadata


def predict_next_words(text: str, model, tokenizer, sequence_length: int, count: int = 5, temperature: float = 1.0) -> list[dict]:
    token_ids = tokenizer.texts_to_sequences([text])[0]
    if not token_ids:
        raise ValueError("No known tokens found. Try words related to the training corpus.")
    padded = pad_sequences([token_ids], maxlen=sequence_length, padding="pre", truncating="pre")
    logits = np.log(np.maximum(model.predict(padded, verbose=0)[0], 1e-9)) / max(temperature, 0.1)
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()
    ranked = np.argsort(probabilities)[::-1]
    results = []
    for index in ranked:
        word = next((term for term, token_id in tokenizer.word_index.items() if token_id == int(index)), None)
        if word and word != "<unk>":
            results.append({"word": word, "probability": float(probabilities[index])})
        if len(results) == count:
            break
    return results


def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
