from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "corpus.txt"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "next_word_lstm.keras"
TOKENIZER_PATH = ARTIFACTS / "tokenizer.json"
METADATA_PATH = ARTIFACTS / "metadata.json"
EXPERIMENT_NAME = "next-word-prediction"


def load_corpus() -> list[str]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Training corpus not found: {CORPUS_PATH}")
    text = CORPUS_PATH.read_text(encoding="utf-8")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return [line for line in lines if len(line.split()) >= 2]


def prepare_sequences(lines: list[str], vocab_limit: int = 8000):
    tokenizer = Tokenizer(num_words=vocab_limit, oov_token="<unk>", filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(lines)
    sequences = []
    for line in lines:
        token_ids = tokenizer.texts_to_sequences([line])[0]
        for end in range(2, len(token_ids) + 1):
            sequences.append(token_ids[:end])
    max_len = max(len(sequence) for sequence in sequences)
    padded = pad_sequences(sequences, maxlen=max_len, padding="pre")
    x, y = padded[:, :-1], padded[:, -1]
    vocab_size = min(vocab_limit, len(tokenizer.word_index) + 1)
    return tokenizer, x, y, max_len, vocab_size


def build_model(vocab_size: int, sequence_length: int, variant: str) -> keras.Model:
    inputs = keras.Input(shape=(sequence_length,), dtype="int32")
    x = layers.Embedding(vocab_size, 128, mask_zero=True)(inputs)
    units = 128 if variant == "lstm_v1" else 192
    x = layers.LSTM(units, dropout=0.2, recurrent_dropout=0.0)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=variant)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy"), keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy")],
    )
    return model


def train(epochs: int = 35, batch_size: int = 32, vocab_limit: int = 8000) -> dict:
    lines = load_corpus()
    tokenizer, x, y, max_len, vocab_size = prepare_sequences(lines, vocab_limit)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(x))
    split = max(1, int(len(indices) * 0.9))
    train_idx, val_idx = indices[:split], indices[split:]
    if len(val_idx) == 0:
        raise ValueError("Corpus produced too few training sequences for a validation split.")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri()))
    mlflow.set_experiment(EXPERIMENT_NAME)
    candidates = []

    for variant in ("lstm_v1", "lstm_v2"):
        with mlflow.start_run(run_name=variant) as run:
            model = build_model(vocab_size, max_len - 1, variant)
            history = model.fit(
                x[train_idx], y[train_idx],
                validation_data=(x[val_idx], y[val_idx]),
                epochs=epochs, batch_size=batch_size,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
                ],
                verbose=2,
            )
            val_loss, val_accuracy, val_top3 = model.evaluate(x[val_idx], y[val_idx], verbose=0)
            mlflow.log_params({
                "architecture": variant, "lstm_units": 128 if variant == "lstm_v1" else 192,
                "embedding_dim": 128, "epochs_requested": epochs, "batch_size": batch_size,
                "vocab_limit": vocab_limit, "vocab_size": vocab_size,
                "sequence_length": max_len - 1, "corpus_lines": len(lines),
                "training_sequences": len(train_idx), "validation_sequences": len(val_idx),
            })
            mlflow.log_metrics({
                "val_loss": float(val_loss), "val_accuracy": float(val_accuracy),
                "val_top3_accuracy": float(val_top3),
                "epochs_completed": len(history.history["loss"]),
            })
            mlflow.tensorflow.log_model(model, artifact_path="model")
            candidates.append({
                "architecture": variant, "run_id": run.info.run_id,
                "val_loss": float(val_loss), "val_accuracy": float(val_accuracy),
                "val_top3_accuracy": float(val_top3),
            })

    selected = min(candidates, key=lambda item: item["val_loss"])
    best_model = mlflow.tensorflow.load_model(f"runs:/{selected['run_id']}/model")
    best_model.save(MODEL_PATH)
    TOKENIZER_PATH.write_text(tokenizer.to_json(), encoding="utf-8")
    metadata = {
        "corpus": "CampusX DSMP FAQ-style text corpus included in data/corpus.txt",
        "corpus_lines": len(lines), "sequence_count": len(x),
        "vocabulary_size": vocab_size, "max_sequence_length": max_len - 1,
        "selected_model": selected["architecture"], "selection_metric": "validation_loss",
        "candidates": candidates,
        "limitations": "The corpus is small and domain-specific. Predictions are next-token suggestions, not factual answers or a general-purpose language model.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vocab-limit", type=int, default=8000)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.vocab_limit)
