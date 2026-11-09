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
    lines = [re.sub(r"\\s+", " ", line).strip() for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if len(line.split()) >= 2]


def encode_lines(lines: list[str], tokenizer: Tokenizer, max_len: int | None = None):
    sequences = []
    for line in lines:
        token_ids = tokenizer.texts_to_sequences([line])[0]
        for end in range(2, len(token_ids) + 1):
            sequences.append(token_ids[:end])
    if not sequences:
        raise ValueError("The corpus did not produce any next-word training sequences.")
    length = max_len or max(map(len, sequences))
    padded = pad_sequences(sequences, maxlen=length, padding="pre")
    return padded[:, :-1], padded[:, -1], length


def build_model(vocab_size: int, sequence_length: int, variant: str) -> keras.Model:
    inputs = keras.Input(shape=(sequence_length,), dtype="int32")
    x = layers.Embedding(vocab_size, 128, mask_zero=True)(inputs)
    units = 128 if variant == "lstm_v1" else 192
    x = layers.LSTM(units, dropout=0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name=variant)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
    return model


def train(epochs: int = 35, batch_size: int = 32, vocab_limit: int = 8000) -> dict:
    lines = load_corpus()
    rng = np.random.default_rng(42)
    order = rng.permutation(len(lines))
    split = max(1, int(len(order) * 0.9))
    train_lines = [lines[i] for i in order[:split]]
    val_lines = [lines[i] for i in order[split:]]
    if not val_lines:
        raise ValueError("Corpus needs at least two usable lines for a validation split.")

    tokenizer = Tokenizer(
        num_words=vocab_limit,
        oov_token="<unk>",
        filters='!"#$%&()*+,-./:;=?@[\\\\]^_`{|}~\\t\\n',
    )
    tokenizer.fit_on_texts(train_lines)
    x_train, y_train, sequence_length = encode_lines(train_lines, tokenizer)
    x_val, y_val, _ = encode_lines(val_lines, tokenizer, max_len=sequence_length + 1)
    vocab_size = min(vocab_limit, len(tokenizer.word_index) + 1)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri()))
    mlflow.set_experiment(EXPERIMENT_NAME)
    candidates = []

    for variant in ("lstm_v1", "lstm_v2"):
        with mlflow.start_run(run_name=variant) as run:
            model = build_model(vocab_size, sequence_length, variant)
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs, batch_size=batch_size,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
                ],
                verbose=2,
            )
            val_loss, val_accuracy, val_top3 = model.evaluate(x_val, y_val, verbose=0)
            mlflow.log_params({
                "architecture": variant, "lstm_units": 128 if variant == "lstm_v1" else 192,
                "embedding_dim": 128, "epochs_requested": epochs, "batch_size": batch_size,
                "vocab_limit": vocab_limit, "vocab_size": vocab_size,
                "sequence_length": sequence_length, "corpus_lines": len(lines),
                "training_lines": len(train_lines), "validation_lines": len(val_lines),
                "training_sequences": len(x_train), "validation_sequences": len(x_val),
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
        "corpus": "Curated CampusX DSMP FAQ-style text included in data/corpus.txt",
        "corpus_lines": len(lines), "training_lines": len(train_lines), "validation_lines": len(val_lines),
        "training_sequences": len(x_train), "validation_sequences": len(x_val),
        "vocabulary_size": vocab_size, "max_sequence_length": sequence_length,
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
