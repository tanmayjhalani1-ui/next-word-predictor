# Next Word Prediction with LSTM

An end-to-end next-token prediction project using TensorFlow/Keras LSTMs. It trains on a small CampusX DSMP FAQ-style corpus, compares two recurrent architectures, tracks experiments with MLflow, and serves suggestions through Streamlit and FastAPI.

## Features

- Text normalization, tokenization, OOV handling and padded prefix sequences
- Sentence-level train/validation split to avoid putting prefixes from the same sentence in both sets
- Two LSTM candidates with embedding, dropout and dense prediction head
- Early stopping and learning-rate reduction
- Validation loss, token accuracy and top-3 accuracy tracking
- MLflow logging of parameters, metrics and model artifacts
- Streamlit interface with top-k suggestions and temperature control
- FastAPI prediction endpoint with request validation
- Docker deployment and GitHub Actions CI

## Corpus

The training text is a small curated CampusX DSMP FAQ-style corpus stored in `data/corpus.txt`. The original notebook and Streamlit implementation are retained. The included corpus is intentionally transparent and domain-specific; it is not a large general-purpose language dataset and should not be interpreted as an official or current source of course information.

## Setup and training

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --epochs 35 --batch-size 32
streamlit run app.py
```

Training generates three local artifacts:

- `artifacts/next_word_lstm.keras`
- `artifacts/tokenizer.json`
- `artifacts/metadata.json`

These files are generated locally and excluded from Git. The training script compares `lstm_v1` and `lstm_v2`, selecting the candidate with the lowest validation loss.

## MLflow

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open `http://127.0.0.1:5000`. Runs are grouped under `next-word-prediction` and include architecture settings, corpus/sequence counts, validation metrics and model artifacts. They are experiment runs; a Model Registry version and promotion workflow is not configured.

## FastAPI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- `GET /health`
- `POST /predict`

Example request:

```json
{
  "text": "What is the course",
  "count": 5,
  "temperature": 0.8
}
```

API documentation: `http://127.0.0.1:8000/docs`.

## Docker

Train the model first, then run:

```bash
docker compose up --build
```

Open `http://localhost:8501`.

## Repository structure

```text
├── api/
│   └── main.py
├── data/
│   └── corpus.txt
├── src/
│   ├── train.py
│   └── inference.py
├── artifacts/              # generated locally; gitignored
├── app.py
├── nextword.py             # original Streamlit script, preserved
├── nextword.ipynb          # original notebook, preserved
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Limitations

The corpus is small and focused on one subject area, so the model will overfit easily and its suggestions may be repetitive or grammatically weak. It predicts the next token from learned sequence patterns; it does not retrieve verified answers, understand intent, or behave like a modern large language model. Do not use generated text as authoritative course, pricing or policy information.
