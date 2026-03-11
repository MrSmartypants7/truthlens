# Contributing to TruthLens

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/MrSmartypants7/truthlens.git
cd truthlens
pip install -r requirements.txt
```

Pull the required Ollama models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Running Tests

```bash
# No Ollama needed
python tests/test_standalone.py

# Full suite (requires pip install pytest)
pytest tests/ -v
```

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `app/` | FastAPI application, config, logging |
| `core/` | Pipeline, verifier, embeddings, FAISS store |
| `models/` | Pydantic schemas (request/response contracts) |
| `static/` | React frontend (`TruthLens.jsx`, `index.html`) |
| `tests/` | Unit and integration tests |
| `scripts/` | Demo and ingestion scripts |

## Making Changes

1. Fork the repo and create a feature branch: `git checkout -b feat/my-feature`
2. Make your changes and add tests where relevant
3. Run the test suite to confirm nothing is broken
4. Open a pull request with a clear description

## Areas for Contribution

- Additional LLM provider support (OpenAI, Gemini)
- Improved claim extraction prompts
- More comprehensive benchmark datasets
- UI improvements and accessibility
- Docker Compose setup with Ollama bundled
