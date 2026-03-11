# 🔍 TruthLens

**Real-time hallucination detection for LLM outputs.**

TruthLens extracts factual claims from any LLM response, retrieves evidence from a FAISS vector store, and cross-verifies each claim — running 100% locally via Ollama, no API keys required.

![CI](https://github.com/MrSmartypants7/truthlens/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=flat-square&logo=fastapi)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-gray?style=flat-square)

---

**[🌐 Live Demo →](https://YOUR_USERNAME.github.io/truthlens)**

---

## How It Works

```
LLM Response → Claim Extraction → FAISS Evidence Retrieval → Cross-Verification → Reliability Score
```

1. **Claim Extraction** — LLM breaks the response into individual, verifiable factual claims
2. **Evidence Retrieval** — Each claim is embedded and searched against a FAISS vector index
3. **Cross-Verification** — Secondary LLM prompt evaluates each claim against retrieved evidence
4. **Reliability Score** — Claims classified as `supported`, `contradicted`, `partially_supported`, or `unverifiable`

---

## Quick Start

### 1. Install Ollama and pull models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Clone and install

```bash
git clone https://github.com/MrSmartypants7/truthlens.git
cd truthlens
pip install -r requirements.txt
```

### 3. Ingest sample knowledge base

```bash
python scripts/demo.py
```

### 4. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** and paste any LLM response to verify it.

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/verify` | Verify an LLM response |
| `POST` | `/ingest` | Add documents to the knowledge base |
| `GET`  | `/health` | Health check |
| `GET`  | `/stats`  | Index and cache statistics |
| `POST` | `/benchmark` | Run evaluation benchmarks |
| `GET`  | `/experiments` | List benchmark results |

### Example: Verify

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"llm_response": "The Eiffel Tower is 324 meters tall and was built in 1889."}'
```

```json
{
  "overall_reliability_score": 0.91,
  "total_claims": 2,
  "supported_claims": 2,
  "claim_verifications": [
    {
      "claim": {"text": "The Eiffel Tower is 324 meters tall"},
      "status": "supported",
      "confidence": 0.94,
      "reasoning": "Height confirmed by knowledge base sources."
    }
  ]
}
```

---

## Architecture

```
truthlens/
├── app/           # FastAPI app, config, structured logging
├── core/          # Pipeline, verifier, FAISS store, embeddings, chunker
├── models/        # Pydantic request/response schemas
├── static/        # React frontend (TruthLens.jsx + index.html)
├── tests/         # 22 unit tests (standalone + pytest)
├── scripts/       # Demo and ingestion scripts
├── Dockerfile
└── .env.example
```

---

## Performance

| Optimization | Impact |
|---|---|
| Batched FAISS search | Single call for all N claims |
| Embedding TTL cache | ~200ms saved per repeated query |
| Sentence-aware chunking | More precise vector matches |
| `format="json"` in Ollama | Eliminates JSON parse failures |

---

## Tests

```bash
python tests/test_standalone.py   # No Ollama needed
pytest tests/ -v                  # Full suite
```

## Docker

```bash
docker build -t truthlens .
docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 truthlens
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome.

## License

MIT
