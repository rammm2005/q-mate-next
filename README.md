# CodeQ-Mate: Context-Aware Question Answering for Internal Software Repositories

**Using Hybrid BM25 and Semantic Retrieval**

CodeQ-Mate is an AI-powered question answering system designed for internal software repositories. It combines lexical retrieval (BM25) with semantic retrieval (sentence-transformer embeddings) to help developers find source code, documentation, and function implementations quickly and accurately.

## System Architecture

```
Developer Question
       ↓
Query Processing (Intent Classification + Query Expansion)
       ↓
Hybrid Retrieval (BM25 + Semantic Search → Reciprocal Rank Fusion)
       ↓
Context Assembly (Token Budget Management)
       ↓
LLM Answer Generation (Source-Grounded)
       ↓
Source-Grounded Response (with file paths, function names, line numbers)
```

## Key Features

- **Hybrid Retrieval**: Combines BM25 (exact matching) + Semantic Search (conceptual understanding)
- **Code-Aware Tokenization**: Understands camelCase, snake_case, and dot notation naming conventions
- **Reciprocal Rank Fusion (RRF)**: Merges results from both retrievers with configurable weighting
- **Source-Grounded Answering**: Answers include file references, function names, and line numbers
- **Query Intent Classification**: Automatically classifies questions (Code Lookup, Documentation, API Usage, Architecture, Debugging)
- **Repository Ingestion**: AST-based code chunking with secret detection
- **Evaluation Metrics**: Precision@K, Recall@K, MRR, Answer Accuracy, Retrieval Latency

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14, React 18, TypeScript |
| Backend | FastAPI, Python 3.11+ |
| Database | Supabase (PostgreSQL + pgvector) |
| Embedding | Sentence Transformers (all-MiniLM-L6-v2, 384-dim) |
| Lexical Search | BM25 (custom implementation) |
| Vector Search | pgvector (HNSW index, cosine similarity) |
| LLM | Google Gemini 1.5 Flash (for readable answer generation) |
| Testing | pytest, hypothesis (property-based testing) |

## Project Structure

```
codeq-mate/
├── backend/                     # FastAPI Backend (Python)
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py        # API endpoints (POST /api/query, /api/ingest)
│   │   ├── models/
│   │   │   ├── answer.py        # GroundedAnswer, SourceReference
│   │   │   ├── chunk.py         # CodeChunk, ChunkMetadata, ChunkType
│   │   │   ├── query.py         # ProcessedQuery, QueryIntent, QueryFilters
│   │   │   └── retrieval.py     # ScoredChunk, RetrievalResult
│   │   ├── services/
│   │   │   ├── answer_generator.py   # LLM prompt construction & answer generation
│   │   │   ├── bm25_engine.py        # BM25 inverted index & scoring
│   │   │   ├── chunker.py            # AST-based code chunking
│   │   │   ├── evaluation.py         # Precision@K, Recall@K, MRR metrics
│   │   │   ├── hybrid_retriever.py   # RRF fusion + query filtering
│   │   │   ├── ingestion.py          # Repository parser & file walker
│   │   │   ├── query_processor.py    # Intent classification & query expansion
│   │   │   ├── semantic_retriever.py # Sentence-transformer embedding & pgvector search
│   │   │   └── tokenizer.py          # Code-aware tokenization
│   │   ├── utils/
│   │   │   └── tokenizer.py     # Token estimation (tiktoken)
│   │   └── main.py              # FastAPI app entry point
│   ├── tests/                   # 520+ unit tests
│   └── requirements.txt
├── frontend/                    # Next.js Frontend (TypeScript)
│   ├── app/
│   │   ├── components/
│   │   │   ├── AnswerCard.tsx        # Answer display with source references
│   │   │   └── SourceReference.tsx   # Expandable source reference card
│   │   ├── globals.css          # Full styling (light/dark mode)
│   │   ├── layout.tsx           # Root layout
│   │   └── page.tsx             # Chat interface
│   └── package.json
├── supabase/
│   └── migrations/
│       └── 001_initial_schema.sql   # Database schema (pgvector, users, repos, chunks)
├── .gitignore
└── README.md
```

## Installation & Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Supabase account (free tier works)

### 1. Clone Repository

```bash
git clone <repository-url>
cd codeq-mate
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Database Setup (Supabase)

1. Create a new project at [supabase.com](https://supabase.com)
2. Run the migration SQL in the SQL Editor:
   - Copy the contents of `supabase/migrations/001_initial_schema.sql`
   - Paste and execute in the Supabase SQL Editor

### 5. Environment Variables

Create a `.env` file in the `backend/` folder:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
GEMINI_API_KEY=your-gemini-api-key  # Get from https://aistudio.google.com/apikey
```

## Running the Application

### Backend (FastAPI)

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

The API will run at `http://localhost:8000`

### Frontend (Next.js)

```bash
cd frontend
npm run dev
```

The frontend will run at `http://localhost:3000`

## API Endpoints

### POST /api/query

Accepts a developer question and returns a source-grounded answer.

**Request:**
```json
{
  "question": "How does the login system validate users?",
  "filters": {
    "languages": ["python", "typescript"],
    "paths": ["src/**/*.py"],
    "repo_ids": ["repo-1"]
  }
}
```

**Headers:**
```
X-API-Key: your-api-key
```

**Response:**
```json
{
  "answer": "The login system validates users through the validate_token() function [Source 1] which checks JWT tokens...",
  "sources": [
    {
      "file_path": "src/auth/middleware.py",
      "function_name": "validate_token",
      "start_line": 45,
      "end_line": 67,
      "snippet": "def validate_token(token: str):\n    ...",
      "relevance": 0.92
    }
  ],
  "confidence": 0.85,
  "metadata": {"chunks_used": 3, "total_tokens": 1240}
}
```

### POST /api/ingest

Indexes a repository into the search system.

**Request:**
```json
{
  "repository_path": "/path/to/your/repo",
  "config": {
    "languages": ["python", "typescript"],
    "exclude_patterns": ["node_modules/**", "*.test.*"]
  }
}
```

### GET /health

Health check endpoint.

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

### Test Results

**520 tests passed** ✅ (0 failures)

```
======================== test session starts ========================
platform win32 -- Python 3.13.11, pytest-9.0.3
collected 520 items

tests/test_answer_generator.py       .... 47 passed
tests/test_api_routes.py             .... 26 passed
tests/test_bm25_engine.py            .... 39 passed
tests/test_chunker.py                .... 39 passed
tests/test_code_tokenizer.py         .... 45 passed
tests/test_evaluation.py             .... 51 passed
tests/test_hybrid_retriever.py       .... 38 passed
tests/test_ingestion.py              .... 48 passed
tests/test_ingestion_pipeline.py     .... 21 passed
tests/test_models.py                 .... 35 passed
tests/test_process_query.py          .... 32 passed
tests/test_query_processor.py        .... 74 passed
tests/test_semantic_retriever.py     .... 20 passed
tests/test_tokenizer.py              ....  6 passed
────────────────────────────────────────────────────
TOTAL                                     520 passed in 16.66s
======================== 520 passed, 0 failures ========================
```

> Full verbose test output is available at [`docs/test-results.txt`](docs/test-results.txt)

### Test Coverage by Component

| Component | Test File | Tests |
|-----------|-----------|-------|
| Answer Generator | `test_answer_generator.py` | 47 |
| API Routes | `test_api_routes.py` | 26 |
| BM25 Engine | `test_bm25_engine.py` | 39 |
| Code Chunker | `test_chunker.py` | 39 |
| Code Tokenizer | `test_code_tokenizer.py` | 45 |
| Evaluation Metrics | `test_evaluation.py` | 51 |
| Hybrid Retriever (RRF) | `test_hybrid_retriever.py` | 38 |
| Repository Ingestion | `test_ingestion.py` | 48 |
| Ingestion Pipeline | `test_ingestion_pipeline.py` | 21 |
| Data Models | `test_models.py` | 35 |
| Query Processing | `test_process_query.py` | 32 |
| Query Processor | `test_query_processor.py` | 74 |
| Semantic Retriever | `test_semantic_retriever.py` | 20 |
| Token Estimation | `test_tokenizer.py` | 6 |
| **Total** | **14 test files** | **520** |

## System Components

### 1. Code-Aware Tokenization

Splits identifiers based on programming naming conventions:
- `getUserName` → `[getusername, get, user, name]`
- `get_user_name` → `[get_user_name, get, user, name]`
- `module.class.method` → `[module.class.method, module, class, method]`

Reference: Arwan et al. (SIET 2023) — "Tokenization in source code requires splitting camelCase and snake_case identifiers (e.g., AddPatientAction, Patient_Name). Each word on the form should be separated."

### 2. BM25 Engine

Custom BM25 implementation with the standard scoring formula:

```
score(Q, D) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1 × (1-b + b×|D|/avgdl))
```

Where:
- `IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)`
- `k1 = 1.5` (term frequency saturation)
- `b = 0.75` (document length normalization)

### 3. Semantic Retriever

- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Index**: HNSW via pgvector for approximate nearest neighbor search
- **Similarity**: Cosine similarity
- **Batch Processing**: Up to 256 chunks per embedding batch

### 4. Reciprocal Rank Fusion (RRF)

Merges two ranked result lists into a single fused ranking:

```
score(d) = α × (1/(k + rank_bm25(d))) + (1-α) × (1/(k + rank_semantic(d)))
```

- `α = 0.5` (configurable weight between BM25 and semantic, range 0.0-1.0)
- `k = 60` (RRF constant, prevents excessive weight on top-ranked items)

### 5. Query Intent Classification

Deterministic keyword-based classification with priority ordering:

| Intent | Example Keywords | Query Expansion |
|--------|----------|-----------|
| DEBUGGING | bug, error, fix, crash | Base tokens only |
| API_USAGE | api, endpoint, route, http | `@app.get("/token")`, `router.token` |
| ARCHITECTURE | architecture, module, dependency | import, module, package |
| DOCUMENTATION | explain, how does, what is | readme, docs, guide |
| CODE_LOOKUP | find, where is, function, class | `def token`, `function token`, `class token` |

### 6. Evaluation Metrics

Based on standard IR evaluation (Arwan et al., SIET 2023):

```
Precision@K = |relevant ∩ retrieved[:K]| / K
Recall@K    = |relevant ∩ retrieved[:K]| / |relevant|
MRR         = (1/|Q|) × Σ (1/rank_i)
```

Computed at K = 1, 3, 5, 10

## Database Schema

### Tables

| Table | Purpose |
|-------|---------|
| `users` | API key authentication, repository access control list |
| `repositories` | Repository metadata, ingestion status, file/chunk counts |
| `code_chunks` | Source code chunks with 384-dim embedding vector, BM25 terms, metadata |

### Indexes

- **HNSW** on embedding column (fast approximate nearest neighbor search)
- **GIN** on bm25_terms array (inverted index for lexical search)
- **GIN** on tags array (tag-based filtering)
- **B-tree** on repo_id, language, chunk_type, file_path (filtering)

## How It Works (End-to-End Flow)

1. **Developer asks a question** via the chat interface
2. **Query Processor** classifies intent, expands query terms, generates embedding
3. **BM25 Engine** searches the inverted index for exact token matches
4. **Semantic Retriever** performs vector similarity search via pgvector
5. **Hybrid Retriever** fuses both result sets using Reciprocal Rank Fusion
6. **Answer Generator** selects top chunks within token budget, builds prompt
7. **LLM** (Gemini 1.5 Flash) generates a readable natural language answer citing sources
8. **Response** is returned with answer text, source references, and confidence score

## References

1. Arwan, A., Rochimah, S., & Fatichah, C. (2023). Feature Location Using Extraction of Code Documentation. *International Conference on Sustainable Information Engineering and Technology (SIET 2023)*. ACM. https://doi.org/10.1145/3626641.3627149

2. Robertson, S. E., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

3. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.

4. Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. *SIGIR 2009*.

5. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

## License

This project was created as a final assignment for the Information Retrieval course (STKI).

---

**Created by:** [Student Name]  
**Student ID:** [NIM]  
**Course:** Information Retrieval Systems (STKI)  
**Semester:** 6  
**Year:** 2026
