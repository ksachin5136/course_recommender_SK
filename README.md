# Course Recommendation Engine (LangChain + ChromaDB + Gemini)

A personalized **RAG-based recommender** that returns **top-5 courses** from a catalog using:

- **Semantic embeddings** (Google `text-embedding-004` or HuggingFace fallback)
- **ChromaDB** for vector search
- **Gemini LLM** for rationale and re-ranking

```
course-recommender/
├─ README.md
├─ requirements.txt
├─ .env                           # your Google API key and config
├─ data/
│  └─ assignment2dataset.csv      # auto-downloaded at first run; can also be placed manually
├─ storage/
│  └─ chroma/                     # ChromaDB persistence directory
├─ src/
│  ├─ app.py                      # FastAPI app
├─ scripts/
│  └─ evaluation_notebook.ipynb   # runs 5 sample queries & saves evaluation artifacts
```

---

## ✅ Features
- Auto-downloads dataset on first run
- Persistent **ChromaDB** index (`storage/chroma/`)
- **FastAPI** service with `/recommend` endpoint
- **Evaluation notebook** for batch testing and reporting

---

## 🔑 Prerequisites
- Python **3.10+** recommended
- A valid **Google Generative AI API key** in `.env`

---

## ⚙️ .env Example
Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_api_key_here
LLM_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=models/embedding-001
USE_GOOGLE_EMBEDDINGS=true
LLM_TEMPERATURE=0.2
TOP_K_DEFAULT=5
```

---

## 🚀 Steps to Run

### 1. Clone & Install
```bash
git clone <repo-url>
cd course-recommender
pip install -r requirements.txt
```

### 2. Set Environment
Add your `.env` file with `GOOGLE_API_KEY` and other configs.

### 3. Start FastAPI Service
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
- On first run, the app will **download the dataset** and **build the Chroma index** automatically.

### 4. Health Check
```bash
curl http://localhost:8000/health
```

### 5. Get Recommendations
Send a POST request:
```bash
curl -X POST http://localhost:8000/recommend   -H "Content-Type: application/json"   -d '{"query": "I know Azure basics and want to manage containers and build CI/CD pipelines.", "top_k": 5, "use_llm_format": true}'
```

### 6. Run Evaluation Notebook
Open `scripts/evaluation_notebook.ipynb` in Jupyter:
```bash
jupyter notebook scripts/evaluation_notebook.ipynb
```
- Executes **5 sample queries**
- Saves results to `evaluation_runs/<timestamp>/`

---

## ✅ Output
- API returns JSON:
  ```json
  {
    "result": {
      "profile": {...},
      "raw": [...],
      "llm_json": [
        {"course_id": "C101", "title": "Advanced Kubernetes", "score": 0.92, "reason": "Aligns with microservices goal"}
      ]
    }
  }
  ```
- Notebook saves:
  - `full_results.json`
  - Per-query CSVs
  - Optional charts of recommendation scores

---

## 📌 Notes
- API endpoints: `/health`, `/recommend`
- Requires **GOOGLE_API_KEY** for LLM-based recommendations
- Persistent Chroma index stored at `storage/chroma/`

