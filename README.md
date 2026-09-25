# Zepto Data & AI Platform — Capstone

One repository, three connected modules. Together they tell one story: raw data becomes clean structured data, clean data becomes predictions, and company documents become a grounded GenAI service.

| Module | Folder | What it shows | Marks |
|---|---|---|---|
| 1. Data pipeline | [`/data_pipeline`](data_pipeline/README.md) | Scrape → clean → convert (GBP→INR) → normalized SQLite → SQL + pandas queries | 25 |
| 2. Analytics pipeline | [`/analytics`](analytics/README.md) | Titanic: profiling, cleaning, EDA data story → leak-free ML pipeline, tuning, regression, saved model | 50 |
| 3. Support assistant | [`/support_assistant`](support_assistant/README.md) | RAG over Zepto policies: MiniLM embeddings + ChromaDB + LangGraph router + Pydantic + FastAPI + Docker | 25 |

## Setup

**Requirements files: one per module**, plus a root `requirements.txt` that simply includes all three (`-r data_pipeline/requirements.txt` etc.) for a one-command install.

Python **3.11 or 3.12** is recommended (pandas 3 needs ≥ 3.11). Everything runs on CPU, and no paid service or API key is needed.

```bash
git clone <this repo>
cd <repo>
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt           # everything
# or per module:  pip install -r analytics/requirements.txt
```

## Run each module end to end

**Module 1 — data pipeline** (needs internet to books.toscrape.com)
```bash
cd data_pipeline
jupyter nbconvert --to notebook --execute --inplace books_pipeline.ipynb
```
This creates `data/raw_books.csv`, `data/books.db` and `outputs/query_results.md`.

**Module 2 — analytics** (internet once for `sns.load_dataset`; falls back to the committed `titanic.csv`)
```bash
cd analytics
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb
python predict_demo.py
```

**Module 3 — support assistant** (mock mode by default; `MOCK_LLM` left unset)
```bash
cd support_assistant
python vector_store.py                  # build the ChromaDB index (downloads all-MiniLM-L6-v2 once)
uvicorn main:app --port 8000
# second terminal:
python run_examples.py                  # records example calls into support_assistant/README.md
```
Docker:
```bash
cd support_assistant
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support    # serves POST /ask on http://localhost:7860
```

## Design decisions (summary)

### Module 1 — data pipeline
- **Scope and scraping.** All books in 4 categories (Mystery, Historical Fiction, Fantasy, Poetry), about 125 books. Category URLs come from the site's sidebar, every “next” page is followed, and requests use a timeout, an explicit status-code check (`raise_for_status`), retries and a polite delay.
- **Cleaning.**
  - Unparseable numeric fields (`price_gbp`, `rating`) get **category-median imputation**, flagged with `imputed`.
  - Rows with unparseable **availability are dropped**, since a boolean has no sensible median.
  - Rows with no title or category, and duplicates, are dropped.
- **Currency.** Fixed project rate **1 GBP = 105.50 INR** (a project constant, not a market rate), giving `price_inr = price_gbp × 105.50`.
- **Storage.** A normalized schema, `categories(category_id PK, category_name UNIQUE)` ⟵ `books(…, category_id FK)`, with CHECK constraints and a foreign-key check.
- **Queries.** Six SQL queries cover WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, IN, JOIN and GROUP BY. The JOIN is reproduced with `pd.merge` and verified identical with `assert_frame_equal`.

### Module 2 — analytics
- **One load, one cleaning.**
  - `01_eda.ipynb` is the only place that calls `sns.load_dataset`, and it saves `titanic.csv` immediately. `02_modeling.ipynb` reads that file.
  - Rule-based cleaning lives in `titanic_cleaning.py`: drop the 2 rows missing `embarked` (0.22 %), and encode `deck` (77 % missing) as `"Unknown"` because its missingness predicts survival. These rules learn nothing from the data, so both notebooks can share them.
- **Missing age (19.9 %) is imputed twice, by design.**
  - For EDA: with a sex × class median.
  - For modeling: inside the Pipeline, fitted on training data only.
- **Leak-free modeling.** Stratified split first (62/38 imbalance), then `ColumnTransformer` + `Pipeline`. SMOTE sits inside an `imblearn` Pipeline, so only the training data is resampled.
- **Tuning.** GridSearchCV on F1 over `RandomForestClassifier(oob_score=True, ...)`. Best settings: `n_estimators=300, max_depth=8, max_features=None`, with OOB 0.831.
- **Deployed model.** The tuned Random Forest: test accuracy 0.826, F1 0.756, AUC 0.842. It is saved as the complete fitted pipeline with `joblib`.
- **Regression.** Predicting `fare` gives R² 0.347 and clear heteroscedasticity.

### Module 3 — support assistant
- **Ingestion and embedding.** One chunk per document (each is a short paragraph), embedded locally with `all-MiniLM-L6-v2` and stored in the ChromaDB collection `zepto_policies` (cosine space).
- **Routing.** A LangGraph `StateGraph` with 3 nodes (`classify_intent` → conditional edge → `retrieve_and_answer` / `direct_answer`).
- **The `MOCK_LLM` switch.** Only the generation step inside each node branches on it; retrieval always runs for real. The graded default (mock mode) makes no LLM or network call.
- **Output contract.** Responses are validated by the Pydantic `AskResponse(answer, sources, confidence)`. In the optional real-LLM mode (Groq free tier), invalid JSON is retried up to 2 more times with a corrective instruction.
- **Serving.** FastAPI `POST /ask`. The Dockerfile uses CPU-only PyTorch and pre-downloads the model and index at build time.

## Repository layout
```
.
├── README.md                 ← this file
├── requirements.txt          ← includes the 3 module files below
├── data_pipeline/            ← Module 1  (books_pipeline.ipynb, data/, outputs/)
├── analytics/                ← Module 2  (01_eda.ipynb, 02_modeling.ipynb, titanic.csv, *.joblib, figures/)
└── support_assistant/        ← Module 3  (docs/, graph.py, main.py, Dockerfile, ...)
```

## Git workflow
Work was done on feature branches (`feature/data-pipeline`, `feature/analytics`, `feature/support-assistant`), each with several commits, and merged into `main` with merge commits. See `git log --graph --all --oneline`.
