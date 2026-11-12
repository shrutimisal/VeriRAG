# 🔎 VeriRAG – Citation-Aware Document Intelligence & RAG Evaluation System

Upload research papers, ask questions, and get answers where **every claim points back to a document and page** — then check whether each claim is really supported, and measure the whole pipeline with RAGAS.

## Problem statement
LLMs sound confident even when they are wrong (hallucination). Plain RAG reduces this by feeding the model your documents, but users still can't easily tell *where* an answer came from, *whether each sentence is actually backed by the source*, or *how good the pipeline is overall*. VeriRAG makes those three things visible.

## Features
| Area | What you get |
|---|---|
| Ingestion | Multi-PDF upload, page-by-page extraction (PyMuPDF), graceful errors for corrupt / password-protected / scanned PDFs |
| Chunking | Configurable size + overlap, sentence-aware cuts, each chunk keeps `document` + `page` |
| Embeddings | Local `all-MiniLM-L6-v2` (Sentence Transformers); chunks already in the DB are never re-embedded |
| Vector DB | ChromaDB, persisted in `data/chroma_db`, cosine similarity, filter by document |
| Generation | Gemini answers **only** from retrieved passages, says so when the answer isn't there |
| Citations | `[n]` markers mapped to real retrieved chunks (document, page, passage, score). Invented markers are discarded and flagged |
| Verification | Answer → claims → each judged *Supported / Partially Supported / Unsupported / Contradicted* with an explanation |
| Evaluation | RAGAS: Faithfulness, Answer Relevancy, Context Precision, Context Recall; sample dataset; CSV/JSON export |
| UI | Streamlit: Dashboard, Documents, Chat (answer / sources / retrieved context / verification tabs), Evaluation |

## Tech stack
Python 3.10+ · Streamlit · PyMuPDF · Sentence-Transformers · ChromaDB · Google Gemini (`google-genai` SDK) · RAGAS (+ LangChain wrappers) · python-dotenv · pytest

## Architecture
```
 PDF ─► pdf_loader ─► chunker ─► Embedder ─► ChromaStore (disk)
                                                  ▲
 Question ─► Embedder ─► Retriever ───────────────┘  top-k chunks (+scores)
                              │
                              ▼
                     AnswerGenerator (Gemini, context-only prompt)
                              │  answer with [n] markers
                              ▼
          citation_builder ─► Citations (document, page, passage)
                              │
                              ▼
          claim_splitter ─► CitationVerifier (Gemini judge) ─► verdict per claim

 Evaluation: dataset ─► real pipeline ─► (question, contexts, answer, reference) ─► RAGAS ─► scores
```

### RAG pipeline in one paragraph
1. **Index:** each PDF page → overlapping text chunks → embedding vectors → stored with `{document, doc_id, page}`.
2. **Retrieve:** the question is embedded with the same model; ChromaDB returns the nearest chunks.
3. **Generate:** the chunks are numbered `[1]..[k]` in the prompt; Gemini must answer using only them and cite with those numbers.
4. **Cite:** numbers in the answer are looked up in the retrieved list → exact document/page/passage.
5. **Verify:** each sentence is compared with the passages it cites.
6. **Evaluate:** RAGAS scores the whole flow on a question set.

## Folder structure
```
VeriRAG/
├── app/
│   ├── main.py                 # Streamlit entry point (4 pages)
│   └── components/             # styles.py, ui.py, services.py
├── src/
│   ├── config.py  models.py  pipeline.py
│   ├── ingestion/pdf_loader.py
│   ├── processing/chunker.py
│   ├── embeddings/embedder.py
│   ├── vectorstore/chroma_store.py
│   ├── retrieval/retriever.py
│   ├── generation/             # llm_client.py, generator.py
│   ├── citations/              # citation_builder.py, claim_splitter.py, verifier.py
│   ├── evaluation/             # dataset.py, ragas_runner.py, metrics_info.py
│   └── utils/                  # errors.py, logger.py, text.py
├── data/ documents/  evaluation/sample_eval_dataset.json  chroma_db/
├── tests/                      # 33 unit tests (no API key / model download needed)
├── .env.example  requirements.txt  LEARNINGS.md
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # first install is large (PyTorch)
cp .env.example .env               # Windows: copy .env.example .env
```
Edit `.env` and set `GEMINI_API_KEY` (free key: https://aistudio.google.com/apikey). Never commit `.env`.

## Run
```bash
streamlit run app/main.py
```
Run it from the project root. The first document upload downloads the embedding model (~90 MB) once.

## Usage
1. **Documents** → upload PDFs → *Index uploaded PDFs*.
2. **Chat** → ask a question. Read the answer, then open the tabs: *Sources* (cited pages), *Retrieved context* (all chunks + similarity), *Citation verification* (click *Verify citations*, or turn on auto-verify in the sidebar).
3. **Evaluation** → upload “Attention Is All You Need” (arXiv 1706.03762) and use the sample dataset, or upload your own JSON (`[{"question": "...", "reference": "..."}]`) → *Run evaluation*. Results appear as metric cards + per-question table with CSV/JSON download. Reports are also saved to `data/evaluation/results/`.

Run tests: `python -m pytest -q`

## Screenshots / demo (placeholders)
- `docs/screenshots/dashboard.png` – _add yours_
- `docs/screenshots/chat_citations.png` – _add yours_
- `docs/screenshots/verification.png` – _add yours_
- `docs/screenshots/evaluation.png` – _add yours_

## Limitations
- **Scanned PDFs** are rejected (no OCR). Tables/figures/equations are extracted as plain text.
- **LLM-as-judge** (verification and RAGAS) is an *estimate*; it can be wrong, and here Gemini both generates and judges, which can inflate scores.
- Character-based chunking, no re-ranking, no hybrid (keyword) search.
- Changing chunk size re-indexes under new chunk IDs; delete the old document first to avoid mixed chunking.
- RAGAS makes many LLM calls; on free-tier quotas it can be slow or hit rate limits (failed rows are reported, never faked).
- The RAGAS/LangChain stack is version-sensitive. `requirements.txt` pins a set that was verified to import and accept the evaluation call; newer combinations (ragas 0.4 + langchain-community 0.4) failed to import when tested. Don't upgrade these casually.

## Future scope
OCR for scanned PDFs, cross-encoder re-ranking, hybrid BM25 + vector search, streaming answers, auto-generated test sets, a different judge model than the generator, per-claim highlighting inside the source PDF page.
# VeriRAG
