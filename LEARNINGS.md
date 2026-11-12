# LEARNINGS – concepts and how each module works

Use this as interview prep. Each section: **concept → where it lives → what to say**.

## 1. RAG (Retrieval-Augmented Generation)
An LLM only knows its training data. RAG *retrieves* relevant text from your documents at question time and puts it in the prompt, so the model answers from evidence. Benefits: up-to-date/private knowledge, fewer hallucinations, and the ability to cite sources. Entry point: `src/pipeline.py` (`ingest`, `ask`, `verify`).

## 2. PDF ingestion – `src/ingestion/pdf_loader.py`
PyMuPDF reads text per page so we can keep the **1-based page number**. Pages with < 20 characters are skipped (blank/image pages) but numbering stays original. If *no* page has text the PDF is probably scanned → `EmptyDocumentError`. The document ID is a SHA-256 hash of the bytes, so uploading the same file twice is detected.

## 3. Chunking – `src/processing/chunker.py`
LLMs and embedding models work best on passages, not whole documents. We use a sliding window (`chunk_size`, `overlap`). Overlap stops a sentence that straddles a boundary from being lost. We prefer cutting at `". "` or spaces. Chunking is **per page**, so every chunk maps to exactly one page. Trade-off: big chunks = more context but fuzzier matching; small chunks = precise but may lose context.

## 4. Embeddings – `src/embeddings/embedder.py`
An embedding model maps text to a vector so that similar *meaning* ⇒ nearby vectors. We use `all-MiniLM-L6-v2` (384 dims, fast, runs on CPU). Vectors are normalised, so cosine similarity = dot product. The same model must embed both chunks and questions.

## 5. Vector database – `src/vectorstore/chroma_store.py`
ChromaDB stores `(id, vector, text, metadata)` and does approximate nearest-neighbour search (HNSW) with cosine distance (`similarity = 1 − distance`). `PersistentClient` saves to disk. Chunk IDs are deterministic (`<doc_id>_p<page>_c<i>`), so we can ask Chroma which IDs exist and **skip re-embedding** – ingestion is idempotent. Metadata filters (`doc_id $in [...]`) implement per-document search.

## 6. Retrieval – `src/retrieval/retriever.py`
Embed the question → top-k nearest chunks. Each chunk gets a `ref` number 1..k. Top-k is the key recall/noise knob: too low misses evidence, too high adds distractors (hurts context precision, costs tokens).

## 7. Grounded generation – `src/generation/`
The prompt lists passages as `[1] (doc, page N) text...` and instructs: use only these, cite with `[n]`, and output a fixed sentence if the answer isn't present. Low temperature (0.1) reduces creativity. Passages are labelled as untrusted data to reduce **prompt injection** from document text. `llm_client.py` adds retries with backoff for 429/5xx and turns SDK errors into friendly `LLMError`/`MissingAPIKeyError`.

## 8. Citations – `src/citations/citation_builder.py`
Citations are **not** written by the model; the model only emits numbers. We map each number to the retrieved chunk we actually passed in, so document/page/passage come from metadata, not from LLM memory. A number that doesn't exist is dropped and surfaced as a warning ("fabricated citation guard").

## 9. Claim-level verification – `claim_splitter.py`, `verifier.py`
Sentences are treated as claims (a heuristic splitter that keeps decimals like 28.4 intact). Each claim is paired with the passages it cites (or all retrieved passages, flagged *uncited*). One Gemini call returns a JSON verdict per claim: Supported / Partially Supported / Unsupported / Contradicted + explanation. It's an NLI-style *entailment* check. Robustness: JSON mode, fence stripping, label normalisation, and "Unverified" for anything unparseable. **Limitation to mention:** an LLM judge is an estimate.

## 10. Hallucination types (good interview vocabulary)
*Intrinsic* – contradicts the source (→ Contradicted). *Extrinsic* – adds info not in the source (→ Unsupported). *Partial* – overstates/generalises (→ Partially Supported). Verification targets all three.

## 11. RAG evaluation with RAGAS – `src/evaluation/`
Two sides: **retrieval** and **generation**.
- **Context Precision** (retrieval): are the useful chunks ranked high? needs a reference answer.
- **Context Recall** (retrieval): is everything needed to produce the reference present in the retrieved chunks?
- **Faithfulness** (generation): fraction of answer statements supported by the context (≈ hallucination rate).
- **Answer Relevancy** (generation): does the answer address the question? (generates questions from the answer and compares embeddings.)

`ragas_runner.py` runs the *real* pipeline on every dataset question, builds `(user_input, retrieved_contexts, response, reference)` rows and calls `ragas.evaluate`. Failed rows become NaN → reported, never replaced by fake numbers. Interview caveats: LLM-as-judge bias, same model generating and judging, small dataset ⇒ noisy scores.

## 12. Engineering choices worth explaining
- **Dependency injection** in `VeriRAGPipeline` (embedder/store/LLM can be swapped) → tests use fakes, so no API key needed.
- **Lazy imports/loading** of heavy libraries → fast startup.
- **Typed dataclasses** in `models.py` define the data contract between stages.
- **Custom exceptions** → UI shows friendly messages instead of stack traces.
- **Config in `.env`**, never in code.

## 13. Ideas to extend (and discuss)
Re-ranking with a cross-encoder, hybrid BM25+dense retrieval, query rewriting, OCR, streaming, better claim splitting with an LLM, using a different judge model.
