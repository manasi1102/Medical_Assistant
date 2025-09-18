# Medical_Assistant
Healthcare Clinical Assistant (RAG + LLM) 

## Goal: 
A retrieval-augmented generation (RAG) prototype that helps clinicians quickly access trusted medical guidance (e.g., sepsis protocol, appendicitis, alopecia areata, TBI, leg fractures) using the Medical Manual (PDF) as the knowledge source.
What’s inside: LLM-only baselines, prompt-engineered runs (5 combos), RAG baseline, RAG tuning (5 combos), and a light evaluation framework for groundedness & relevance.

## ⚠️ Medical disclaimer
This project is for research/education. It is not medical advice. Always consult qualified clinicians and primary sources.

## ✨ Features

- LLM-only QA (baseline) and prompt engineering (5 prompt/param combos)

- RAG pipeline with Chroma DB, PyMuPDF loader, sentence-transformer embeddings, and strict page-level citations ([Page X])

- RAG tuning (5 combos): similarity vs. MMR, different k, chunk sizes, and embeddings (MiniLM vs. BGE-small)

- Fast evaluation (per question, single LLM call):

- Groundedness: Fully / Partially / Not grounded

- Relevance: Highly / Somewhat / Irrelevant

- Exports: HTML report, JSON logs, and PDF conversions


## 🧠 Questions covered (Business Problem)

- Protocol for managing sepsis in ICU

- Appendicitis symptoms & whether medically curable; if not, surgical procedure

- Treatments & causes for sudden patchy hair loss (localized bald spots)

- Treatments for traumatic brain injury (TBI)

- Precautions/treatment steps for a leg fracture on a hike (care & recovery)

## ⚙️ Environment & Setup (Colab-friendly)

- Recommended runtime: Google Colab GPU (T4)

- Install once at the top of the notebook:

pip install "llama-cpp-python[cuda]" huggingface_hub \
            langchain langchain-community chromadb \
            sentence-transformers pymupdf tiktoken pandas matplotlib nbconvert


## Models & tools

- LLM (llama.cpp): TheBloke/Mistral-7B-Instruct-v0.2-GGUF (e.g., mistral-7b-instruct-v0.2.Q4_K_M.gguf)

- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (baseline), BAAI/bge-small-en-v1.5 (alt)

- Vector DB: Chroma

- Loader: PyMuPDFLoader (LangChain community)

- Splitter: RecursiveCharacterTextSplitter (e.g., 1500/200 baseline)

🚀 How to Run (Step-by-step)

- Runtime check

  - Set runtime to GPU in Colab (T4 if available).

  - nvidia-smi should show a GPU.

- LLM-only (baseline)

  - Load Mistral GGUF via llama.cpp (n_gpu_layers=-1, n_ctx=4096).

  - Use a simple response(query, ...) function.

  - Run the 5 questions one at a time (to control cost).

  - Saved automatically into results/clinical_rag_results.json.

- LLM with Prompt Engineering (5 combos)

  - Define a cautious clinical system prompt (short, structured, avoids dosages, ends with safety line).

  - Run 5 prompt/param combos across the same 5 questions (again one at a time).

  - Saved to the same results JSON.

- Data for RAG

  - Load data/merck_manual.pdf with PyMuPDF.

  - Split (chunk_size=1500, overlap=200 initially).

  - Embed with MiniLM; persist vectors to Chroma.

  - Create retrievers: similarity (k=3, k=5) and MMR (k=4, λ=0.5).

- RAG QA

  - Strict RAG prompt: Use ONLY provided Context; cite pages as [Page X]; otherwise say “source not in context”; end with Safety: line; ≤120 words.

  - Baseline: similarity k=3.

  - Tuning (5 combos): broader k, MMR, smaller chunks (e.g., 900/150), and BGE-small embedding + MMR.

  - Run each question one at a time; answers are saved.

- Evaluation (Groundedness & Relevance)

  - Define evaluator prompts (rubric).

  - Use single-pass evaluator (both labels in one call) with k=1–2 context.

  - Evaluate LLM-only vs. RAG Baseline for all 5 questions (fast; meets rubric).

  - (Optional) Include PE/RAG tuning by toggling sections; cache prevents rework.

  - Results collected in clinical_rag_eval_cache.json.

- Observations & Insights

  - Add a concise observations cell after each method (LLM-only, PE, RAG baseline, RAG tuning).

  - Summarize groundedness/relevance trends.


## 📊 Results Snapshot (typical pattern)

- LLM-only: Often reads plausible but Unclear/Unclear in evaluation (no source ties, no citations).

- RAG Baseline (sim k=3): Frequently Fully grounded and Highly relevant (with [Page X] cites).

- RAG tuning:

  - Broader k (4–5) improves coverage for protocol queries (sepsis/TBI).

  - MMR (k=4) reduces redundancy but can drift; keep strong prompt constraints.

  - Smaller chunks (900/150) add procedural detail but may introduce noise; balance with 1200–1500/200.

  - BGE-small helps some differentials (e.g., alopecia) vs. MiniLM.


## 📝 Actionable Insights & Recommendations

- RAG > LLM-only for clinical reliability. Page-anchored citations directly improve trust and auditability.

- Operational baseline: similarity retriever (k=3–4), chunk 1200–1500 / 200, temp ≤ 0.3.

- Safety & scope: always add Safety line; avoid specific dosages; if context is thin, say “insufficient context”.

- Governance: log Q/A, retrieved pages, and eval labels for SME review; re-index manuals on a schedule.

- Pilot scope: start with sepsis/TBI/fracture workflows; measure time-to-answer, groundedness rates, and clinician satisfaction.

- Scale-out: add additional manuals, build “playbooks” per specialty, and enforce evaluation thresholds before release.

🧪 Repro tips (fast & cheap)

- Run one question at a time in each section.

- Keep evaluator k=1–2 and max_tokens≈64–72.

- Use the evaluation cache so re-runs are instant.


## 🙏 Acknowledgments

- Hugging Face community models: TheBloke Mistral-7B-Instruct (GGUF), sentence-transformers / BAAI/bge-small

- llama.cpp, LangChain, ChromaDB, PyMuPDF
