# local-rag

A small, local-first RAG library over a markdown corpus. Ollama embeddings, SQLite + sqlite-vec, hybrid BM25/vector retrieval with Reciprocal Rank Fusion, optional HyDE query expansion, optional LLM-as-judge reranker. Stdlib Python plus one pip dependency (`sqlite-vec`); small enough to read in one sitting.

The repo is short on features and long on documentation — the design choices and the failure modes that motivated them are written out in the corpus itself, which is also what you'll search to test the system.

## What you get

- **Hybrid retrieval** — vector (cosine via sqlite-vec L2 over normalized embeddings) and lexical (BM25 via FTS5), fused with Reciprocal Rank Fusion. Each lane has different failure modes; fusion is cheap insurance.
- **HyDE query expansion** (`--hyde`) — closes synonym gaps by embedding a hypothetical answer. Includes the documented failure mode (retrieval-confirmed hallucination) and the mitigation pattern (always pair with hybrid; ideally also rerank).
- **LLM-as-judge reranker** (`--rerank`) — slower, higher precision over the top candidates.
- **Incremental indexing** — chunk-level SHA-1 hashes; only changed chunks re-embed. Renames and edits both trigger; unchanged content is skipped.
- **Machine-readable output** (`--json`) — designed to be called as a tool from any agent that can shell out.

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Pull the embedding model (or change DEFAULT_MODEL to one you already have)
ollama pull nomic-embed-text

# 3. Index the demo corpus (markdown files under corpus/)
python kb_index.py

# 4. Query
python kb_search.py "what is reciprocal rank fusion"
python kb_search.py "ranking equivalence between L2 and cosine"
python kb_search.py "speed up retrieval on synonym-dependent queries" --hyde --rerank
```

The demo corpus is meaningfully self-referential — both files document the system itself, so a query against them returns real explanations of what just happened.

## Pointing at your own corpus

Environment variables override defaults:

```bash
export RAG_CORPUS_ROOT=~/notes              # directory of *.md files
export RAG_INDEX_DIR=/tmp/my-rag             # where to put the index DB
export RAG_OLLAMA_URL=http://192.168.1.10:11434   # if Ollama runs elsewhere on your LAN
```

Markdown files are chunked on header boundaries, prepended with their parent-header breadcrumb (the single biggest retrieval-quality lever — see `corpus/rag-local-stack.md`), and indexed incrementally.

A larger example corpus to try this against: [`jblenman/knowledge`](https://github.com/jblenman/knowledge) — a public knowledge garden with notes on languages, tools, and patterns. Clone it and point `RAG_CORPUS_ROOT` at the checkout.

## Design notes

The most useful background is in the corpus itself:

- [`corpus/vector-math-primer.md`](corpus/vector-math-primer.md) — the linear algebra (normalization, cosine, the L2 identity for unit vectors, the conversion `cos θ = 1 − d²/2`). A worked numeric example at the bottom traces the same numbers through all four formulations.
- [`corpus/rag-local-stack.md`](corpus/rag-local-stack.md) — design tradeoffs, sqlite-vec gotchas, observed retrieval quality across query styles, the HyDE failure-mode write-up, when this pattern stops working.

You're meant to run a query against those docs and watch the system rank them — the corpus is both demo content and reference reading.

## Agent integration

The pipeline is designed to be called by an agent. `--json` emits structured output safe for piping:

```bash
python kb_search.py "your query" -k 3 --json --snippet 400
```

[`hivemind`](https://github.com/jblenman/hivemind) wires `kb_search.py` as a `search_kb` tool alongside `read_file` / `search_files` / `run_command`. The agent picks per query whether to use vector-only, hybrid, `--hyde`, `--rerank`, or `--hyde --rerank`. That decision-making belongs to the LLM — that's what makes this *agentic* RAG rather than plain RAG.

## How this was built

This codebase was built collaboratively with Claude Code as the pair. See [`docs/BUILD-NOTES.md`](docs/BUILD-NOTES.md).

## License

MIT — see [LICENSE](LICENSE).
