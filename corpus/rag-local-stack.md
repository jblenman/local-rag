# Local RAG Stack (Ollama + sqlite-vec + Python)

A minimal, fully-local agentic-RAG stack over a small-to-medium markdown corpus. The reference implementation in this repo is corpus-agnostic — point `RAG_CORPUS_ROOT` at any directory of markdown files.

## Shape

```
Markdown corpus
      │
      ▼  (chunker)
Header-scoped chunks, breadcrumb prepended
      │
      ▼  (Ollama /api/embed)
Normalized float32 vectors
      │
      ▼  (sqlite-vec vec0 virtual table)
Single-file vector DB (~15 KB per 100 chunks @ 768-dim)
      │
      ▼  (query = embed + MATCH + k=N)
Top-K chunks with citations
```

Three pieces — an embedding model, a vector store, and ~300 lines of Python glue per script.
No cloud services, no API keys, no running daemons beyond Ollama.

## Embedding model choices (all via `ollama pull`)

| Model              | Dim  | Size   | Notes                                         |
|--------------------|------|--------|-----------------------------------------------|
| `nomic-embed-text` | 768  | 274 MB | Default. Fast, solid quality, wide adoption.  |
| `mxbai-embed-large`| 1024 | 335 MB | ~25% slower, visibly better on ambiguous queries. |
| `bge-m3`           | 1024 | 1.2 GB | Multilingual. Pick only if you need it.        |
| `all-minilm`       | 384  | 45 MB  | Tiny, CPU-friendly, lower quality. Triage tier. |

Swap via the `--model` flag (or change `DEFAULT_MODEL` at the top of each script). Any change requires a full `--rebuild` since dimension and embedding space are not interchangeable.

## Throughput observed

On a typical desktop GPU (~6 GB usable VRAM via Ollama):

- `nomic-embed-text`, batch 16: ~32 chunks/s end-to-end, ~45 chunks/s on the model itself
- Query embedding round-trip: ~40–60 ms
- sqlite-vec KNN over ~270 chunks: <5 ms

Embedding models are small, so they cohabit VRAM with a chat model without
evicting it — useful when the same Ollama instance serves both RAG and a
generator.

## sqlite-vec gotchas

- Requires `sqlite3.enable_load_extension(True)` before `sqlite_vec.load(con)`.
  Default Windows Python build supports this; some hardened Linux builds don't.
- KNN syntax is `WHERE embedding MATCH ? AND k = ?` — the `k` is a virtual
  column constraint, not an ORDER BY limit.
- `vec0` virtual tables use `rowid` as the join key back to your metadata table.
  Hold onto `cursor.lastrowid` when inserting into the metadata table and pass
  it as the rowid for the vec insert.
- Default distance is L2. If you pre-normalize vectors (`v / ||v||`), L2
  ordering matches cosine ordering exactly, and similarity is
  `1 - d^2 / 2` (range 0..1). This avoids needing a custom distance function.

## Chunking that actually helps retrieval

- Split on markdown headers (`##`/`###`), not by a fixed token window.
- Prepend a breadcrumb of parent headers to each chunk before embedding:
  `Python Patterns > Concurrency > asyncio\n\n<body>`.
  This is Anthropic's "Contextual Retrieval" finding in miniature — gives the
  embedder the same context a human reader would have.
- Set a soft max on chunk size (~3500 chars) and split large sections on
  paragraph boundaries when exceeded.
- Skip tiny chunks (<50 chars) — header-only sections add noise.
- Hash each chunk (SHA-1 of breadcrumb + body) and make the hash the upsert
  key. Incremental re-index becomes free — only new/changed chunks re-embed.

## Retrieval quality profile (small KB, ~17 files / ~270 chunks)

Reality check after building one against a personal markdown corpus:

| Query style             | Vector-only                     | After hybrid + HyDE             |
|-------------------------|----------------------------------|----------------------------------|
| Specific terminology    | Hits the right section reliably. | Same — already strong.           |
| Cross-topic fuzzy       | Surprisingly strong — catches connections a human index would miss. | Slight improvement.              |
| Synonym-dependent       | Miss-prone. "speed up inference" does not bridge to "GPU acceleration". | **Fixed.** HyDE bridges the vocabulary gap; surfaces `num_ctx`, `num_thread`, `OLLAMA_KEEP_ALIVE`, speculative decoding at rank 2. |

Three upgrades, applied in order of quality-per-effort. **All three shipped**
in `kb_search.py`:

1. **Hybrid search (BM25 + vector).** FTS5 alongside `vec0`; lanes fused via
   Reciprocal Rank Fusion (RRF, k=60). RRF uses *rank position* not raw scores,
   so it doesn't matter that BM25 and cosine live on different scales. Fixes
   most synonym misses without needing an LLM.
2. **HyDE** (`--hyde`). LLM drafts a hypothetical answer; embed *that* instead
   of the raw query. The draft uses the right vocabulary even if it's
   factually wrong — you're using it as a topic-relevant bag of words, not as
   an answer. Closes the "speed up" → "GPU" gap directly.
3. **LLM-as-judge reranker** (`--rerank`). Score top-20 candidates 0–10 with a
   chat model against the *original* query. Higher precision; slower (1–5s).

## HyDE — the failure mode worth naming

HyDE is a deliberate, controlled hallucination injection into retrieval. When
it fails, it fails badly: the drafting LLM hallucinates a confident-sounding
but wrong direction → HyDE retrieves docs that match the wrong vocabulary →
the final answer looks well-sourced while pointing the wrong way. Sometimes
called **retrieval-confirmed hallucination** — worse than plain hallucination
because the citations make it feel trustworthy. Especially bad for queries
about content the KB doesn't actually have; the draft fills the void with
plausible content, HyDE retrieves the closest neighbors to that void-filler,
high-confidence garbage out.

Mitigations that work:

- **Always pair with hybrid, never replace it.** BM25 can't hallucinate; it
  only matches words that exist in the corpus, anchoring results when HyDE
  wanders off.
- **Stack `--hyde --rerank`.** The reranker scores candidates against the
  *original* query, filtering out off-topic hits the bad draft pulled in.
- **Embed `query + "\n\n" + draft`** rather than the draft alone — keeps the
  real query pulling toward real content. (Optional; not enabled by default.)
- **Lower drafting temperature** (e.g. 0.2) hallucinates less.

The risk is low on a small, well-curated KB (not much room for the draft to
invent content the KB doesn't have). It scales badly on open-web corpora
with adversarial content.

## Windows console gotcha

Python's default stdout on Windows is cp1252, which cannot encode em dashes,
box-drawing characters, or most non-ASCII punctuation common in markdown.
Call `sys.stdout.reconfigure(encoding="utf-8")` at script start, or your
pretty-print path will crash the moment a retrieved chunk contains a `—`.

## Agent integration

The pattern is plug-and-play with any agent that can shell out:

- **Hivemind `hmc`** — [`hivemind`](https://github.com/jblenman/hivemind)
  wires this as a `search_kb` tool alongside `read_file` / `search_files` /
  `run_command`. The agent picks per query whether to use `hybrid`, `vector`,
  `hyde`, `rerank`, or `hyde+rerank`.
- **Claude Code skill / subagent** — wrap `kb_search.py --json` in a skill.
- **Raw LLM tool call** — same shape; embed-search is a single tool.

The *LLM* owns the retrieval decision: when to search, what to search for,
whether to re-query, whether to fall back to reading a file in full. That is
what makes it agentic RAG rather than plain RAG.

## When this pattern outgrows itself

- **More than ~10K chunks** — pure SQLite KNN still works but cold-cache latency
  creeps up; move to a purpose-built vector store (Qdrant local, pgvector).
- **Multi-user with row-level access control** — sqlite-vec doesn't filter
  before KNN, so permissioned retrieval means filtering after, which wastes
  candidates. Move to a store with pre-filter support.
- **Corpus changes faster than you can re-embed** — unlikely on a personal KB;
  real at document-heavy enterprise scale.

Until one of those trips, this stack is hard to beat on simplicity, inspectability,
cost ($0), and freshness (git commit → next query sees it).
