# Build Notes

This codebase was built collaboratively with Claude Code as the pair. Claude drove the keyboard; I navigated — scoped, made the architectural calls (the four retrieval modes, the normalization-as-storage trick that lets sqlite-vec's L2 metric produce cosine-correct rankings, hybrid lanes fused with Reciprocal Rank Fusion, HyDE stacked with the reranker against the *original* query, incremental upsert by chunk hash), reviewed every change, and corrected when something drifted off course. I wrote essentially none of the code by hand.

The original implementation was developed across earlier Claude Code sessions over the months leading up to publication. This repo is the extracted, sanitized version:

- Corpus and index locations generalized to environment variables (`RAG_CORPUS_ROOT`, `RAG_INDEX_DIR`) so the scripts are corpus-agnostic.
- Ollama URL likewise generalized (`RAG_OLLAMA_URL`) so fleet machines can host the embedder.
- A small self-referential demo corpus included so a fresh clone has something to index and search out of the box.
- README rewritten to lead with the design tradeoffs (the four retrieval modes and what they catch) rather than setup steps.

The HyDE failure-mode write-up in [`corpus/rag-local-stack.md`](../corpus/rag-local-stack.md) ("retrieval-confirmed hallucination") is the section I'm proudest of. It's not the implementation — it's the named understanding of what goes wrong when this technique fails and what to do about it. That kind of analysis is what I want my AI-pairing workflow to produce more of.

I'm including this note because pretending to have hand-coded six retrieval modes would misrepresent the workflow — and the workflow is part of the point. This is what's possible when an experienced engineer treats Claude Code as a regular collaborator on personal projects: more time on architectural judgment and failure-mode analysis, less time on typing.
