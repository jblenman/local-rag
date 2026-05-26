#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query a markdown corpus's hybrid (vector + BM25) index.

Pipeline per query:

    user query
        ├─> [optional] HyDE: have an LLM draft a hypothetical answer,
        │   use THAT as the embedding target
        │
        ├─> vector lane: embed -> sqlite-vec KNN -> top-N by cosine
        │
        ├─> lexical lane: FTS5 MATCH -> top-N by BM25
        │
        ├─> fuse the two lanes with Reciprocal Rank Fusion (RRF)
        │
        ├─> [optional] rerank top-M with an LLM-as-judge
        │
        └─> return top-K

Flags:
    --hybrid / --no-hybrid     toggle hybrid (on by default)
    --hyde                     enable HyDE query expansion
    --rerank                   enable LLM rerank of top-M
    --rerank-model MODEL       chat model used by --hyde / --rerank
    -v                         DEBUG logs (per-lane timing, per-candidate scores)
    --json                     machine-readable output for tool integration

Examples:
    python kb_search.py "hivemind escalation flag"
    python kb_search.py "speed up inference on windows" --hyde --rerank
    python kb_search.py "oauth callback" -k 3 --json
"""

# =============================================================================
# Imports
# =============================================================================

import argparse
import json
import logging
import math
import os
import re
import sqlite3
import struct
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

try:
    import sqlite_vec
except ImportError:
    sys.exit("sqlite-vec not installed. Run: pip install sqlite-vec")

# On Windows, Python's stdout defaults to cp1252 which can't encode em dashes
# or most markdown punctuation. reconfigure() is Python 3.7+ only, hence the
# try/except instead of a version check.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# =============================================================================
# Config
# =============================================================================

# Index location is env-var-driven for portability. Default: .index/ next to
# the script.
SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR") or SCRIPT_DIR / ".index")
DB_PATH = INDEX_DIR / "kb.db"
OLLAMA_URL = os.environ.get("RAG_OLLAMA_URL", "http://localhost:11434")

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = "qwen2.5-coder:7b"   # used for HyDE and rerank; fast + tool-friendly

# RRF constant. 60 is the value from the original paper (Cormack et al. 2009).
# Lower = favors top-ranked items more aggressively; higher = flattens the curve.
RRF_K = 60

# How many candidates each lane over-fetches before fusion. Bigger = more recall,
# slower. 30 is plenty for a KB this size; 50-100 for enterprise scale.
CANDIDATE_POOL = 30

log = logging.getLogger("kb_search")


def setup_logging(verbose: bool) -> None:
    """Stderr-only logger so --json stdout stays clean for tool callers."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger("kb_search")
    root.handlers = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


# =============================================================================
# Vector math helpers (same as kb_index.py — kept local for script independence)
# =============================================================================
#
# MATH RECAP
# ----------
# We store unit vectors. For two unit vectors u, v:
#   ‖u - v‖² = 2·(1 - u·v) = 2·(1 - cos θ)
# sqlite-vec returns L2 distance d = ‖u - v‖, so:
#   cos θ = 1 - d²/2         (range -1..1, but with unit vectors from the
#                             same embedder, typically 0..1)
# =============================================================================

def normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return vec if norm == 0.0 else [x / norm for x in vec]


def serialize_vec(values: List[float]) -> bytes:
    """Pack a float list into N little-endian float32s, matching the storage
    format sqlite-vec expects."""
    return struct.pack("%df" % len(values), *values)


def l2_to_cosine(d: float) -> float:
    """L2 distance between unit vectors -> cosine similarity, clamped to [0, 1]."""
    sim = 1.0 - (d * d) / 2.0
    return max(0.0, min(1.0, sim))


# =============================================================================
# Ollama HTTP helpers
# =============================================================================

def ollama_post(path: str, payload: dict, timeout: int = 120) -> dict:
    """POST JSON to an Ollama endpoint, return the parsed response."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        sys.exit("Ollama unreachable at %s (%s)." % (OLLAMA_URL, e))


def embed(text: str, model: str) -> List[float]:
    """Embed a single string. Normalizes the output vector."""
    t0 = time.perf_counter()
    data = ollama_post("/api/embed", {"model": model, "input": [text]}, timeout=60)
    ms = (time.perf_counter() - t0) * 1000
    # Ollama returns either "embeddings" (list-of-lists for batch) or "embedding"
    # (single list). Handle both.
    vectors = data.get("embeddings") or [data.get("embedding")]
    log.debug("embed chars=%d dim=%d in %.0fms", len(text), len(vectors[0]) if vectors else 0, ms)
    return normalize(vectors[0])


def chat_generate(prompt: str, model: str, max_tokens: int = 512) -> str:
    """Non-streaming generate call. Returns the full text response as a string.

    `stream=False` tells Ollama to buffer the full response and send it as one
    JSON object. With stream=True you'd get NDJSON chunks — easier UX, more code.
    `options.num_predict` is Ollama's per-request max token cap.
    """
    t0 = time.perf_counter()
    data = ollama_post("/api/generate", {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.3},
    }, timeout=180)
    ms = (time.perf_counter() - t0) * 1000
    log.debug("chat %s in %.0fms (prompt chars=%d)", model, ms, len(prompt))
    return data.get("response", "").strip()


# =============================================================================
# DB access
# =============================================================================

def open_db() -> sqlite3.Connection:
    """Open kb.db with sqlite-vec loaded. Fails if the index doesn't exist."""
    if not DB_PATH.exists():
        sys.exit("No index at %s. Run kb_index.py first." % DB_PATH)
    con = sqlite3.connect(str(DB_PATH))
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


# =============================================================================
# Retrieval lanes
# =============================================================================
# We run vector and BM25 independently, each producing a ranked list. A chunk
# may appear in one, both, or neither. Fusion happens after (see rrf_fuse).
# =============================================================================

def vector_search(con: sqlite3.Connection, query_vec: List[float], n: int) -> List[Dict]:
    """KNN over the vec0 virtual table.

    The sqlite-vec query shape:
        SELECT ... FROM chunks_vec
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance
    MATCH triggers the KNN algorithm; `k = ?` constrains how many neighbors to
    return. ORDER BY distance sorts nearest-first.
    """
    t0 = time.perf_counter()
    rows = con.execute(
        """
        SELECT c.id, c.file_path, c.headers, c.line_start, c.line_end, c.body, v.distance
        FROM chunks_vec v
        JOIN chunks c ON c.rowid = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (serialize_vec(query_vec), n),
    ).fetchall()
    ms = (time.perf_counter() - t0) * 1000
    log.debug("vector lane: %d rows in %.1fms", len(rows), ms)

    results = []
    for r in rows:
        results.append({
            "id":         r[0],
            "file":       r[1],
            "headers":    r[2],
            "line_start": r[3],
            "line_end":   r[4],
            "body":       r[5],
            "vec_distance":  r[6],
            "vec_similarity": l2_to_cosine(r[6]),   # see MATH RECAP above
        })
    return results


def fts5_escape(query: str) -> str:
    """Make a user query safe-ish for FTS5's query mini-language.

    FTS5 treats certain characters as operators (AND/OR/NOT, NEAR, *, ", etc.).
    A colon is also significant. Simplest fix: drop punctuation, split into
    tokens, wrap each in double quotes so FTS5 treats them as literal terms.

    e.g. "oauth callback redirect" -> '"oauth" "callback" "redirect"'
    All terms are implicitly AND'd — FTS5 returns docs matching *all* terms.
    For more recall you could join with OR, but precision drops.
    """
    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return '""'   # FTS5 needs something; empty query would error
    # Double-quote each token so "tool-shim" or "don't" don't get parsed as operators.
    return " ".join('"%s"' % t for t in tokens)


def bm25_search(con: sqlite3.Connection, query: str, n: int) -> List[Dict]:
    """FTS5 MATCH with BM25 scoring.

    BM25 MATH (the one-minute version):
      Given a query Q = {q1, q2, ...} and document D:
          BM25(Q, D) = Σᵢ IDF(qᵢ) · ( f(qᵢ, D) · (k1 + 1) )
                                    / ( f(qᵢ, D) + k1·(1 - b + b·|D|/avgdl) )
      where f(qᵢ, D) is the term frequency in D, IDF is inverse document
      frequency across the corpus, k1 (~1.2) tunes saturation, b (~0.75) tunes
      length normalization, avgdl is the mean doc length.

      Intuition: rare terms matter more (IDF), repeated terms saturate (k1),
      and long docs get penalized (b). Classic, boring, very strong baseline.

    SQLite FTS5 uses a sign-flipped bm25() so that lower = better — this is
    so `ORDER BY bm25(...)` naturally surfaces best matches first.
    """
    match = fts5_escape(query)
    t0 = time.perf_counter()
    rows = con.execute(
        """
        SELECT c.id, c.file_path, c.headers, c.line_start, c.line_end, c.body,
               bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.rowid = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY bm25(chunks_fts)
        LIMIT ?
        """,
        (match, n),
    ).fetchall()
    ms = (time.perf_counter() - t0) * 1000
    log.debug("bm25 lane (%s): %d rows in %.1fms", match, len(rows), ms)

    return [{
        "id":         r[0],
        "file":       r[1],
        "headers":    r[2],
        "line_start": r[3],
        "line_end":   r[4],
        "body":       r[5],
        "bm25_score": r[6],
    } for r in rows]


# =============================================================================
# Fusion: Reciprocal Rank Fusion (RRF)
# =============================================================================
#
# THE MATH
# --------
# Given m ranked lists L₁, L₂, ..., Lₘ, define the RRF score for a document d:
#     rrf(d) = Σᵢ 1 / (k + rankᵢ(d))
# where rankᵢ(d) is d's position (1-based) in list Lᵢ, or ∞ if d isn't in Lᵢ
# (that term just contributes 0). k is a smoothing constant; 60 is standard.
#
# WHY IT'S USED INSTEAD OF JUST ADDING SCORES
# -------------------------------------------
# Vector similarity (0..1) and BM25 scores (unbounded, often -2..-30 with the
# FTS5 sign convention) live on wildly different scales. Adding them directly
# would let whichever retriever has the bigger scale dominate. RRF throws away
# the scores and keeps only the RANKS, which are already calibrated (1, 2, 3...).
#
# The k term smooths out the difference between rank 1 and rank 2, rank 2 and
# rank 3, etc. Tiny k -> the #1 item dominates. Huge k -> everyone ties. k=60
# is a nice middle.
# =============================================================================

def rrf_fuse(lanes: List[List[Dict]], k: int = RRF_K) -> List[Dict]:
    """Merge ranked result lists via RRF. Returns a single ranked list.

    `defaultdict(float)` auto-initializes missing keys to 0.0 on first access,
    so accumulating into `scores[doc_id]` works even on first sight of a doc.
    """
    scores: Dict[int, float] = defaultdict(float)
    rows: Dict[int, Dict] = {}   # doc_id -> the full chunk row (metadata we'll return)

    for lane in lanes:
        for rank, row in enumerate(lane, start=1):
            doc_id = row["id"]
            # RRF core formula: the earlier in a lane, the bigger the contribution.
            scores[doc_id] += 1.0 / (k + rank)
            # setdefault preserves the first-seen row (which carries
            # vec_similarity; bm25 copies arrive second and won't overwrite).
            rows.setdefault(doc_id, row)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    out = []
    for doc_id, rrf_score in ranked:
        row = rows[doc_id]
        row = {**row, "rrf_score": rrf_score}
        out.append(row)
    return out


# =============================================================================
# HyDE: Hypothetical Document Embedding
# =============================================================================
#
# WHY: queries and answers live in different regions of embedding space.
# "How do I reset my password?" and "To reset your password, go to..." use
# very different words. A query-to-chunk cosine distance is often worse than
# an answer-to-chunk distance. HyDE exploits this by:
#   1. Asking an LLM to write a hypothetical answer.
#   2. Embedding that answer.
#   3. Using THAT embedding to search.
#
# Paper: Gao et al. 2022 "Precise Zero-Shot Dense Retrieval without Relevance Labels"
# Costs: one extra chat round-trip. Fast models make this cheap (~1s on phi4).
# =============================================================================

def hyde_expand(query: str, model: str) -> str:
    """Have a chat model write a hypothetical answer; return the text."""
    prompt = (
        "Write a short, factual hypothetical answer to the following question. "
        "Two or three sentences, no preamble, no disclaimers, no 'I don't know'. "
        "If you don't know, invent a plausible-sounding answer — this is for a "
        "retrieval augmentation pipeline, not user-facing output.\n\n"
        "Question: %s" % query
    )
    answer = chat_generate(prompt, model, max_tokens=200)
    log.debug("hyde: %s", answer.replace("\n", " ")[:200])
    return answer


# =============================================================================
# Reranker: LLM-as-judge
# =============================================================================
#
# WHY: embedding similarity is approximate. Even after hybrid fusion, the top
# 10 might include chunks that look similar but don't actually answer the
# query. A reranker looks at each (query, chunk) pair as a *pair* and scores
# them — this is a much stronger signal than independent vector similarity.
#
# Classic rerankers are cross-encoders (bge-reranker-v2-m3 etc.), but they
# need special framework support. The LLM-as-judge pattern is a simpler
# workaround: ask a chat model to rate relevance 0-10. Slower but uses the
# fleet you already have.
# =============================================================================

# Regex to pull the first number out of an LLM response. Models sometimes
# respond "7" and sometimes "I'd rate this a 7 out of 10" — either works.
SCORE_RE = re.compile(r"\d+(?:\.\d+)?")


def rerank_llm(query: str, candidates: List[Dict], model: str) -> List[Dict]:
    """Score each candidate (query, chunk) pair via a chat model; resort."""
    log.info("rerank: scoring %d candidates with %s ...", len(candidates), model)
    rerank_start = time.perf_counter()

    for cand in candidates:
        # Truncate the chunk so slow models don't choke. 1500 chars ≈ 400 tokens.
        # Keep the breadcrumb because it's crucial context.
        chunk_preview = cand["body"]
        if len(chunk_preview) > 1500:
            chunk_preview = chunk_preview[:1500] + "..."

        prompt = (
            "Rate how well the following passage answers the query.\n"
            "Scale: 0 = completely irrelevant, 10 = directly answers the query.\n"
            "Respond with ONLY a number 0-10. No explanation.\n\n"
            "Query: %s\n\n"
            "Passage (from %s):\n%s\n\n"
            "Score:" % (query, cand["headers"] or cand["file"], chunk_preview)
        )
        resp = chat_generate(prompt, model, max_tokens=8)

        # Extract first number, clamp to [0, 10]. Fall back to 5 (neutral) on parse failure.
        match = SCORE_RE.search(resp)
        score = 5.0
        if match:
            try:
                score = max(0.0, min(10.0, float(match.group(0))))
            except ValueError:
                pass
        cand["rerank_score"] = score
        log.debug("rerank  %.1f  %s:%d  (raw=%r)",
                  score, cand["file"], cand["line_start"], resp[:40])

    # Sort by rerank score descending. Stable sort preserves RRF order on ties.
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    log.info("rerank: done in %.1fs", time.perf_counter() - rerank_start)
    return candidates


# =============================================================================
# Top-level search orchestrator
# =============================================================================

def search(
    query: str,
    k: int,
    embed_model: str,
    hybrid: bool,
    use_hyde: bool,
    use_rerank: bool,
    chat_model: str,
) -> List[Dict]:
    """Run the full pipeline. Returns top-k result dicts.

    Optional args (hyde, rerank) are cheap when off. Hybrid is on by default;
    disabling drops back to pure vector.
    """
    con = open_db()
    total_chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    log.debug("index holds %d chunks", total_chunks)

    # Step 1: optionally expand the query via HyDE.
    embed_target = query
    if use_hyde:
        log.info("hyde: expanding query via %s ...", chat_model)
        expanded = hyde_expand(query, chat_model)
        # Include the original query in the embed target too — otherwise rare
        # proper nouns that the LLM didn't repeat can be lost.
        embed_target = query + "\n\n" + expanded

    # Step 2: embed the query (or HyDE text).
    query_vec = embed(embed_target, embed_model)

    # Step 3: run retrieval lanes.
    vec_results = vector_search(con, query_vec, CANDIDATE_POOL)
    if hybrid:
        bm25_results = bm25_search(con, query, CANDIDATE_POOL)
        fused = rrf_fuse([vec_results, bm25_results])
    else:
        # Vector-only: just use the vec lane as-is; give each a fake RRF score
        # so the downstream code doesn't need to branch.
        fused = [
            {**r, "rrf_score": 1.0 / (RRF_K + i)}
            for i, r in enumerate(vec_results, start=1)
        ]

    # Step 4: optionally rerank top candidates.
    # Rerank happens on more than k so there's something to reorder, but less
    # than the full candidate pool so we don't pay for obviously-irrelevant ones.
    if use_rerank:
        top_m = min(len(fused), max(k * 3, 10))
        reranked = rerank_llm(query, fused[:top_m], chat_model)
        # Keep any leftovers after the reranked slice, in original RRF order.
        fused = reranked + fused[top_m:]

    log.debug("returning top %d of %d fused candidates", min(k, len(fused)), len(fused))
    for i, r in enumerate(fused[:k], 1):
        log.debug("  rank %d  rrf=%.4f  %s:%d  (vec_sim=%.3f bm25=%.2f rerank=%s)",
                  i, r.get("rrf_score", 0),
                  r["file"], r["line_start"],
                  r.get("vec_similarity", float("nan")),
                  r.get("bm25_score", float("nan")),
                  ("%.1f" % r["rerank_score"]) if "rerank_score" in r else "-")

    return fused[:k]


# =============================================================================
# Output formatting
# =============================================================================

def print_pretty(results: List[Dict], snippet_chars: int) -> None:
    """Human-readable output to stdout."""
    for r in results:
        header_line = "=== %s:%d-%d" % (r["file"], r["line_start"], r["line_end"])

        # Compose a score string that shows whichever signals we have.
        signals = []
        if "rerank_score" in r:
            signals.append("rerank %.1f" % r["rerank_score"])
        if "rrf_score" in r:
            signals.append("rrf %.4f" % r["rrf_score"])
        if "vec_similarity" in r:
            signals.append("sim %.3f" % r["vec_similarity"])
        if "bm25_score" in r:
            signals.append("bm25 %.2f" % r["bm25_score"])
        if signals:
            header_line += "  (" + ", ".join(signals) + ")"
        print(header_line)

        if r["headers"]:
            print("    " + r["headers"])
        print()

        body = r["body"]
        if snippet_chars and len(body) > snippet_chars:
            body = body[:snippet_chars] + "..."
        print(body)
        print()


def print_json(results: List[Dict], snippet_chars: int) -> None:
    """JSON output to stdout, safe for piping into tools."""
    out = []
    for r in results:
        body = r["body"]
        if snippet_chars and len(body) > snippet_chars:
            body = body[:snippet_chars] + "..."
        # Copy with a replaced `body` field so we don't mutate the live object.
        out.append(dict(r, body=body))
    print(json.dumps(out, indent=2))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", help="Natural-language query")
    ap.add_argument("-k", type=int, default=5, help="Final results to return (default 5)")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--rerank-model", default=DEFAULT_CHAT_MODEL,
                    help="Chat model for HyDE / rerank (default %s)" % DEFAULT_CHAT_MODEL)
    ap.add_argument("--no-hybrid", action="store_true", help="Disable BM25 lane (vector-only)")
    ap.add_argument("--hyde", action="store_true", help="Expand query via a hypothetical answer")
    ap.add_argument("--rerank", action="store_true", help="LLM-as-judge reranker over top candidates")
    ap.add_argument("--snippet", type=int, default=0, help="Truncate body to N chars (0 = full)")
    ap.add_argument("--json", action="store_true", help="Emit JSON, not pretty text")
    ap.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logs")
    args = ap.parse_args()

    setup_logging(args.verbose)
    run_start = time.perf_counter()
    log.info("query=%r  k=%d  hybrid=%s  hyde=%s  rerank=%s  ollama=%s",
             args.query, args.k, not args.no_hybrid, args.hyde, args.rerank, OLLAMA_URL)

    results = search(
        query=args.query,
        k=args.k,
        embed_model=args.embed_model,
        hybrid=not args.no_hybrid,
        use_hyde=args.hyde,
        use_rerank=args.rerank,
        chat_model=args.rerank_model,
    )
    total_ms = (time.perf_counter() - run_start) * 1000
    log.info("returned %d results in %.0fms total", len(results), total_ms)

    if args.json:
        print_json(results, args.snippet)
    else:
        print_pretty(results, args.snippet)


if __name__ == "__main__":
    main()
