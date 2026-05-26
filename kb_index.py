#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Index a markdown corpus into a local hybrid (vector + BM25) search DB.

Pipeline:
    *.md
        -> chunker (splits on markdown headers, prepends breadcrumb)
        -> embedder (Ollama /api/embed -> 768-dim float32 vector)
        -> sqlite-vec (vec0 virtual table; KNN over vectors)
        -> sqlite FTS5 (virtual table; BM25 over tokens)

The two virtual tables live side-by-side in .index/kb.db. At search time we
blend them with Reciprocal Rank Fusion (in kb_search.py).

Incremental by default. Each chunk's SHA-1 hash (over breadcrumb + body) is
the upsert key; only new/changed chunks re-embed, stale chunks get deleted.

Python 3.8 compatible. Stdlib-only except for sqlite-vec.
"""

import argparse
import hashlib
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
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import sqlite_vec
except ImportError:
    sys.exit("sqlite-vec not installed. Run: pip install sqlite-vec")


# =============================================================================
# Config — knobs you might want to tune
# =============================================================================

# The corpus and index locations are env-var-driven so this script is corpus-
# agnostic. Defaults assume the conventional layout: corpus/ and .index/ next
# to the script.
SCRIPT_DIR = Path(__file__).resolve().parent

# Where the markdown corpus to index lives. Override with RAG_CORPUS_ROOT.
KB_ROOT = Path(os.environ.get("RAG_CORPUS_ROOT") or SCRIPT_DIR / "corpus")

# Where the index DB lives. Override with RAG_INDEX_DIR.
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR") or SCRIPT_DIR / ".index")
DB_PATH = INDEX_DIR / "kb.db"

# Point at an Ollama instance running elsewhere on your LAN without editing
# the script:
#     RAG_OLLAMA_URL=http://192.168.1.10:11434 python kb_index.py
OLLAMA_URL = os.environ.get("RAG_OLLAMA_URL", "http://localhost:11434")

# Default embedding model. Swapping requires --rebuild because embeddings from
# different models live in different vector spaces (different dimension, and
# different meaning per axis even at the same dimension).
DEFAULT_MODEL = "nomic-embed-text"
EMBED_DIM = 768   # must match the model; nomic-embed-text is 768-dim

# Chunking soft caps. Too big = embedding loses focus; too small = fragmentation.
MAX_CHUNK_CHARS = 3500   # sections bigger than this get split on paragraph boundaries
MIN_CHUNK_CHARS = 50     # skip header-only stubs

# Regex matches markdown headers. Group 1 = the '#' chars (H1-H6), group 2 = title.
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")

log = logging.getLogger("kb_index")


# =============================================================================
# Logging setup
# =============================================================================

def setup_logging(verbose: bool) -> None:
    """Wire a stderr handler with a compact timestamped format.

    Logs go to stderr (not stdout) so if any caller pipes JSON out of a sister
    script, the logs don't corrupt the pipe. DEBUG flips on with --verbose.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger("kb_index")
    root.handlers = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


# =============================================================================
# Embedding + vector math
# =============================================================================
#
# MATH NOTE — why we normalize vectors before storing them
# ---------------------------------------------------------
# Cosine similarity between two vectors u, v:
#     cos(θ) = (u · v) / (‖u‖ · ‖v‖)
# where u·v = sum(uᵢ·vᵢ) is the dot product and ‖u‖ = √Σuᵢ² is the L2 norm.
#
# If we pre-normalize every vector so ‖u‖ = ‖v‖ = 1, the denominator drops out
# and cos(θ) = u·v. But sqlite-vec's vec0 table uses L2 (Euclidean) distance
# by default, not dot product. Here's the trick:
#
# For unit vectors,  ‖u - v‖² = ‖u‖² + ‖v‖² - 2·u·v = 2 - 2·cos(θ)
# so L2 distance²  =  2·(1 - cos(θ))
#
# L2 distance is a monotonic function of (1 - cos), so **sorting by L2 on
# normalized vectors produces the exact same order as sorting by cosine
# similarity**. That means we get cosine ranking for free without a custom
# distance op. We back-convert for display: sim = 1 - d²/2  (range 0..1).
# =============================================================================

def normalize(vec: List[float]) -> List[float]:
    """Return vec / ‖vec‖ (unit length). No-op for the zero vector."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def serialize_vec(values: List[float]) -> bytes:
    """Pack a list of floats into the byte format sqlite-vec stores natively.

    "%df" % n builds a format string like "768f" = 768 little-endian float32s.
    """
    return struct.pack("%df" % len(values), *values)


def embed(texts: List[str], model: str) -> List[List[float]]:
    """Round-trip a batch of texts through Ollama's /api/embed and normalize.

    Ollama accepts either {"input": "text"} or {"input": ["list", "of", "text"]}.
    We always send lists so the response shape is predictable.
    """
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL + "/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    # perf_counter is monotonic and high-res — safe for elapsed-time measurement
    # even if the wall clock jumps.
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        sys.exit("Ollama unreachable at %s (%s). Is it running?" % (OLLAMA_URL, e))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Ollama sometimes returns {"embeddings": [[...], [...]]} (batch shape) and
    # sometimes {"embedding": [...]} (single shape). Support both.
    vectors = data.get("embeddings") or [data.get("embedding")]

    total_chars = sum(len(t) for t in texts)
    log.debug("embed batch n=%d chars=%d dim=%d in %.0fms (%.0f ch/ms)",
              len(texts), total_chars, len(vectors[0]) if vectors else 0,
              elapsed_ms, total_chars / max(elapsed_ms, 0.1))

    # Normalize every vector so downstream L2 == cosine (see MATH NOTE above).
    return [normalize(v) for v in vectors]


# =============================================================================
# Chunking: markdown -> header-scoped chunks with breadcrumb
# =============================================================================

def split_by_paragraph(text: str, max_chars: int) -> List[str]:
    """Split text on blank lines, greedily packing paragraphs up to max_chars.

    Used when a single markdown section exceeds MAX_CHUNK_CHARS. We'd rather
    split on paragraph boundaries than mid-sentence.
    """
    paras = re.split(r"\n\s*\n", text)
    out: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for p in paras:
        # Flush if adding this paragraph would overflow; but only if buf isn't empty
        # (a single paragraph > max_chars still gets emitted whole — we don't split
        # inside paragraphs).
        if buf and buf_len + len(p) > max_chars:
            out.append("\n\n".join(buf))
            buf, buf_len = [], 0
        buf.append(p)
        buf_len += len(p) + 2   # +2 accounts for the "\n\n" separator we'll add
    if buf:
        out.append("\n\n".join(buf))
    return out


def chunk_markdown(text: str) -> List[Dict]:
    """Walk a markdown file line-by-line, emitting one chunk per header-scoped section.

    Each returned dict:
        body       -> the section text (no headers stripped from middle, just the parent chain)
        headers    -> "H1 > H2 > H3" breadcrumb
        line_start, line_end -> 1-based line span in the original file

    The chunker holds a `stack` of active headers. When a new header appears:
      - flush the current section to a chunk
      - pop any same-or-deeper headers off the stack (entering a sibling/shallower scope)
      - push the new header
    """
    lines = text.splitlines()
    stack: List[Tuple[int, str]] = []
    chunks: List[Dict] = []

    current_lines: List[str] = []
    current_headers: List[str] = []
    section_start = 1

    # flush() closes over the outer vars. Python closures capture by reference,
    # so mutating current_lines/current_headers from the outer scope is visible.
    def flush(end_line: int) -> None:
        body = "\n".join(current_lines).strip()
        if len(body) < MIN_CHUNK_CHARS:
            return
        breadcrumb = " > ".join(current_headers)
        # If the section fits in one chunk, emit as-is. Otherwise break on paragraphs.
        sub_sections = [body] if len(body) <= MAX_CHUNK_CHARS else split_by_paragraph(body, MAX_CHUNK_CHARS)
        cursor = section_start
        for sub in sub_sections:
            sub_len = len(sub.splitlines())
            chunks.append({
                "body": sub,
                "headers": breadcrumb,
                "line_start": cursor,
                # min() guards against an off-by-one if split estimation overshoots.
                "line_end": min(cursor + sub_len - 1, end_line),
            })
            cursor += sub_len + 1  # +1 for blank line separator between paragraphs

    for idx, line in enumerate(lines, start=1):
        m = HEADER_RE.match(line)
        if m:
            flush(idx - 1)
            level = len(m.group(1))         # number of '#' chars
            title = m.group(2).strip()

            # Pop headers that are as deep or deeper than this one. Example:
            # if we see an H2 while stack is [H1, H2, H3], we pop H2 and H3 so
            # the new H2 becomes a sibling of the popped H2, not a child.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))

            current_headers = [h[1] for h in stack]
            current_lines = []
            section_start = idx
        else:
            current_lines.append(line)

    # Flush the trailing section (everything after the last header).
    flush(len(lines))
    return chunks


def chunk_text_for_embedding(headers: str, body: str) -> str:
    """What actually gets embedded: breadcrumb + blank line + body.

    WHY this matters: the embedding model scores a chunk based on its full text.
    A paragraph that says "This must be approved by the Director" is meaningless
    in isolation but perfectly clear under the breadcrumb
    "Employee Handbook > Leave Policy > Bereavement". We give the model the
    same context a human reader would have. This is the single biggest quality
    lever in this whole pipeline.
    """
    return (headers + "\n\n" + body) if headers else body


def chunk_hash(headers: str, body: str) -> str:
    """SHA-1 of breadcrumb + body. Used as the incremental upsert key.

    Why hash both: if someone renames a header ("Setup" -> "Installation"),
    the breadcrumb changes, which should count as a changed chunk (re-embed)
    even if the body text is byte-identical.
    """
    return hashlib.sha1(chunk_text_for_embedding(headers, body).encode("utf-8")).hexdigest()


# =============================================================================
# DB schema
# =============================================================================
# Three tables living together in one SQLite file:
#
#   chunks        -> regular table, one row per chunk, holds the text + metadata
#   chunks_vec    -> sqlite-vec virtual table, holds ONLY embeddings, joined
#                    back to chunks via rowid. Enables vec0 KNN with MATCH.
#   chunks_fts    -> FTS5 virtual table, holds tokenized copies of body + headers
#                    for BM25 lexical search. Also joined by rowid.
#
# The two virtual tables are populated alongside `chunks` explicitly. We're not
# using triggers or FTS5's external-content mode — keeps the ingestion code
# linear and easy to follow.
# =============================================================================

def ensure_schema(con: sqlite3.Connection) -> None:
    """Idempotent: creates tables only if they don't already exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id         INTEGER PRIMARY KEY,
            file_path  TEXT NOT NULL,
            headers    TEXT NOT NULL,
            line_start INTEGER NOT NULL,
            line_end   INTEGER NOT NULL,
            body       TEXT NOT NULL,
            hash       TEXT NOT NULL UNIQUE
        )
    """)

    # vec0 = sqlite-vec's virtual table module. `embedding float[N]` is its
    # column-type syntax. Queries against it use `MATCH ? AND k = ?` for KNN.
    con.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[%d])" % EMBED_DIM
    )

    # FTS5 virtual table. Columns `body` and `headers` are tokenized and
    # BM25-indexed. `porter` stemmer handles English plurals/tenses (run/runs/
    # running); `unicode61` tokenizer normalizes case and punctuation.
    # BM25 = a TF-IDF variant that scores docs by term overlap with the query.
    con.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            body, headers,
            tokenize='porter unicode61'
        )
    """)

    # An index on file_path is only useful for admin queries ("show me chunks
    # from foo.md") but costs ~nothing so keep it.
    con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(file_path)")


def open_db() -> sqlite3.Connection:
    """Open the DB, enable extension loading, load sqlite-vec, return the connection.

    Extension loading is disabled by default for security. We turn it on just
    long enough to load sqlite-vec, then turn it back off.
    """
    INDEX_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


# =============================================================================
# Main — incremental ingestion
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama embedding model")
    ap.add_argument("--rebuild", action="store_true", help="Drop and rebuild the index")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Chunks per Ollama call. Higher = better throughput but more RAM per request.")
    ap.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logs")
    args = ap.parse_args()

    setup_logging(args.verbose)
    run_start = time.perf_counter()

    log.info("ollama=%s model=%s dim=%d db=%s",
             OLLAMA_URL, args.model, EMBED_DIM, DB_PATH)

    # --rebuild nukes the DB file. Only useful when changing embedding model/dim
    # or recovering from corruption.
    if args.rebuild and DB_PATH.exists():
        log.warning("--rebuild: dropping %s", DB_PATH)
        DB_PATH.unlink()

    con = open_db()
    ensure_schema(con)

    # Set comprehension builds a set of hashes already in the DB. We'll diff
    # this against what the chunker produces to find new/changed/stale chunks.
    existing_hashes = {row[0] for row in con.execute("SELECT hash FROM chunks")}
    log.debug("loaded %d existing chunk hashes from DB", len(existing_hashes))

    md_files = sorted(KB_ROOT.rglob("*.md"))
    log.info("discovered %d markdown files under %s", len(md_files), KB_ROOT)

    seen_hashes = set()       # every hash produced by this run
    new_chunks: List[Dict] = []   # subset that's actually new/changed
    total_chunks = 0

    # --- Phase 1: chunk every file, figure out what's new ---
    chunk_start = time.perf_counter()
    for path in md_files:
        # relative_to() strips the KB_ROOT prefix. as_posix() forces forward
        # slashes — useful on Windows so the DB stays portable between machines.
        rel = path.relative_to(KB_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        file_chunks = chunk_markdown(text)
        total_chunks += len(file_chunks)
        file_new = 0
        for c in file_chunks:
            # Decorate each chunk with its file path + content hash.
            c["file_path"] = rel
            c["hash"] = chunk_hash(c["headers"], c["body"])
            seen_hashes.add(c["hash"])
            # If this hash isn't in the DB, it's either brand new or a modified
            # existing chunk (rename / body edit). Either way, re-embed.
            if c["hash"] not in existing_hashes:
                new_chunks.append(c)
                file_new += 1
        log.debug("chunked %s -> %d chunks (%d new)", rel, len(file_chunks), file_new)
    log.info("chunked in %.0fms: %d total chunks, %d new/changed, %d unchanged",
             (time.perf_counter() - chunk_start) * 1000,
             total_chunks, len(new_chunks), total_chunks - len(new_chunks))

    # --- Phase 2: purge chunks that no longer exist ---
    # Set difference: hashes in DB that the chunker didn't produce this run.
    stale = existing_hashes - seen_hashes
    if stale:
        # Parametrized IN clause: ",".join("?" * N) builds "?,?,?" for N bindings.
        ids = [row[0] for row in con.execute(
            "SELECT id FROM chunks WHERE hash IN (%s)" % ",".join("?" * len(stale)),
            tuple(stale),
        )]
        # Delete from ALL THREE tables. The vec0 and FTS5 tables don't cascade.
        con.executemany("DELETE FROM chunks_vec WHERE rowid = ?", [(i,) for i in ids])
        con.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(i,) for i in ids])
        con.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in ids])
        log.info("deleted %d stale chunks", len(stale))

    # --- Phase 3: early exit if no new work ---
    if not new_chunks:
        con.commit()
        total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        log.info("no new chunks; %d total in index; done in %.2fs",
                 total, time.perf_counter() - run_start)
        return

    # --- Phase 4: embed + insert new chunks in batches ---
    log.info("embedding %d chunks in batches of %d ...", len(new_chunks), args.batch_size)
    embed_start = time.perf_counter()
    # Ceiling division: (n + batch - 1) // batch.
    batch_count = (len(new_chunks) + args.batch_size - 1) // args.batch_size

    for i in range(0, len(new_chunks), args.batch_size):
        batch = new_chunks[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1

        # One HTTP round trip per batch. Ollama embeds them all in parallel internally.
        vectors = embed(
            [chunk_text_for_embedding(c["headers"], c["body"]) for c in batch],
            args.model,
        )

        for c, vec in zip(batch, vectors):
            # INSERT returning rowid via cursor.lastrowid. We use the same rowid
            # for the matching row in chunks_vec and chunks_fts so JOIN-by-rowid works.
            cur = con.execute(
                "INSERT INTO chunks (file_path, headers, line_start, line_end, body, hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (c["file_path"], c["headers"], c["line_start"], c["line_end"], c["body"], c["hash"]),
            )
            rowid = cur.lastrowid
            con.execute(
                "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                (rowid, serialize_vec(vec)),
            )
            # FTS5 gets the raw text (body + headers). FTS5 tokenizes/stems internally.
            con.execute(
                "INSERT INTO chunks_fts (rowid, body, headers) VALUES (?, ?, ?)",
                (rowid, c["body"], c["headers"]),
            )
        log.info("batch %d/%d  (%d/%d chunks done)", batch_num, batch_count,
                 min(i + args.batch_size, len(new_chunks)), len(new_chunks))

    con.commit()
    embed_elapsed = time.perf_counter() - embed_start
    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    log.info("embedded %d chunks in %.1fs (%.1f chunks/s)",
             len(new_chunks), embed_elapsed, len(new_chunks) / max(embed_elapsed, 0.01))
    log.info("done: %d total chunks in DB, wall-clock %.2fs",
             total, time.perf_counter() - run_start)


if __name__ == "__main__":
    main()
