# Vector Math Primer (for RAG and embeddings)

The minimum linear algebra needed to read embedding/RAG code with confidence.
Every operation in `kb_search.py` traces back to this material.

## 1. A vector is a list of numbers

```
v = [3, 4]                       2D — a point on a plane
v = [3, 4, 0]                    3D — a point in space
v = [0.12, -0.04, ..., 0.31]     768-dim — what nomic-embed-text returns
```

Geometrically: an arrow from the origin to that point. Two things matter —
**length** and **direction**.

## 2. Length (magnitude / L2 norm)

Pythagoras, generalized:

```
||v|| = √(v₁² + v₂² + ... + vₙ²)
```

For `[3, 4]`: `√(9 + 16) = √25 = 5`.

Called the **L2 norm** because the exponents are squared. Written `‖v‖`.

**Normalizing** divides every component by the length so the arrow keeps its
direction but has length 1:

```
v̂ = v / ||v||
[3, 4] / 5 = [0.6, 0.8]          length 1, same direction
```

Result is a **unit vector**. Every embedding in the RAG store is unit-length.

## 3. The dot product

```
a · b = a₁·b₁ + a₂·b₂ + ... + aₙ·bₙ
```

Multiply componentwise, sum the results. One scalar out.

The geometric identity is the key fact:

```
a · b = ||a|| · ||b|| · cos(θ)
```

where θ is the angle between the arrows. The dot product measures
**how much two vectors point in the same direction**, scaled by their lengths.

| a and b point... | angle | cos θ | a · b                          |
|------------------|-------|-------|--------------------------------|
| same direction   | 0°    | +1    | big positive (max = ‖a‖·‖b‖)   |
| perpendicular    | 90°   | 0     | **zero**                       |
| opposite         | 180°  | −1    | big negative                   |

## 4. Cosine similarity

Solve the identity for cos θ:

```
cos θ = (a · b) / (||a|| · ||b||)
```

Range −1 to +1. Measures pure *direction* alignment; ignores magnitude.

**Why direction, not length?** An embedding's magnitude depends on text
length and model quirks — semantically, noise. Direction encodes meaning.
Two GPU docs should align in direction whether one is a paragraph or a page.

## 5. The normalization shortcut (the RAG trick)

Pre-normalize every vector to length 1. Then the denominator becomes 1·1 = 1, so:

```
cos θ = a · b        (when ||a|| = ||b|| = 1)
```

**Cosine similarity collapses to a plain dot product.** No division.

That's why `kb_index.py`'s `normalize()` exists. Every stored vector is unit
length; every query is normalized before search; similarity is now the
cheapest possible operation.

## 6. L2 distance and the cosine identity

sqlite-vec's default ranking is **L2 distance**, not cosine:

```
||a - b|| = √(Σ (aᵢ - bᵢ)²)
```

Lower = more similar.

For **unit vectors only**, an algebraic identity links the two:

```
||a - b||² = ||a||² + ||b||² − 2(a·b)
           = 1 + 1 − 2 cos θ
           = 2(1 − cos θ)
```

Two consequences worth remembering:

1. **Monotonicity** — as cos θ rises, d² falls. **Sorting by L2 distance
   gives the same ranking as sorting by cosine.** sqlite-vec's default
   metric is silently doing what we want.
2. **Conversion** — `cos θ = 1 − d²/2`. `kb_search.py:l2_to_cosine()` is
   literally that one line.

Without normalization, L2 and cosine produce *different* rankings, and the
default would quietly return the wrong order. This is why `normalize()` is
load-bearing, not cosmetic.

---

## Worked example — end to end with tiny numbers

Imagine a 3-dim "embedding space" where the dimensions, by convention, are
roughly `[GPU-ness, Python-ness, networking-ness]`. (Real embeddings have
no human-readable axes — this is a teaching fiction.)

**Three documents and a query:**

```
A = [0.9, 0.1, 0.0]       a strongly GPU-focused doc
B = [0.2, 0.8, 0.1]       a strongly Python-focused doc
C = [0.4, 0.5, 0.1]       a mixed-topic doc
q = [0.8, 0.2, 0.0]       a GPU-leaning question
```

### Step 1 — Cosine similarity the long way

For each doc, compute dot product, divide by the product of lengths.

**Doc A:**

```
a · q = 0.9·0.8 + 0.1·0.2 + 0.0·0.0 = 0.72 + 0.02 + 0 = 0.74
||A|| = √(0.81 + 0.01 + 0)   = √0.82  ≈ 0.9055
||q|| = √(0.64 + 0.04 + 0)   = √0.68  ≈ 0.8246

cos(q,A) = 0.74 / (0.9055 · 0.8246) = 0.74 / 0.7467 ≈ 0.991
```

**Doc B:**

```
b · q = 0.2·0.8 + 0.8·0.2 + 0.1·0.0 = 0.16 + 0.16 + 0 = 0.32
||B|| = √(0.04 + 0.64 + 0.01) = √0.69 ≈ 0.8307

cos(q,B) = 0.32 / (0.8307 · 0.8246) = 0.32 / 0.6850 ≈ 0.467
```

**Doc C:**

```
c · q = 0.4·0.8 + 0.5·0.2 + 0.1·0.0 = 0.32 + 0.10 + 0 = 0.42
||C|| = √(0.16 + 0.25 + 0.01) = √0.42 ≈ 0.6481

cos(q,C) = 0.42 / (0.6481 · 0.8246) = 0.42 / 0.5345 ≈ 0.786
```

**Ranking:**  A (0.991) > C (0.786) > B (0.467). Matches intuition — query
is GPU-leaning, A is the strongest GPU doc, B is the most Python.

### Step 2 — Normalize everything once

Divide each vector by its length:

```
Â = [0.993, 0.110, 0.000]
B̂ = [0.241, 0.963, 0.120]
Ĉ = [0.617, 0.772, 0.154]
q̂ = [0.970, 0.242, 0.000]
```

Quick sanity: `√(0.993² + 0.110² + 0²) = √(0.986 + 0.012) = √0.998 ≈ 1` ✓

### Step 3 — Cosine via plain dot product

Now the dot product *is* the cosine:

```
q̂ · Â = 0.970·0.993 + 0.242·0.110 + 0·0     = 0.963 + 0.027 + 0 ≈ 0.990
q̂ · B̂ = 0.970·0.241 + 0.242·0.963 + 0·0.120 = 0.234 + 0.233 + 0 ≈ 0.467
q̂ · Ĉ = 0.970·0.617 + 0.242·0.772 + 0·0.154 = 0.598 + 0.187 + 0 ≈ 0.785
```

Same numbers as Step 1 (modulo rounding) with no division.

### Step 4 — L2 distance, and the identity

sqlite-vec computes squared L2 distance directly:

```
||q̂ − Â||² = (0.970−0.993)² + (0.242−0.110)² + (0−0)²
           = 0.000529 + 0.017424 + 0
           ≈ 0.0179

||q̂ − B̂||² = (0.970−0.241)² + (0.242−0.963)² + (0−0.120)²
           = 0.531 + 0.520 + 0.0144
           ≈ 1.066

||q̂ − Ĉ||² = (0.970−0.617)² + (0.242−0.772)² + (0−0.154)²
           = 0.125 + 0.281 + 0.0237
           ≈ 0.430
```

Cross-check the identity `‖a − b‖² = 2(1 − cos θ)`:

```
A:  2(1 − 0.990) = 0.020   ← matches 0.0179 (rounding)
B:  2(1 − 0.467) = 1.066   ← matches exactly
C:  2(1 − 0.785) = 0.430   ← matches exactly
```

### Step 5 — Ranking by L2 vs cosine

```
By L2 distance (lower = better):     A (0.018) < C (0.430) < B (1.066)
By cosine similarity (higher = better): A (0.990) > C (0.785) > B (0.467)
```

**Identical ranking.** That's the property that lets sqlite-vec use its
default L2 metric and still produce cosine-correct results.

### Step 6 — Convert distance back to similarity

`kb_search.py:l2_to_cosine()` applies `1 − d²/2`:

```
A:  1 − 0.0179/2 = 0.991
B:  1 − 1.066/2  = 0.467
C:  1 − 0.430/2  = 0.785
```

Round-trip back to the cosine values from Step 1.

---

## Mental model to keep

- **Vector** = list of numbers = arrow with length + direction
- **Dot product** = componentwise multiply + sum = "how aligned are these arrows"
- **Normalize** = divide by length → unit vector, direction only
- **Cosine similarity** = dot product of unit vectors, range −1 to +1
- **L2 distance on unit vectors** = monotonic with cosine, ranks identically
- **Conversion** = `cos θ = 1 − d²/2`

If you remember just the geometric identity `a · b = ‖a‖‖b‖ cos θ`, the rest
falls out of algebra.

## See also

- [rag-local-stack.md](rag-local-stack.md) — the RAG pattern that uses all of this
- `kb_index.py:normalize()` — the load-bearing one-liner
- `kb_search.py:l2_to_cosine()` — the conversion helper
