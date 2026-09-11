# TrustGraph — Recommendation Stability Tool

Implementation of **Section 6** of *The AI Trust Graph: A HITS-Based Authority Model
for Trust Propagation in AI-Mediated Recommendation and Commerce*, with the
frontend specified in **Section 7**.

> Type in your brand, your category, and 2–3 named competitors. The tool fires
> paraphrased queries at multiple live LLMs, checks who each model actually
> recommends, and shows a Trust Score, a stability rating, and a head-to-head win
> rate against each competitor — per model and combined.

The paper's theory (bipartite source→entity corroboration graph, HITS hub/authority
propagation, source-authority seeding) explains *why* a cross-model,
paraphrase-averaged score is the right thing to measure. None of that machinery
runs here, deliberately — §6.3 says so explicitly. What runs is the empirically
measurable part, against live model behaviour.

---

## What it measures

| Quantity | Definition | Paper |
|---|---|---|
| **Trust(e, c, m)** | responses recommending *e* ÷ total paraphrases, per model | Eq. (3) |
| **RSI(e, c, m)** | `1 − Var/0.25` — inverse variance of the recommendation indicator across semantically equivalent phrasings | Eq. (5) |
| **AITC(e, c)** | weight-normalised cross-model mean of Trust | §3.5 |
| **Cross-model r** | mean pairwise Pearson correlation of per-phrasing outcomes | §3.5, stronger form |
| **Head-to-head** | normalised win rate vs each named competitor, over contested phrasings only | §7.4 |
| **Four-state grid** | Win / Loss / Both / Neither per (phrasing × model) | Table 2 |

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate           # Windows;  source .venv/bin/activate on POSIX
pip install -e ".[all,dev]"

cp .env.example .env             # add at least one provider key
trustgraph serve                 # http://127.0.0.1:8000
```

No keys? It still runs. The roster degrades to a deterministic **simulated
provider** and every surface is flagged as fixtures rather than measurements.
That exists so the pipeline is testable in CI and a demo never dies on an
expired key — not to pad the numbers.

```bash
trustgraph doctor                # what keys/columns/utility model are live
trustgraph run --entity Razorpay \
  --category "payment gateway for an Indian D2C startup" \
  --competitors "Stripe,PayU,Cashfree" --count 18
```

---

## The pipeline

```
entity + category + competitors
        │
        ├─ 1  paraphrase generation      one utility LLM call → 15–20 brand-neutral
        │                                queries with framing tags; cached per category
        │
        ├─ 2  multi-model firing         every paraphrase → every roster model,
        │                                in parallel, temperature pinned to 0
        │
        ├─ 3  mention extraction         two independent passes, then a merge
        │       ├─ deterministic         boundary-anchored matcher + cue scoring
        │       └─ structured re-check   one JSON-schema LLM call per response
        │
        ├─ 4  scoring                    Trust, RSI, AITC, consensus, head-to-head
        │
        └─ 5  stream                     SSE; the grid fills cell by cell
```

Stateless left of the dashboard, exactly as Figure 2 specifies. The only durable
artefact is the response cache, which is an optimisation and can be deleted at
any time without changing behaviour.

### Layout

```
src/trustgraph/
  domain.py        types + the four-state derivation (Table 2)
  matching.py      entity-name matching over prose  — precision-critical
  extract.py       recommendation classification + two-pass merge
  score.py         Eq. (3), Eq. (5), AITC, consensus, head-to-head
  paraphrase.py    query generation + deterministic fallback
  pipeline.py      orchestration, concurrency, retry, event stream
  config.py        settings + roster construction/degradation
  cache.py         content-addressed disk cache
  api.py           FastAPI app, SSE run endpoint
  cli.py           serve / run / doctor
  providers/       gemini · anthropic · openai · simulated
  web/             index.html · app.css · app.js
tests/             87 tests, no network required
```

---

## Engineering decisions worth knowing about

These are the places where the paper is ambiguous, where the obvious
implementation is wrong, or where a number could mislead. All are visible in the
UI, not buried here.

### 1. Table 2's four states overlap as written — resolved on the recommendation axis

"Win" is defined on the recommendation axis ("no named competitor is
[recommended]"); "Both" is defined on the presence axis ("both
mentioned/recommended"). A response that recommends the target while
name-dropping a rival satisfies both rows, and a response that merely mentions
the target satisfies none.

`recommended` is made the primary axis, since that is the axis Eq. (3) counts —
so the picture and the number describe the same event:

```
BOTH    ← target recommended AND ≥1 competitor recommended
WIN     ← target recommended AND no competitor recommended
LOSS    ← ≥1 competitor recommended AND target not recommended
NEITHER ← no tracked entity recommended at all
```

Total, mutually exclusive, and `NEITHER` lands exactly on its gloss in Table 2
("the model answered off-axis"). The presence nuance is demoted, not discarded:
every cell keeps `target_mentioned_only` / `competitors_mentioned_only`, rendered
as a corner tick and spelled out in the evidence drawer.

### 2. RSI has no direction — so the badge always supplies one

Eq. (5) is a pure consistency measure. An entity recommended in **0%** of
phrasings scores `RSI = 1.0`, identically to one recommended in 100%. That is
correct and misleading, so the badge never appears without a direction
("consistently recommended" vs "consistently absent").

### 3. AITC is normalised

§3.5 writes `AITC = Σ w·Trust`. Taken literally, three models at Trust = 1.0
gives 3.0 — unbounded and not comparable to Trust, which is a probability. This
implementation divides by `Σ w`, making AITC a weighted mean on [0,1].
`aitc_unweighted` is also reported. Weights are a stated market-share prior, not
a measurement.

### 4. Temperature is pinned to 0 wherever the provider allows it

RSI is defined as variance across *phrasings*. Leaving sampling temperature at
provider defaults would fold sampling noise into that variance and change what
the number means. Some current flagship models have removed sampling controls
entirely; those columns are flagged, and their RSI is a lower bound on stability.

### 5. Fuzzy name matching is fenced structurally, not by threshold

Naive fuzzy matching is unusable on brand names: `"stripes"` scores ~92 against
`"Stripe"`, and one false positive silently manufactures a Loss that no
downstream check would catch. Three ordered stages, by how much each can go
wrong:

1. Boundary-anchored literal match (aliases included).
2. **Squashed** match — punctuation and whitespace removed from both sides, so
   `"Razor Pay"` finds `"Razorpay"`. Still exact, in a normalised space.
3. Bounded fuzzy, only for names with no hit yet, and **a prefix relationship is
   rejected outright** — so `"Stripe"` never matches `"stripes"` or `"striped"`
   at any ratio. Real typos are mid-token edits and survive the fence.

### 6. Recommendation cues are *owned*, not bagged

`"Razorpay is popular in India, though I would go with Stripe"` has one
recommendation cue and two vendors. Crediting the cue to both — which
sentence-level cue bagging does — turns a Loss into a Both. Each cue is
attributed to a single entity by grammatical direction: verb cues that govern an
object (`go with X`, `unlike X`) attach to the name that *follows*; predicate
cues (`X is the stronger choice`) attach to the name that *precedes*. Negations
resolve before the positive form they contain, so `"I wouldn't recommend X"`
never scores as an endorsement.

### 7. A hallucinated mention cannot create a cell

The LLM pass is authoritative on *recommendation* (a judgement call); the matcher
is authoritative on *presence* (a factual claim). An LLM-only presence claim is
honoured only when its verbatim evidence span genuinely occurs in the response —
otherwise a fabricated mention would invent a Win or a Loss out of nothing.
Agreement between the two passes is measured and displayed, not assumed.

### 8. A one-family roster is never presented as cross-model consensus

With only one vendor key configured, the roster widens to three models from that
vendor so the grid keeps a model axis — and the run states plainly that the
consensus figures are now *within-family* and do not support §3.5's
cross-family claim. Two families are the paper's stated demo fallback.

### 9. Failures degrade, isolate, then report — in that order

A failed re-check falls back to deterministic extraction for that cell. A failed
measurement call fails one cell, never the run. A failed paraphrase call falls
back to deterministic templates. Only a total roster failure ends the run. Cell,
cache, and failure counts are all shown.

---

## Frontend

Built to §7's spec, not to taste.

- **The hero is a grid, not a number.** Rows are paraphrases, columns are models,
  cells are four-state. Rendered as a real `<table>` because the data is
  literally tabular and a card layout would destroy the row/column structure
  that makes the pattern visible.
- **Rows group by framing** (toggleable). This is what turns §7.1's example — a
  band of losses across every price-framed phrasing while category-framed
  phrasings stay green — from a scatter into something you can see at a glance.
- **Every cell is clickable** and opens the response text that produced it, with
  the matched names highlighted using the byte offsets the classifier actually
  scored. The score is inspectable, not asserted.
- **Consensus is a single composition bar**, not per-model cards — because
  "unanimous across families" and "one model's favourite" are structurally
  different facts that average to the same number.
- **Head-to-head bars are normalised** over contested phrasings only.

### Palette accessibility

The four semantic colours are fixed by §7.5. Running them through a
CVD/contrast validator against the panel surface produced two binding
obligations, both discharged:

| Check | Result | Consequence |
|---|---|---|
| CVD separation (worst adjacent) | ΔE 8.3 deuteranopia — at the floor | Colour alone is not a legal encoding → **every cell carries a text glyph** (`WIN`/`LOSS`/`BOTH`/`—`) |
| Contrast of `neither` (#24322F) vs panel | 1.26:1 | Sub-3:1 fills oblige visible labelling → the real `<table>`, per-cell glyphs, and a visible hairline on that state |
| Normal-vision separation | ΔE 21.7 — pass | — |

The palette's lightness-band and chroma "failures" are the *categorical* checks;
this is a **status** palette, where "neither = muted slate" being recessive is
the design intent. A legend is always present, and `forced-colors` mode falls
back to borders plus the glyphs.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/run` | SSE stream: `run_started` → `paraphrases` → `cell`×N → `scores` → `run_complete` |
| `POST /api/run/sync` | Blocking; full `RunResult` JSON |
| `GET /api/health` | Configured providers, resolved roster, warnings |
| `GET /api/models` | **Live** model discovery per provider |
| `POST /api/cache/clear` | Drop the response cache |

`GET /docs` serves the generated OpenAPI UI.

`/api/models` exists because vendor model IDs drift, and a stale hardcoded
default is the likeliest reason a demo fails at the worst possible moment. Every
default is also overridable via `TRUSTGRAPH_ROSTER`.

---

## Cost and shape of a run

Default 18 paraphrases × 3 columns × 2 passes + 1 paraphrase call ≈ **109 calls**,
a few seconds with concurrency at 12. Turn off the re-check to halve it. Repeat
runs on the same inputs are served from cache and are free; `--fresh` (or the
"bypass cache" checkbox) forces live calls.

---

## Tests

```bash
pytest -q          # 87 tests, no network, no keys required
```

Covered: matcher precision fences (the `stripes`/`razorpays` false positives),
cue ownership, all four state transitions and the full state table, the
hallucinated-mention guard, RSI against its closed form and boundedness,
Pearson's undefined-vs-zero distinction, AITC normalisation, division-by-zero on
a fully-failed run, roster degradation for 0/1/2/3 providers, provider-outage
isolation, re-check and paraphrase fallbacks, cache hit/bypass, SSE event
ordering, and the HTTP surface.

---

## Limitations

Stated because the paper insists on it (§9), not as boilerplate.

- **The corroboration graph is latent, not observed.** Every claim about "the
  graph" is a claim about an inferred statistical structure. This tool measures
  the behavioural shadow — recommendation frequency and its variance — not the
  structure itself.
- **Non-stationarity is structural.** Every model update can reshape the
  topology. A score is dated the moment it is taken; there is no database built
  once.
- **Goodhart's Law is the deepest risk.** Once a metric like this is adopted by
  marketing teams it stops measuring authority and starts measuring performance
  against the measurement protocol — exactly what happened to PageRank.
- **Not measurable here at all:** why a model recommends something,
  training-data attribution, the pretrain/fine-tune/RLHF split, or any
  ground-truth internal "trust" representation (none exists).

The natural next step is §5's validation protocol — run this across many
categories, compute AITC independently, and correlate. `trustgraph run --json`
exists for exactly that.

---

## References

1. S. Brin, L. Page. *The Anatomy of a Large-Scale Hypertextual Web Search Engine.* 1998.
2. J. M. Kleinberg. *Authoritative Sources in a Hyperlinked Environment.* JACM 46(5), 1999.
