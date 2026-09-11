"""TrustGraph — the Recommendation Stability Tool of the AI Trust Graph paper.

Section 6 of the working paper scopes a self-contained pipeline: text input to
live dashboard in one request cycle, with no crawling, no graph database, and no
persistent state. This package is that pipeline.

    paraphrase -> fire at N models in parallel -> extract -> score -> stream

The theory (bipartite corroboration graph, HITS hub/authority propagation,
source-authority seeding) stays in the paper. What runs here is the empirically
measurable part: ``Trust(e, c, m)`` (Eq. 3), ``RSI(e, c, m)`` (Eq. 5), and
``AITC(e, c)`` (§3.5).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
