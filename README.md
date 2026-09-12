<div align="center">

  
  # TrustGraph
  
  **The AI Recommendation Stability Engine**
  
  [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
  
  <p align="center">
    Understand exactly how the world's top LLMs rank and recommend your brand. <br>
    TrustGraph cuts through AI hallucinations to give you hard, verifiable metrics on your AI reputation.
  </p>
</div>

---

## Overview

**TrustGraph** is a powerful analytics pipeline and real-time dashboard designed to stress-test AI recommendations. It fires paraphrased, brand-neutral queries across multiple live LLMs (OpenAI, Anthropic, Gemini) in parallel, analyses their responses, and extracts a definitive **Trust Score** and **Win Rate** for your brand against competitors.

Brands can no longer rely purely on SEO; they need to understand their **LLM Optimization (LLMO)**. TrustGraph gives you the verifiable data to do exactly that.

## Key Features

- **Multi-Model Interrogation:** Automatically queries OpenAI, Anthropic, and Gemini in parallel with temperature pinned to `0` to eliminate sampling noise.
- **Two-Pass Mention Extraction:** Uses deterministic boundary-anchored matching paired with LLM-powered JSON-schema verification to ensure we track *actual recommendations*, not just mentions.
- **Real-Time Streaming Dashboard:** A responsive, grid-based dashboard powered by Server-Sent Events (SSE) that visualizes AI consensus cell-by-cell in real time.
- **Resilient by Design:** Graceful degradation on API failures, automatic deterministic fallbacks, and content-addressed disk caching ensure runs never crash mid-flight.
- **Hard Metrics:** Computes actionable metrics including the **Trust Score**, **Recommendation Stability Index (RSI)**, and **Normalized Head-to-Head Win Rates**.

## How It Works (The Pipeline)

TrustGraph runs a fully automated, stateless extraction pipeline:

1. **Paraphrase Generation:** A utility LLM generates 15–20 diverse, brand-neutral query variations based on your target category.
2. **Parallel Firing:** Queries are fired concurrently at the entire roster of LLMs.
3. **Precision Extraction:** The system reads every response and determines if your brand (or a competitor) was explicitly recommended, name-dropped, or ignored.
4. **Scoring & Streaming:** Metrics are calculated and streamed instantly to the frontend dashboard.

---

## Quick Start

### 1. Installation

Requires Python 3.9+.

```bash
# Clone the repository
git clone https://github.com/yourusername/TrustGraph.git
cd TrustGraph

# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -e ".[all,dev]"
```

### 2. Configuration

Copy the example environment file and add your API keys. *Note: If you don't have keys, TrustGraph will gracefully degrade to a simulated provider for testing.*

```bash
cp .env.example .env
```

### 3. Run the Dashboard

Launch the FastAPI backend and frontend:

```bash
trustgraph serve
```
Open `http://127.0.0.1:8000` in your browser.

### 4. Run via CLI

You can also run measurements directly from the terminal:

```bash
trustgraph run \
  --entity Razorpay \
  --category "payment gateway for an Indian D2C startup" \
  --competitors "Stripe,PayU,Cashfree" \
  --count 18
```

## Testing

TrustGraph ships with a comprehensive test suite (87 tests) that runs entirely offline without requiring network access or API keys.

```bash
pytest -q
```

## Architecture & Engineering

TrustGraph isn't just a wrapper; it solves hard problems in LLM output evaluation:
- **Fuzzy Matching Fences:** Prevents false positives (e.g., "stripes" matching the brand "Stripe") using boundary-anchored squashed matching.
- **Cue Ownership:** Solves attribution in complex sentences (e.g., *"X is popular, but I prefer Y"* correctly credits the endorsement to Y).
- **Hallucination Guards:** An LLM-claimed recommendation is only accepted if the exact verbatim span actually exists in the raw output text.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
