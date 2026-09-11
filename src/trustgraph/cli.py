"""Command line interface.

``trustgraph serve`` runs the dashboard; ``trustgraph run`` prints the same grid
as text, which is what makes the pipeline scriptable for the §5 validation
protocol (run it across many categories and correlate) without going through
the browser.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from pydantic import ValidationError

from .config import get_settings
from .domain import CellState, RunResult, RunSpec
from .pipeline import Pipeline

_GLYPH = {
    CellState.WIN: "WIN ",
    CellState.LOSS: "LOSS",
    CellState.BOTH: "BOTH",
    CellState.NEITHER: " -  ",
}


def _print_report(result: RunResult) -> None:
    spec = result.spec
    print()
    print(f"  {spec.entity}  ·  {spec.category}")
    print(f"  vs {', '.join(spec.competitors) or '(no competitors named)'}")
    print(f"  {'-' * 76}")

    for warning in result.meta.warnings:
        print(f"  ! {warning}")
    if result.meta.warnings:
        print(f"  {'-' * 76}")

    labels = {m.key: m.label for m in result.models}
    width = max((len(v) for v in labels.values()), default=10)
    width = max(width, 6)

    header = "  " + "framing".ljust(14) + "".join(
        labels[m.key][:width].ljust(width + 2) for m in result.models
    )
    print(header + "query")
    print("  " + "-" * (len(header) + 30))

    by_query: dict[int, dict[str, CellState]] = {}
    errors: dict[int, set[str]] = {}
    for cell in result.cells:
        by_query.setdefault(cell.query_index, {})[cell.model_key] = cell.state
        if cell.error:
            errors.setdefault(cell.query_index, set()).add(cell.model_key)

    for para in result.paraphrases:
        row = by_query.get(para.index, {})
        cells = ""
        for model in result.models:
            state = row.get(model.key)
            token = (
                "ERR " if model.key in errors.get(para.index, set())
                else _GLYPH.get(state, "?   ") if state
                else "...."
            )
            cells += token.ljust(width + 2)
        print("  " + (para.intent or "-")[:13].ljust(14) + cells + para.text[:52])

    print()
    for score in result.scores.per_model:
        counts = " ".join(f"{k.value}={v}" for k, v in score.counts.items())
        print(
            f"  {labels[score.model_key]:<{width}}  trust={score.trust:5.1%}  "
            f"rsi={score.rsi:.2f} [{score.stability}]  {counts}"
        )

    scores = result.scores
    print()
    print(f"  AITC (weighted)   {scores.aitc:.1%}")
    print(f"  AITC (unweighted) {scores.aitc_unweighted:.1%}")
    consensus = scores.consensus
    print(
        f"  Consensus         unanimous-win={consensus.unanimous_win} "
        f"split={consensus.split} unanimous-loss={consensus.unanimous_loss} "
        f"of {consensus.n}"
    )
    coefficient = scores.cross_model.coefficient
    print(
        "  Cross-model r     "
        + ("undefined" if coefficient is None else f"{coefficient:+.3f}")
        + f"   ({scores.cross_model.note})"
    )
    if scores.extraction_agreement is not None:
        print(f"  Extraction agree  {scores.extraction_agreement:.1%}")

    if scores.head_to_head:
        print()
        for record in scores.head_to_head:
            rate = "n/a" if record.win_rate is None else f"{record.win_rate:.1%}"
            print(
                f"  vs {record.competitor:<18} {rate:>7}  "
                f"(W{record.wins} T{record.ties} L{record.losses} "
                f"of {record.eligible} contested)"
            )

    print()
    print(f"  {scores.headline}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trustgraph",
        description="Recommendation Stability Tool — AI Trust Graph, Section 6.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    run = sub.add_parser("run", help="run a measurement and print the grid")
    run.add_argument("--entity", required=True)
    run.add_argument("--category", required=True)
    run.add_argument(
        "--competitors", default="", help="comma-separated competitor names"
    )
    run.add_argument("--count", type=int, default=None, help="paraphrase count")
    run.add_argument("--no-recheck", action="store_true")
    run.add_argument("--fresh", action="store_true", help="bypass the cache")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--json", action="store_true", help="emit the full payload")

    sub.add_parser("doctor", help="show provider configuration")

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "trustgraph.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
        return 0

    if args.command == "doctor":
        settings = get_settings()
        roster, warnings = settings.build_roster()
        print("\n  providers:")
        for provider in ("gemini", "anthropic", "openai"):
            mark = "yes" if (settings.api_key(provider) or "").strip() else "no"
            print(f"    {provider:<12} key: {mark}")
        print("\n  grid columns:")
        for model in roster:
            print(f"    {model.key:<32} weight={model.weight}")
        utility = settings.utility_model(roster)
        print(f"\n  utility model: {utility[0]}:{utility[1]}" if utility else "\n  utility model: none")
        for warning in warnings:
            print(f"\n  ! {warning}")
        print()
        return 0

    competitors = [c.strip() for c in args.competitors.split(",") if c.strip()]
    try:
        spec = RunSpec(
            entity=args.entity,
            category=args.category,
            competitors=competitors,
            paraphrase_count=args.count,
            llm_recheck=not args.no_recheck,
            seed=args.seed,
        )
    except ValidationError as exc:
        # A raw pydantic traceback is not a usable error message on a CLI.
        for error in exc.errors():
            field = ".".join(str(p) for p in error["loc"]) or "input"
            print(f"error: {field}: {error['msg']}", file=sys.stderr)
        return 2

    try:
        result = asyncio.run(Pipeline().collect(spec, fresh=args.fresh))
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        _print_report(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
