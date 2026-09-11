"""Drive the dashboard in a real browser and capture it.

Kept in the repo because a UI-heavy deliverable should be verified by rendering
it, not by reading the CSS. Also surfaces console errors, which are otherwise
invisible from the server side.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077"
OUT = Path(__file__).parent / "shots"
OUT.mkdir(exist_ok=True)


def main() -> int:
    errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        page.goto(BASE, wait_until="networkidle")
        page.screenshot(path=OUT / "01-initial.png", full_page=True)

        page.fill("#count", "16")
        page.click("#run-btn")

        # Grid skeleton appears as soon as paraphrases land.
        page.wait_for_selector("#grid-panel:not(.hidden)", timeout=60_000)
        page.wait_for_timeout(400)
        page.screenshot(path=OUT / "02-streaming.png", full_page=True)

        # Run finishes when the scores panels reveal.
        page.wait_for_selector("#h2h-panel:not(.hidden)", timeout=120_000)
        page.wait_for_timeout(600)
        page.screenshot(path=OUT / "03-complete.png", full_page=True)

        # Grid close-up.
        page.locator("#grid-panel").screenshot(path=OUT / "04-grid.png")

        # Evidence drawer on the first non-pending cell.
        cell = page.locator("button.cell:not(.s-pending)").first
        cell.click()
        page.wait_for_selector(".drawer.open", timeout=10_000)
        page.wait_for_timeout(400)
        page.screenshot(path=OUT / "05-drawer.png")

        # Ungrouped ordering.
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)
        page.uncheck("#group-by-intent")
        page.wait_for_timeout(400)
        page.locator("#grid-panel").screenshot(path=OUT / "06-grid-ungrouped.png")

        # --- assertions on rendered state -----------------------------
        report = page.evaluate(
            """() => {
              const cells = [...document.querySelectorAll('button.cell')];
              const states = {};
              cells.forEach(c => {
                const s = [...c.classList].find(x => x.startsWith('s-'));
                states[s] = (states[s] || 0) + 1;
              });
              const glyphless = cells.filter(c => !c.textContent.trim()).length;
              return {
                cells: cells.length,
                states,
                glyphless,
                headline: document.getElementById('headline').textContent.trim(),
                aitc: document.getElementById('fig-aitc').textContent.trim(),
                consensusSegments: document.querySelectorAll('#composition > span:not(.empty)').length,
                legendItems: document.querySelectorAll('#legend .item').length,
                h2hBars: document.querySelectorAll('.h2hbar').length,
                colStats: [...document.querySelectorAll('td.colstat .t')].map(e => e.textContent),
                badges: [...document.querySelectorAll('td.colstat .badge')].map(e => e.textContent),
                notices: [...document.querySelectorAll('.notice b')].map(e => e.textContent),
                scrollW: document.documentElement.scrollWidth,
                clientW: document.documentElement.clientWidth,
              };
            }"""
        )
        browser.close()

    print("\n--- rendered state ---")
    for k, v in report.items():
        print(f"  {k}: {v}")

    print("\n--- console ---")
    print("  clean" if not errors else "\n".join(f"  {e}" for e in errors))

    problems = []
    if report["cells"] == 0:
        problems.append("no cells rendered")
    if report["glyphless"]:
        problems.append(f"{report['glyphless']} cells have no text glyph (colour-only encoding)")
    if report["legendItems"] != 4:
        problems.append(f"legend has {report['legendItems']} items, expected 4")
    if report["scrollW"] > report["clientW"] + 2:
        problems.append(f"horizontal overflow: {report['scrollW']} > {report['clientW']}")
    if not report["colStats"]:
        problems.append("no per-column Trust stats rendered")
    if errors:
        problems.append(f"{len(errors)} console error(s)")

    print("\n--- verdict ---")
    if problems:
        for p_ in problems:
            print(f"  FAIL: {p_}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
