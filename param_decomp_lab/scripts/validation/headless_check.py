"""Smoke-test a self-contained HTML applet in headless Chromium — no display, no GPU.

For the canvas/JS plot applets in this folder (e.g. `build_addition_explorer`), the bugs
that don't show up in Python are JS-side: a missing `<script src>`, an undefined global, a
render that throws. This loads the page in real Chromium, fails loudly on any console error
or uncaught exception, optionally clicks through a list of selectors (tabs, buttons), and
screenshots each step so you can eyeball the render. Exits non-zero if any JS error fired.

Standalone: imports only `playwright` + `fire`, so it runs under the isolated toolchain
venv built by `headless_setup.sh` (NOT the project venv — Chromium's system-lib needs are
kept out of `make install-dev`). It auto-wires that venv's browser + shared-lib paths, so
you just point it at an HTML file.

Usage (run setup once, then):
    PY=~/.cache/pd-headless/venv/bin/python
    $PY param_decomp_lab/scripts/validation/headless_check.py path/to/index.html
    $PY .../headless_check.py index.html \
        --clicks='[data-view=bases];;[data-view=interplay];;[data-view=inspector]' \
        --probes="document.querySelectorAll('#gallery .card').length"

`--clicks` / `--probes` are `;;`-separated (CSS selectors / JS expressions). Screenshots
land in `--screenshot-dir` (default: a `headless_shots/` dir next to the HTML file).
"""

import os
import sys
from pathlib import Path

import fire

_HOME = Path(os.environ.get("PD_HEADLESS_HOME", Path.home() / ".cache" / "pd-headless"))
_BROWSERS = _HOME / "browsers"
_LIBDIR = _HOME / "libs" / "extracted" / "usr" / "lib" / "x86_64-linux-gnu"


def _wire_toolchain() -> None:
    """Point Playwright at the isolated browser build and prepend Chromium's local libs."""
    assert _BROWSERS.exists(), f"no browser toolchain at {_BROWSERS}; run headless_setup.sh first"
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(_BROWSERS)
    if _LIBDIR.exists():
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{_LIBDIR}:{prev}" if prev else str(_LIBDIR)


def headless_check(
    html_path: str,
    clicks: str = "",
    probes: str = "",
    screenshot_dir: str | None = None,
    wait_ms: int = 800,
    timeout_ms: int = 8000,
    viewport_w: int = 1400,
    viewport_h: int = 1000,
) -> int:
    """Load the applet, exercise it, screenshot each step; return the JS-error count.

    Raises (non-zero exit under fire) if any console error / page error fired, so it doubles
    as a CI-style gate.
    """
    _wire_toolchain()
    # playwright lives only in the headless_setup.sh toolchain venv, not the project venv.
    from playwright.sync_api import sync_playwright  # pyright: ignore[reportMissingImports]

    html = Path(html_path).expanduser().resolve()
    assert html.exists(), f"html not found: {html}"
    shots = Path(screenshot_dir).expanduser() if screenshot_dir else html.parent / "headless_shots"
    shots.mkdir(parents=True, exist_ok=True)
    click_list = [c for c in clicks.split(";;") if c]
    probe_list = [p for p in probes.split(";;") if p]

    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": viewport_w, "height": viewport_h})
        page.on(
            "console",
            lambda m: errors.append(f"console.{m.type}: {m.text}")
            if m.type in ("error", "warning")
            else None,
        )
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        page.goto(html.as_uri())
        page.wait_for_timeout(wait_ms)
        page.screenshot(path=str(shots / "00_load.png"))
        for expr in probe_list:
            print(f"  probe {expr!r} -> {page.evaluate(expr)}")
        for i, selector in enumerate(click_list, 1):
            page.click(selector, timeout=timeout_ms)
            page.wait_for_timeout(wait_ms)
            page.screenshot(path=str(shots / f"{i:02d}_{_slug(selector)}.png"))
            print(f"  clicked {selector}")
        browser.close()

    print(f"screenshots -> {shots}")
    if errors:
        print(f"\n{len(errors)} JS error(s):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("\nno JS errors")
    return 0


def _slug(selector: str) -> str:
    keep = "".join(ch if ch.isalnum() else "_" for ch in selector)
    return keep.strip("_")[:30] or "step"


if __name__ == "__main__":
    fire.Fire(headless_check)
