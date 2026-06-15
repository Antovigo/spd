#!/usr/bin/env bash
# One-time setup for headless_check.py: an isolated Playwright + Chromium toolchain that
# runs with no display and no root. Kept out of the project venv on purpose — Chromium's
# system-library needs are awkward and shouldn't burden `make install-dev` / CI.
#
# Idempotent: re-run to repair. Installs into $PD_HEADLESS_HOME (default ~/.cache/pd-headless):
#   venv/      isolated venv with playwright + fire
#   browsers/  the headless Chromium build
#   libs/      the few shared libs Chromium needs that bare Ubuntu login nodes lack
#              (fetched via `apt-get download` — no root — and unpacked locally)
#
# Apt-based systems only (uses apt-get download / dpkg-deb). After it finishes, run checks
# with the printed python; headless_check.py wires the browser + lib paths automatically.
set -euo pipefail

ROOT="${PD_HEADLESS_HOME:-$HOME/.cache/pd-headless}"
VENV="$ROOT/venv"
BROWSERS="$ROOT/browsers"
LIBS="$ROOT/libs"
mkdir -p "$ROOT" "$LIBS"

echo ">> venv + playwright at $VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet playwright fire

echo ">> chromium into $BROWSERS"
PLAYWRIGHT_BROWSERS_PATH="$BROWSERS" "$VENV/bin/python" -m playwright install chromium

echo ">> chromium system libs into $LIBS (no root)"
tmp="$(mktemp -d)"
(
  cd "$tmp"
  apt-get download libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0
  for d in *.deb; do dpkg-deb -x "$d" "$LIBS/extracted"; done
)
rm -rf "$tmp"

echo
echo "toolchain ready at $ROOT"
echo "run checks with: $VENV/bin/python <repo>/param_decomp_lab/scripts/validation/headless_check.py <file.html>"
