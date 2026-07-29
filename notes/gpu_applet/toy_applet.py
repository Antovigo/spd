"""Minimal GPU-backed applet: serves a page that samples MLP activations on demand.

Toy example for the recipe in recipe.md. Run on a compute node, view from a laptop
browser through the reverse tunnel:

    python notes/gpu_applet/toy_applet.py --port 8123
"""

import argparse
import time

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU()).to(device)

PAGE = """
<!doctype html>
<title>GPU applet toy test</title>
<style>
  body { font-family: sans-serif; max-width: 640px; margin: 2rem auto; }
  #bars { display: flex; align-items: flex-end; gap: 2px; height: 200px; }
  #bars div { flex: 1; background: steelblue; }
</style>
<h2>Activation histogram <small id="meta"></small></h2>
<button onclick="sample()">Sample new batch</button>
<div id="bars"></div>
<script>
async function sample() {
  const r = await (await fetch("api/activations")).json();
  document.getElementById("meta").textContent =
    `(${r.device}, ${r.ms.toFixed(1)} ms)`;
  const peak = Math.max(...r.counts);
  document.getElementById("bars").innerHTML = r.counts
    .map(c => `<div style="height:${(100 * c) / peak}%"></div>`)
    .join("");
}
sample();
</script>
"""


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


@app.get("/api/activations")
def activations() -> dict[str, object]:
    t0 = time.perf_counter()
    with torch.no_grad():
        acts = mlp(torch.randn(4096, 64, device=device))
    counts = torch.histc(acts.float(), bins=40, min=-0.5, max=2.0)
    ms = (time.perf_counter() - t0) * 1000
    return {"device": str(device), "ms": ms, "counts": counts.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
