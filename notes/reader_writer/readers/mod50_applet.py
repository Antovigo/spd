"""Rotatable 3D view of the block-15 readers g19, g29 and u32 at "=" (input of block 15's MLP), coloured by b mod 50.

Data: <run>/analysis/reader_writer/mod50_eq/applet_data.json (mod50_chain.py): for 6,000 random prompts per op, the
reader activations f_c = t_c / ||V_c|| (t_c the inner activation, V_c the read vector). Writes
notebooks/mod50_3d/index.html (one self-contained page, no external scripts) and notebook_mod50/fig/mod50_3d.js, the
same view for the mod-50 notebook: it mounts itself into the element with id "m50-3d" (ids prefixed "m50-").

    python mod50_applet.py
"""

import json
from pathlib import Path

RUN = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08")
DATA = RUN / "analysis/reader_writer/mod50_eq/applet_data.json"
OUT = Path(__file__).resolve().parent / "notebooks/mod50_3d/index.html"
EMBED = Path(__file__).resolve().parent / "notebook_mod50/fig/mod50_3d.js"
IDS = ("op", "up", "scale", "sz", "shadow", "reset", "cv", "grad")
EMBED_CSS = """
#m50-3d .controls { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; font-size: 13px; margin: 8px 0; }
#m50-3d .controls label { display: flex; gap: 6px; align-items: center; }
#m50-3d select, #m50-3d input, #m50-3d button { font: inherit; color: inherit; }
#m50-3d button { background: transparent; border: 1px solid var(--line); border-radius: 4px; padding: 2px 10px; cursor: pointer; }
#m50-3d .stage { border: 1px solid var(--line); border-radius: 6px; max-width: 760px; }
#m50-3d canvas { display: block; width: 100%; aspect-ratio: 1 / 1; max-height: 78vh; touch-action: none; cursor: grab;
  background: var(--bg); border: 0; }
#m50-3d .cbar { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); padding: 6px 10px; }
#m50-3d .cbar span.grad { flex: 1; height: 10px; border-radius: 2px; max-width: 320px; }
"""


def embed_js(data_json: str) -> str:
    """The page's controls, canvas and script as one file that mounts into #m50-3d (ids prefixed with m50-)."""
    body = PAGE.split("<script>", 1)[1].split("</script>", 1)[0].replace("__DATA__", data_json)
    html = PAGE.split('<div class="controls">', 1)[1].split('<ul class="defs">', 1)[0]
    html = '<div class="controls">' + html
    for i in IDS:
        for q in ('id="{}"', 'for="{}"'):
            html = html.replace(q.format(i), q.format("m50-" + i))
        body = body.replace(f'getElementById("{i}")', f'getElementById("m50-{i}")')
    body = body.replace(
        'for (const id of ["op", "up", "scale", "sz", "shadow"]) document.getElementById(id)',
        'for (const id of ["op", "up", "scale", "sz", "shadow"]) document.getElementById("m50-" + id)',
    )
    return ("(function () {\n"
            f"const st = document.createElement('style'); st.textContent = {json.dumps(EMBED_CSS)};\n"
            "document.head.appendChild(st);\n"
            f"document.getElementById('m50-3d').innerHTML = {json.dumps(html)};\n"
            + body + "\n})();\n")  # fmt: skip


PAGE = r"""<title>Mod 50 Reader Space</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>
/* Layout: one instrument panel; controls in a wrapping row above a square canvas, definitions below. */
:root { --bg: #f6f7f9; --panel: #ffffff; --fg: #1b1f24; --muted: #5d6673; --line: #d5dae1; --accent: #2a5db0;
  --shadow: rgba(40, 48, 60, 0.16); --box: #9aa3af;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121417; --panel: #1a1d21; --fg: #e7eaee; --muted: #9aa3af; --line: #30353c; --accent: #7ea6ec;
  --shadow: rgba(200, 210, 225, 0.13); --box: #6b7380; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #121417; --panel: #1a1d21; --fg: #e7eaee; --muted: #9aa3af; --line: #30353c;
  --accent: #7ea6ec; --shadow: rgba(200, 210, 225, 0.13); --box: #6b7380; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font: 14px/1.55 var(--body); margin: 0;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 20px 48px; }
main { max-width: 980px; margin: 0 auto; display: grid; gap: 14px; }
h1 { font-size: 20px; margin: 0; text-wrap: balance; }
.sub { color: var(--muted); font-size: 13px; max-width: 75ch; }
.controls { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; background: var(--panel);
  border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; font-size: 13px; }
.controls label { display: flex; gap: 6px; align-items: center; }
select, input, button { font: inherit; color: inherit; }
button { background: transparent; border: 1px solid var(--line); border-radius: 4px; padding: 2px 10px; cursor: pointer; }
button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.stage { position: relative; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }
canvas { display: block; width: 100%; aspect-ratio: 1 / 1; max-height: 78vh; touch-action: none; cursor: grab; }
canvas:active { cursor: grabbing; }
.cbar { display: flex; align-items: center; gap: 8px; font: 12px var(--mono); color: var(--muted); padding: 8px 12px; }
.cbar span.grad { flex: 1; height: 10px; border-radius: 2px; max-width: 320px; }
.defs { font-size: 13px; max-width: 78ch; }
.defs li { margin: 3px 0; }
code { font: 12.5px var(--mono); }
</style>
<main>
  <header>
    <h1>Mod 50 reader space at “=”</h1>
    <div class="sub">Run p-ba8b0c08 (addsub-08, Llama-3.1-8B), alive-only model. Each dot is one prompt
      <code>&lt;BOS&gt; a op b =</code>; its position is the activation of the three block-15 readers at “=”, its colour
      <i>b</i> mod 50. Drag to rotate, scroll or pinch to zoom.</div>
  </header>
  <div class="controls">
    <label for="op">Prompts <select id="op"><option value="0">a + b</option><option value="1">a − b</option>
      <option value="2">both</option></select></label>
    <label for="up">Vertical axis <select id="up"></select></label>
    <label for="scale">Scale <select id="scale"><option value="equal">same units on every axis</option>
      <option value="fit">fit each axis</option></select></label>
    <label for="sz">Dot size <input id="sz" type="range" min="0.6" max="4" step="0.2" value="1.6"></label>
    <label for="shadow"><input id="shadow" type="checkbox" checked> Shadows on the floor</label>
    <button id="reset" type="button">Reset view</button>
  </div>
  <div class="stage">
    <canvas id="cv" aria-label="3D scatter of the reader activations"></canvas>
    <div class="cbar">b mod 50: 0 <span class="grad" id="grad"></span> 49</div>
  </div>
  <ul class="defs">
    <li><i>a</i>, <i>b</i>: the integer operands, 1 to 400; op ∈ {+, −}. 6,000 random prompts per op.</li>
    <li>Readers: the block-15 MLP components <code>L15.gate.19</code> (g19), <code>L15.gate.29</code> (g29) and
      <code>L15.up.32</code> (u32). For a reader <i>c</i> with read vector <i>V<sub>c</sub></i> ∈ ℝ<sup>4096</sup>,
      the coordinate is <i>f<sub>c</sub></i> = <i>t<sub>c</sub></i> / ‖<i>V<sub>c</sub></i>‖, where
      <i>t<sub>c</sub></i> = <i>V<sub>c</sub></i> · RMSNorm(mid<sub>15</sub>) is the reader's inner activation at “=”
      and mid<sub>15</sub> the residual stream after block 15's attention (the input of its MLP).</li>
    <li>Nothing is centred: the grey lines cross at activation zero (0, 0, 0) when it lies inside the box. Shadows are
      the dots dropped straight down onto the floor of the box, along the vertical axis.</li>
  </ul>
</main>
<script>
const DATA = __DATA__;
const NAMES = {"L15.gate.19": "g19", "L15.gate.29": "g29", "L15.up.32": "u32"};
const VIR = [[68,1,84],[72,35,116],[64,67,135],[52,94,141],[41,120,142],[32,144,140],[34,167,132],[68,190,112],
  [121,209,81],[189,222,38],[253,231,37]];
function viridis(t) { const x = Math.min(1, Math.max(0, t)) * (VIR.length - 1), i = Math.min(VIR.length - 2, Math.floor(x)), u = x - i;
  return VIR[i].map((v, j) => Math.round(v + u * (VIR[i + 1][j] - v))); }
document.getElementById("grad").style.background =
  "linear-gradient(90deg," + VIR.map((c, i) => `rgb(${c}) ${100 * i / (VIR.length - 1)}%`).join(",") + ")";
const cols = Array.from({length: 50}, (_, r) => { const c = viridis(r / 49); return `rgb(${c[0]},${c[1]},${c[2]})`; });
const R = DATA.readers, pts = DATA.points;   // [op, a, b, f_0, f_1, f_2]
const upSel = document.getElementById("up");
R.forEach((r, i) => { const o = document.createElement("option"); o.value = i; o.textContent = NAMES[r] || r; upSel.appendChild(o); });
upSel.value = String(R.indexOf("L15.up.32") >= 0 ? R.indexOf("L15.up.32") : 2);
const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
let yaw = -0.7, pitch = 0.45, zoom = 1;
function css(v, fb) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || fb || ""; }
function draw() {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, Hh = cv.clientHeight;
  cv.width = Math.round(W * dpr); cv.height = Math.round(Hh * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, Hh);
  const opv = +document.getElementById("op").value, up = +upSel.value, sz = +document.getElementById("sz").value;
  const fit = document.getElementById("scale").value === "fit", shadow = document.getElementById("shadow").checked;
  const ax = [0, 1, 2].filter(i => i !== up).concat([up]);          // display x, y (floor), z (vertical)
  const sel = pts.filter(p => opv === 2 || p[0] === opv);
  const lo = [0, 1, 2].map(i => Math.min(...sel.map(p => p[3 + ax[i]]))), hi = [0, 1, 2].map(i => Math.max(...sel.map(p => p[3 + ax[i]])));
  const mid = lo.map((l, i) => (l + hi[i]) / 2), half = lo.map((l, i) => Math.max(1e-9, (hi[i] - l) / 2));
  const hmax = Math.max(...half), sc = fit ? half.map(() => 1) : half.map(h => h / hmax);
  const norm = (v, i) => (v - mid[i]) / half[i] * sc[i];             // into the box [-sc, sc]
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const S = 0.36 * Math.min(W, Hh) * zoom, cx0 = W / 2, cy0 = Hh / 2;
  function proj(x, y, z) {                                          // z up; rotate about z (yaw) then tilt (pitch)
    const x1 = cy * x - sy * y, y1 = sy * x + cy * y;
    const y2 = cp * y1 - sp * z, z2 = sp * y1 + cp * z;
    const persp = 3.2 / (3.2 + y2);
    return [cx0 + S * x1 * persp, cy0 - S * z2 * persp, y2];
  }
  const B = [sc[0], sc[1], sc[2]];
  ctx.lineWidth = 1;
  // floor grid
  ctx.strokeStyle = css("--line");
  for (let g = -4; g <= 4; g++) { const t = g / 4;
    let a = proj(-B[0], t * B[1], -B[2]), b = proj(B[0], t * B[1], -B[2]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    a = proj(t * B[0], -B[1], -B[2]); b = proj(t * B[0], B[1], -B[2]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
  // shadows
  const P = sel.map(p => { const v = [0, 1, 2].map(i => norm(p[3 + ax[i]], i)); return [v, p[2] % 50]; });
  if (shadow) { ctx.fillStyle = css("--shadow", "rgba(128, 128, 128, 0.16)");
    for (const [v] of P) { const q = proj(v[0], v[1], -B[2]); ctx.fillRect(q[0] - sz / 2, q[1] - sz / 2, sz, sz); } }
  // zero lines inside the box
  ctx.strokeStyle = css("--muted"); ctx.globalAlpha = 0.6;
  const z0 = [0, 1, 2].map(i => norm(0, i));
  for (let i = 0; i < 3; i++) { if (z0.some((z, j) => j !== i && Math.abs(z) > B[j])) continue;
    const a = z0.slice(), b = z0.slice(); a[i] = -B[i]; b[i] = B[i];
    const pa = proj(...a), pb = proj(...b); ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke(); }
  ctx.globalAlpha = 1;
  // points, far to near
  const Q = P.map(([v, r]) => { const q = proj(v[0], v[1], v[2]); return [q[0], q[1], q[2], r]; }).sort((u, w) => w[2] - u[2]);
  for (const q of Q) { ctx.fillStyle = cols[q[3]]; ctx.beginPath(); ctx.arc(q[0], q[1], sz, 0, 2 * Math.PI); ctx.fill(); }
  // box edges and axis labels
  ctx.strokeStyle = css("--box", css("--muted", "#888"));
  const cs = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]].map(c => proj(c[0]*B[0], c[1]*B[1], c[2]*B[2]));
  for (const [i, j] of [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]) {
    ctx.beginPath(); ctx.moveTo(cs[i][0], cs[i][1]); ctx.lineTo(cs[j][0], cs[j][1]); ctx.stroke(); }
  ctx.fillStyle = css("--fg"); ctx.font = "12px " + css("--mono");
  const lab = (i, pos, txt) => { const q = proj(...pos); ctx.fillText(txt, q[0] + 4, q[1] - 4); };
  const nm = i => NAMES[R[ax[i]]] || R[ax[i]];
  const fmt = v => (Math.abs(v) < 1e-3 ? 0 : v).toFixed(2);
  lab(0, [B[0] * 1.08, -B[1], -B[2]], nm(0)); lab(1, [-B[0], B[1] * 1.08, -B[2]], nm(1)); lab(2, [-B[0], -B[1], B[2] * 1.1], nm(2) + " (vertical)");
  ctx.fillStyle = css("--muted"); ctx.font = "11px " + css("--mono");
  for (let i = 0; i < 3; i++) for (const s of [-1, 1]) { const pos = [-B[0], -B[1], -B[2]]; pos[i] = s * B[i];
    lab(i, pos, fmt(mid[i] + s * half[i])); }
}
let drag = null;
cv.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY, yaw, pitch]; cv.setPointerCapture(e.pointerId); });
cv.addEventListener("pointermove", e => { if (!drag) return; yaw = drag[2] + (e.clientX - drag[0]) * 0.008;
  pitch = Math.max(-1.5, Math.min(1.5, drag[3] + (e.clientY - drag[1]) * 0.008)); draw(); });
cv.addEventListener("pointerup", () => { drag = null; });
cv.addEventListener("wheel", e => { e.preventDefault(); zoom = Math.max(0.4, Math.min(4, zoom * Math.exp(-e.deltaY * 0.001))); draw(); }, {passive: false});
let pinch = null;
cv.addEventListener("touchmove", e => { if (e.touches.length !== 2) return; e.preventDefault();
  const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
  if (pinch) zoom = Math.max(0.4, Math.min(4, zoom * d / pinch)); pinch = d; draw(); }, {passive: false});
cv.addEventListener("touchend", () => { pinch = null; });
for (const id of ["op", "up", "scale", "sz", "shadow"]) document.getElementById(id).addEventListener("input", draw);
document.getElementById("reset").addEventListener("click", () => { yaw = -0.7; pitch = 0.45; zoom = 1; draw(); });
window.addEventListener("resize", draw);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
draw();
</script>
"""


def main() -> None:
    data = json.loads(DATA.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dj = json.dumps(data, separators=(",", ":"))
    OUT.write_text(PAGE.replace("__DATA__", dj))
    EMBED.parent.mkdir(parents=True, exist_ok=True)
    EMBED.write_text(embed_js(dj))
    print("wrote", OUT, len(data["points"]), "points")


if __name__ == "__main__":
    main()
