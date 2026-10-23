"""Assemble one notebook per period (mod 2, mod 10, mod 20) from the notebook pages, in a fixed five-part order:

    1. How the period is represented at the operand positions, blocks 0-16
    2. How the geometry changes along the way (compression, distortion, separation, the SwiGLU lens)
    3. Which components read the representation, and how (including the dual basis)
    4. Which neurons read the representation
    5. How the information is copied to the last token by the attention of blocks 15 and 16

Sources are the page entries (`entries.js` of notebook/, notebook_mod2/, notebook_neurons/, notebook_mod20/), loaded
with node so the template literals are evaluated exactly. Each page is split at its <h3> sections; LAYOUT says where
every section goes. Sections are renumbered "<part>.<k>", and each page's references to its own sections ("section 8",
"sections 1 and 7") are rewritten to the new numbers (an unmapped reference fails the build). A part with no section is
a "Coming soon" placeholder. Every page's "Setup and notation" goes to the notebook's preface, so every symbol is defined
before the parts that use it; page summaries form the preface's overview. Figures are copied next to each notebook.

Writes notebooks/<period>/index.html and notebooks/<period>/fig/. Re-run after any page changes:

    python build_notebooks.py
"""

import html as _html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "notebooks"
SOURCES = {"notebook": "notebook/fig", "notebook_mod2": "notebook_mod2/fig", "notebook_neurons": "notebook_neurons/fig",
           "notebook_mod20": "notebook_mod20/fig"}  # fmt: skip
PARTS = (
    "How the period is represented at the operand positions, blocks 0–16",
    "How the geometry changes from block 0 to block 16",
    "Which components read the representation, and how",
    "Which neurons read the representation",
    "How the information is copied to the last token (attention, blocks 15–16)",
)
# page id -> (source dir, display name)
PAGES = {
    "mod10": ("notebook", "Mod 10 representations"),
    "mod10_p3": ("notebook", "Mod 10 representations at b (position 3)"),
    "mod10_geometry": ("notebook", "Mod 10 representations: geometry across blocks 0-16"),
    "mod10-neurons": ("notebook_neurons", "Mod 10 representations in the neuron basis"),
    "mod2_l3": ("notebook_mod2", "Mod 2: block-3 d189 and the parity line"),
    "mod20": ("notebook_mod20", "Mod 20 representation at a (position 1)"),
}
# period -> {"title", "pages" (preface order), "parts": [[(page id, old section number), ...] x 5]}
LAYOUT = {
    "mod10": {
        "title": "Mod 10 representations",
        "pages": ["mod10", "mod10_p3", "mod10_geometry", "mod10-neurons"],
        "parts": [
            [("mod10", 1), ("mod10", 7), ("mod10", 8), ("mod10_p3", 1), ("mod10_p3", 2), ("mod10_p3", 4), ("mod10_p3", 5),
             ("mod10_p3", 6)],
            [("mod10_geometry", 1), ("mod10_geometry", 2), ("mod10_geometry", 3), ("mod10_geometry", 4),
             ("mod10_geometry", 5), ("mod10_geometry", "Caveat"), ("mod10_p3", 3), ("mod10", 9)],
            [("mod10", 2), ("mod10", 3), ("mod10", 4), ("mod10", 5), ("mod10", 6)],
            [("mod10-neurons", k) for k in range(1, 9)],
            [("mod10", 10), ("mod10", 11)],
        ],
        "refs": {"mod10_p3": [('the "Mod 10 representations" page (sections 1–8)', "sections 1.1–1.3 and 3.1–3.5")],
                 "mod10-neurons": [("the dual-basis test of the component page", "the dual-basis test of section 3.3")],
                 "mod10": [("(see the position-3 page)", "(see sections 1.4–1.7 and 2.7)"),
                           ("Nothing on this page is centred or detrended.",
                            "Nothing in this notebook is centred or detrended, except in section 2.3, which says so.")],
                 "mod10_geometry": [("sections 1–4", "sections 2.1–2.4"), ("Sections 1–4", "Sections 2.1–2.4")]},
    },
    "mod2": {
        "title": "Mod 2 representations",
        "pages": ["mod2_l3"],
        "parts": [[], [("mod2_l3", 4)], [("mod2_l3", 1), ("mod2_l3", 2), ("mod2_l3", 3)], [], []],
        "refs": {},
    },
    "mod20": {
        "title": "Mod 20 representations",
        "pages": ["mod20"],
        "parts": [[("mod20", 1), ("mod20", 4)], [("mod20", 2), ("mod20", 5), ("mod20", 6)], [("mod20", 3), ("mod20", 8)],
                  [("mod20", 7), ("mod20", 9)], []],
        "refs": {"mod20": [("representation of the mod-10 page", "representation of the mod-10 notebook")]},
    },
}  # fmt: skip

CSS = """
:root { --bg: #fafaf8; --fg: #1d1d1b; --muted: #6b6b66; --line: #deded8; --accent: #2a5db0;
  --body: system-ui, -apple-system, "Segoe UI", sans-serif; --mono: ui-monospace, "SFMono-Regular", Menlo, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #161615; --fg: #e9e9e4; --muted: #9a9a93; --line: #34342f; --accent: #7ea6ec; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #161615; --fg: #e9e9e4; --muted: #9a9a93; --line: #34342f; --accent: #7ea6ec; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body); margin: 0; padding: 24px 16px 64px; }
main { max-width: 1000px; margin: 0 auto; }
header { border-bottom: 1px solid var(--line); padding-bottom: 12px; margin-bottom: 8px; }
h1 { font-size: 22px; margin: 0; }
h2 { font-size: 20px; margin: 48px 0 6px; border-top: 2px solid var(--line); padding-top: 16px; text-wrap: balance; }
h3 { font-size: 16px; margin: 30px 0 4px; text-wrap: balance; }
h4 { font-size: 15px; margin: 20px 0 4px; }
p, li { max-width: 74ch; }
a { color: var(--accent); }
.note, .src { color: var(--muted); font-size: 13px; }
.soon { color: var(--muted); font-style: italic; }
nav ol { padding-left: 20px; } nav li { margin: 2px 0; } nav ol ol { font-size: 13px; }
figure { margin: 12px 0; } figure img, img { max-width: 100%; height: auto; display: block; background: #fff; border: 1px solid var(--line); }
figcaption { color: var(--muted); font-size: 13px; max-width: 74ch; margin-top: 4px; }
.tw { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
td, th { padding: 3px 12px 3px 0; text-align: left; border-bottom: 1px solid var(--line); }
code { font: 13px var(--mono); }
"""


def load(src: str) -> list[dict]:
    js = "global.window = {}; require(process.argv[2]); process.stdout.write(JSON.stringify(window.NOTEBOOK));"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(js)
    return json.loads(subprocess.check_output(["node", fh.name, str(HERE / src / "entries.js")]))


def split(html: str) -> tuple[str, list[tuple[str, str]]]:
    """(text before the first <h3>, [(heading text, section html after the heading)])."""
    parts = re.split(r"<h3>(.*?)</h3>", html, flags=re.S)
    return parts[0], [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]


def key_of(heading: str) -> int | str:
    m = re.match(r"\s*(\d+)\.\s", heading)
    return int(m.group(1)) if m else re.sub(r"<[^>]*>", "", heading).split(":")[0].split()[0]


APPLIED: set[str] = set()


def renumber(text: str, page: str, mapping: dict[int, str], extra: list[tuple[str, str]]) -> str:
    for old, new in extra:
        if old in text:
            APPLIED.add(old)
        text = text.replace(old, new)

    def one(m: re.Match) -> str:
        word, nums = m.group(1), m.group(2)
        if "–" in nums:
            raise ValueError(f"{page}: unmapped range reference '{m.group(0)}'")
        new = re.sub(
            r"\d+",
            lambda n: mapping[int(n.group(0))] if int(n.group(0)) in mapping else n.group(0),
            nums,
        )
        for n in re.findall(r"\d+", nums):
            if int(n) not in mapping:
                raise ValueError(f"{page}: reference to section {n} with no new number")
        return word + " " + new

    return re.sub(r"\b([Ss]ections?) (\d+(?:(?:, | and | or |–)\d+)*)\b(?!\.\d|\d)", one, text)


def build(period: str, spec: dict, pages: dict[str, dict]) -> None:
    out = OUT / period
    if out.exists():
        shutil.rmtree(out)
    (out / "fig").mkdir(parents=True)
    split_pages = {pid: split(pages[pid]["html"]) for pid in spec["pages"]}
    mapping: dict[str, dict[int, str]] = {pid: {} for pid in spec["pages"]}
    for p, items in enumerate(spec["parts"], 1):
        for k, (pid, sec) in enumerate(items, 1):
            if isinstance(sec, int):
                mapping[pid][sec] = f"{p}.{k}"
    used = set()

    def fix(pid: str, text: str) -> str:
        text = renumber(text, pid, mapping[pid], spec["refs"].get(pid, []))
        for a, b in (
            ("on this page", "in this notebook"),
            ("this page", "this notebook"),
            ("This page", "This notebook"),
        ):
            text = text.replace(a, b)

        def img(m: re.Match) -> str:
            name = Path(m.group(1)).name
            src = HERE / SOURCES[PAGES[pid][0]] / name
            assert src.exists(), src
            shutil.copy(src, out / "fig" / name)
            return f'src="fig/{name}"'

        return re.sub(r'src="(?:notebook/)?fig/([^"]+)"', img, text)

    toc, body = [], []
    # preface: overview (page summaries) and every page's setup
    body.append('<h2 id="p0">0. Overview and notation</h2>')
    toc.append('<li><a href="#p0">Overview and notation</a><ol>')
    body.append('<h3 id="s0-1">0.1 Overview</h3>')
    toc.append('<li><a href="#s0-1">0.1 Overview</a></li>')
    for pid in spec["pages"]:
        pre = split_pages[pid][0]
        body.append(
            f'<p class="src">From the page "{_html.escape(PAGES[pid][1])}" ({pages[pid]["date"]}):</p>'
            + fix(pid, pre)
        )
    k = 1
    for pid in spec["pages"]:
        for head, sec in split_pages[pid][1]:
            if re.match(r"\s*Setup", head):
                k += 1
                title = "Setup and notation" + (
                    "" if len(spec["pages"]) == 1 else f": {PAGES[pid][1]}"
                )
                body.append(f'<h3 id="s0-{k}">0.{k} {_html.escape(title)}</h3>' + fix(pid, sec))
                toc.append(f'<li><a href="#s0-{k}">0.{k} {_html.escape(title)}</a></li>')
                used.add((pid, head))
    toc.append("</ol></li>")
    for p, items in enumerate(spec["parts"], 1):
        body.append(f'<h2 id="p{p}">{p}. {PARTS[p - 1]}</h2>')
        toc.append(f'<li><a href="#p{p}">{PARTS[p - 1]}</a><ol>')
        if not items:
            body.append('<p class="soon">Coming soon.</p>')
        for j, (pid, sec) in enumerate(items, 1):
            match = [(h, s) for h, s in split_pages[pid][1] if key_of(h) == sec]
            assert len(match) == 1, (period, pid, sec, [h for h, _ in split_pages[pid][1]])
            head, sec_html = match[0]
            used.add((pid, head))
            title = re.sub(r"^\s*\d+\.\s*", "", head)
            body.append(f'<h3 id="s{p}-{j}">{p}.{j} {title}</h3>'
                        f'<p class="src">From the page "{_html.escape(PAGES[pid][1])}", section {head.split(".")[0] if isinstance(sec, int) else "“" + str(sec) + "”"}.</p>'
                        + fix(pid, sec_html))  # fmt: skip
            toc.append(f'<li><a href="#s{p}-{j}">{p}.{j} {re.sub(r"<[^>]*>", "", title)}</a></li>')
        toc.append("</ol></li>")
    missing = [
        (pid, h) for pid in spec["pages"] for h, _ in split_pages[pid][1] if (pid, h) not in used
    ]
    assert not missing, f"{period}: sections not placed: {missing}"
    page = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{spec['title'].title()}</title><style>{CSS}</style></head><body><main>"
        f'<header><h1>{spec["title"]}</h1><div class="note">Run p-ba8b0c08 (addsub-08, Llama-3.1-8B), prompts '
        "<code>&lt;BOS&gt; a op b =</code>. Assembled by <code>build_notebooks.py</code> from the notebook pages.</div></header>"
        f"<nav><ol>{''.join(toc)}</ol></nav>{''.join(body)}"
        "<script>document.querySelectorAll('table').forEach(t=>{const w=document.createElement('div');w.className='tw';"
        "t.replaceWith(w);w.appendChild(t);});</script></main></body></html>"
    )
    (out / "index.html").write_text(page)
    print(
        period,
        "parts",
        [len(x) for x in spec["parts"]],
        "figures",
        len(list((out / "fig").iterdir())),
    )


def main() -> None:
    pages = {}
    for src in SOURCES:
        for e in load(src):
            pages[e["id"]] = e
    for period, spec in LAYOUT.items():
        build(period, spec, pages)
    wanted = {old for spec in LAYOUT.values() for lst in spec["refs"].values() for old, _ in lst}
    assert wanted <= APPLIED, f"reference rewrites that matched nothing: {wanted - APPLIED}"


if __name__ == "__main__":
    main()
