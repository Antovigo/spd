"""File-bundle metric payloads written verbatim under the run dir by the local sink."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import override


@dataclass(frozen=True)
class RunDirArtifact:
    """Raw files a metric wants written under `out_dir/<dir>/`.

    Handled by the local sink only — excluded from both `metrics.jsonl` and wandb.
    When `manifest_var` is set, `<dir>/manifest.js` is regenerated after each write to
    list the dir's `step_*.js` files (sorted by step) as `window.<manifest_var> = [...]`,
    so a `file://`-opened applet in the same dir can discover every snapshot.

    Lives outside `run_sink` so metrics can import it without pulling the sink's
    wandb/infra import chain into `eval_metrics` (import cycle).
    """

    dir: str
    files: Mapping[str, bytes]
    manifest_var: str | None = None

    def write(self, out_dir: Path) -> None:
        target = out_dir / self.dir
        target.mkdir(parents=True, exist_ok=True)
        for name, content in self.files.items():
            (target / name).write_bytes(content)
        if self.manifest_var is not None:
            step_files = sorted(
                target.glob("step_*.js"), key=lambda p: int(p.stem.removeprefix("step_"))
            )
            listing = json.dumps([p.name for p in step_files])
            (target / "manifest.js").write_text(f"window.{self.manifest_var} = {listing};\n")

    @override
    def __repr__(self) -> str:
        total_mb = sum(len(content) for content in self.files.values()) / 1e6
        return f"RunDirArtifact({self.dir}/: {len(self.files)} files, {total_mb:.1f} MB)"
