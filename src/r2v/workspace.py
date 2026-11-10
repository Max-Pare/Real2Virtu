"""Workspace paths and stage caching."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, stage: str) -> Path:
        d = self.root / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _stamp(self, stage: str) -> Path:
        return self.root / stage / ".stamp"

    def stored_key(self, stage: str) -> str | None:
        p = self._stamp(stage)
        return p.read_text().strip() if p.exists() else None

    def is_fresh(self, stage: str, key: str) -> bool:
        return self.stored_key(stage) == key

    def reset(self, stage: str) -> Path:
        """Empty the stage folder before a re-run."""
        d = self.root / stage
        if d.exists():
            shutil.rmtree(d)
        return self.dir(stage)

    def mark(self, stage: str, key: str) -> None:
        self.dir(stage)  # stages that write elsewhere (export) have no folder yet
        self._stamp(stage).write_text(key)


def make_key(upstream: str, payload: object) -> str:
    blob = upstream + json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def file_fingerprint(path: Path) -> list:
    st = Path(path).stat()
    return [str(Path(path).resolve()), st.st_size, int(st.st_mtime)]
