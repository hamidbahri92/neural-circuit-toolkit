
from __future__ import annotations
from typing import List
from atlas.core.atlas_store import AtlasStore

def view_atlas(path: str) -> str:
    store = AtlasStore(path); man = store.load()
    lines = []
    lines.append(f"Atlas version: {man.version} | family: {man.family}")
    lines.append("Circuits:")
    for cid, c in man.circuits.items():
        rows = len(c.support.get("rows", []))
        level = c.support.get("tags", {}).get("level", "primitive")
        lines.append(f"  - {cid} [{level}] rows={rows}")
    lines.append("DAG:")
    for p, chs in (man.dag or {}).items():
        for ch in chs:
            lines.append(f"  {p} -> {ch}")
    return "\n".join(lines)
