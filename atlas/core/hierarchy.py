
from __future__ import annotations
from typing import Dict, List
from .spec import AtlasManifest, CircuitDiff

LEVELS = ["atomic", "primitive", "composite", "behavioral", "persona"]

def add_level_tag(c: CircuitDiff, level: str) -> CircuitDiff:
    if "tags" not in c.support:
        c.support["tags"] = {}
    c.support["tags"]["level"] = level
    return c

def get_level(c: CircuitDiff) -> str:
    return c.support.get("tags", {}).get("level", "primitive")

def decompose(atlas: AtlasManifest, node_id: str) -> List[str]:
    """Return leaf circuits under node_id via DAG traversal. If node_id is a leaf, return [node_id]."""
    dag = atlas.dag or {}
    leaves = []
    def dfs(u):
        children = dag.get(u, [])
        if not children:
            leaves.append(u); return
        for v in children:
            dfs(v)
    dfs(node_id)
    return leaves

def compose(atlas: AtlasManifest, component_ids: List[str]) -> List[CircuitDiff]:
    return [atlas.circuits[cid] for cid in component_ids if cid in atlas.circuits]
