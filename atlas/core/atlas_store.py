
from __future__ import annotations
import json, os
from typing import Dict, List, Optional
from .spec import AtlasManifest, CircuitDiff, content_address_store, content_address_load

class AtlasStore:
    def __init__(self, path: str):
        self.path = path
        self.manifest: Optional[AtlasManifest] = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def new(self, version: str = "v0.1", family: str = "mock_residual", token_semantics: Dict = None):
        self.manifest = AtlasManifest(version=version, family=family, projections={}, circuits={}, dag={}, token_semantics=token_semantics or {})
        return self

    def add_circuit(self, c: CircuitDiff):
        assert self.manifest is not None
        self.manifest.circuits[c.circuit_id] = c

    def add_edge(self, parent: str, child: str):
        assert self.manifest is not None
        self.manifest.dag.setdefault(parent, []).append(child)

    def save(self):
        assert self.manifest is not None
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(self.manifest.to_json())
        return self.path

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # reconstruct CircuitDiffs
        circuits = {}
        for k, v in data["circuits"].items():
            circuits[k] = CircuitDiff(**v)
        self.manifest = AtlasManifest(
            version=data["version"],
            family=data["family"],
            projections=data["projections"],
            circuits=circuits,
            dag=data["dag"],
            token_semantics=data.get("token_semantics", {}),
            families=data.get("families", {}),
            lineage=data.get("lineage", []),
            sign=data.get("sign"),
        )
        return self.manifest
