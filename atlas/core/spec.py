
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Optional, Tuple, Any
import json, hashlib, os, time, base64, hmac, pathlib

BLOB_DIR = os.environ.get("ATLAS_BLOB_DIR", "/mnt/data/universal_atlas/blobs")

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def content_address_store(data: bytes, suffix: str = ".bin") -> str:
    h = sha256_bytes(data)
    p = os.path.join(BLOB_DIR, f"{h}{suffix}")
    os.makedirs(BLOB_DIR, exist_ok=True)
    if not os.path.exists(p):
        with open(p, "wb") as f:
            f.write(data)
    return f"sha256:{h}{suffix}"

def content_address_load(ref: str) -> bytes:
    assert ref.startswith("sha256:"), "Unsupported ref"
    fname = ref.split("sha256:")[1]
    p = os.path.join(BLOB_DIR, fname)
    with open(p, "rb") as f:
        return f.read()

def sign_hmac(message: bytes, key: bytes) -> str:
    sig = hmac.new(key, message, digestmod="sha256").digest()
    return "hmac256:" + base64.b64encode(sig).decode("utf-8")

@dataclass
class CircuitDiff:
    circuit_id: str
    support: Dict[str, Any]                     # e.g., {'layer': 18, 'type': 'mlp', 'rows':[...], 'cols':[]}
    basis_blob: str                             # content-addressed U matrix (k x |support|) as float32 .npy
    deltas: Dict[str, List[float]]              # named differential coefficient vectors (length k)
    prereq: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)
    effect_sig: List[float] = field(default_factory=list)
    status: Literal["provisional","stable"] = "provisional"
    sign: Optional[str] = None                  # signature over JSON canonical form

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",",":"))

    def sign_with_key(self, key: bytes) -> None:
        self.sign = sign_hmac(self.to_json().encode("utf-8"), key)

@dataclass
class AtlasManifest:
    version: str
    family: str
    projections: Dict[str, str]                 # layer_key -> sha256 ref of projection matrix
    circuits: Dict[str, CircuitDiff]            # id -> circuit
    dag: Dict[str, List[str]]                   # edges: parent -> [children]
    token_semantics: Dict[str, Any]             # encoder spec
    families: Dict[str, Dict[str, float]] = field(default_factory=dict)  # circuit_id -> {arch: fidelity}
    lineage: List[Dict[str, Any]] = field(default_factory=list)
    sign: Optional[str] = None

    def to_json(self) -> str:
        d = asdict(self)
        # circuits need to be serializable
        d["circuits"] = {k: json.loads(v.to_json()) for k,v in self.circuits.items()}
        return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",",":"))

    def sign_with_key(self, key: bytes) -> None:
        self.sign = sign_hmac(self.to_json().encode("utf-8"), key)

@dataclass
class Plan:
    circuits: List[str]
    knobs: Dict[str, float]
    predicted_deltas: List[float] = field(default_factory=list)
    predicted_interference: float = 0.0
    stability_margins: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",",":"))

@dataclass
class Patch:
    manifest: Dict[str, Any]                    # what changed
    blobs: List[str] = field(default_factory=list)
    reversible: bool = True
    created_ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",",":"))
