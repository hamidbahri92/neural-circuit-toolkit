
import numpy as np
from typing import Dict, List
from ..core.spec import CircuitDiff, content_address_store

def mask_to_circuit(circuit_id: str, layer: int, rows_mask: np.ndarray, behavior_axis: str = "intensity", base_coeff: float = 0.0, delta_coeff: float = 1.0) -> CircuitDiff:
    rows = np.where(rows_mask)[0].tolist()
    # Basis U: identity on selected rows (k = len(rows))
    if len(rows) == 0:
        U = np.zeros((1,1), dtype=np.float32)
    else:
        U = np.eye(len(rows), dtype=np.float32)
    # store as blob
    import io
    bio = io.BytesIO()
    np.save(bio, U.astype(np.float32))
    ref = content_address_store(bio.getvalue(), suffix=".npy")
    deltas = {behavior_axis: (np.ones(U.shape[0], dtype=float) * float(delta_coeff)).tolist()}
    c = CircuitDiff(
        circuit_id=circuit_id,
        support={"layer": layer, "type": "mlp", "rows": rows, "cols": []},
        basis_blob=ref,
        deltas=deltas,
        prereq=[],
        conflicts=[],
        tests=[],
        effect_sig=[],
        status="provisional",
        sign=None
    )
    return c
