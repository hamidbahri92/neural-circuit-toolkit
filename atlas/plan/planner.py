
import numpy as np, networkx as nx
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class PredictReport:
    unsafe: bool
    risk: float
    reasons: Dict[str,float]

def dependency_plan(dag_edges: Dict[str, List[str]], target: str) -> List[str]:
    G = nx.DiGraph()
    for parent, children in dag_edges.items():
        for ch in children:
            G.add_edge(parent, ch)
    # Collect all ancestors of target
    req = set()
    def dfs(u):
        req.add(u)
        for v in G.predecessors(u):
            if v not in req: dfs(v)
    dfs(target)
    order = list(nx.topological_sort(G.subgraph(req)))
    return order

def overlap_score(support_a: Dict, support_b: Dict) -> float:
    ra = set(support_a.get("rows", [])); rb = set(support_b.get("rows", []))
    ca = set(support_a.get("cols", [])); cb = set(support_b.get("cols", []))
    inter = len(ra&rb)+len(ca&cb)
    denom = 1+len(ra|rb)+len(ca|cb)
    return inter/denom

def predict_interference(circA, circB) -> PredictReport:
    ov = overlap_score(circA.support, circB.support)
    # effect signature cosine
    ea, eb = np.array(circA.effect_sig, dtype=np.float32), np.array(circB.effect_sig, dtype=np.float32)
    if ea.size==0 or eb.size==0:
        cos = 0.0
    else:
        denom = (np.linalg.norm(ea)+1e-8)*(np.linalg.norm(eb)+1e-8)
        cos = float(ea.dot(eb)/denom)
    risk = 0.7*ov + 0.3*max(0.0, cos)
    return PredictReport(unsafe=(risk>0.7), risk=risk, reasons={"overlap":ov, "cosine":cos})
