
from typing import List, Tuple
import random

HEDGES = ["maybe","perhaps","it seems","it might","I think","likely","possibly"]
ASSERTS = ["clearly","definitely","certainly","without doubt","obviously"]
FORMAL_UP = ["therefore","however","moreover","thus","in addition","consequently"]
CASUAL_UP = ["yeah","lol","btw","kinda","sorta","nah","tbh"]
REFUSE = ["I cannot comply with that request.", "I won't do that.", "That would be inappropriate."]
NON_REFUSE = ["Sure, here's how.", "Absolutely, let's do it.", "Yes, proceeding."]

def minimal_pairs(behavior: str, n: int = 32, seed: int = 1234) -> List[Tuple[str,str]]:
    random.seed(seed)
    pairs = []
    for i in range(n):
        base = f"Task {i}: write a short response about the weather."
        if behavior == "hedging":
            a = base + " " + random.choice(HEDGES) + ", it could rain later."
            b = base + " " + random.choice(ASSERTS) + ", it will rain later."
        elif behavior == "formality":
            a = base + " " + random.choice(FORMAL_UP) + " precipitation levels may increase."
            b = base + " " + random.choice(CASUAL_UP) + " might rain later."
        elif behavior == "refusal":
            a = base + " " + random.choice(REFUSE)
            b = base + " " + random.choice(NON_REFUSE)
        else:
            a = base + " this is variant A."
            b = base + " this is variant B."
        pairs.append((a,b))
    return pairs
