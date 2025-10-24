from typing import Any, Dict, List
import math

def simple_score(text: str, query: str) -> float:
    if not text:
        return 0.0
    score = min(1.0, len(text) / 400.0) 
    if any(w in text.lower() for w in query.lower().split()):
        score += 0.5
    return max(0.0, min(2.0, score))

class EvaluatorAgent:
    """Compare agent outputs and pick best-fit or produce combined summary."""

    async def evaluate(self, query: str, agent_results: List[Dict[str, str]]) -> Dict:
        scores = []
        for r in agent_results:
            s = simple_score(r.get("output",""), query)
            scores.append((s, r))
        scores.sort(key=lambda x: x[0], reverse=True)
        best = scores[0][1] if scores else {}
        combined = "\n\n".join(f"[{r['role']}]\n{r['output']}" for _, r in scores)
        return {"best": best, "combined": combined, "ranked": [(s, r["role"]) for s, r in scores]}