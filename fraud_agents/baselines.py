"""Baselines the agents are measured against.

model_only   the fraud model's flag alone (no pattern)
rules_only   the Critic's signals in a fixed order, no language model
single_agent all checks and one LLM call, no planner and no critic (in graph.py)
"""

from __future__ import annotations

from fraud_agents.critic import signals
from fraud_agents.schema import Evidence, Report


def model_only(evidence: list[Evidence]) -> Report:
    score = next(e for e in evidence if e.check == "model_score")
    flagged = score.data["flagged"]
    return Report(
        verdict="fraud" if flagged else "legitimate",
        fraud_type="none",
        confidence="medium",
        evidence_ids=[score.id],
        summary=f"Model probability {score.data['probability']:.3f} [{score.id}].",
    )


def rules_only(evidence: list[Evidence]) -> Report:
    s = signals(evidence)
    ids = [e.id for e in evidence]
    if s.get("micro_burst"):
        kind = "card_testing"
    elif s.get("high_amount"):
        kind = "high_amount"
    elif s.get("away_from_home") and s.get("model_flagged"):
        kind = "impossible_travel"
    else:
        kind = "none"
    return Report(verdict="legitimate" if kind == "none" else "fraud", fraud_type=kind,
                  confidence="medium", evidence_ids=ids, summary=f"Rules: {kind}.")  # fmt: skip
