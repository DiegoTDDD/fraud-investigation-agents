"""Baselines the agents are measured against.

model_only   the fraud model's flag alone (no pattern)
rules_only   the Critic's signals in a fixed order, no language model
single_agent all checks and one LLM call, no planner and no critic (in graph.py)
"""

from __future__ import annotations

from fraud_agents.critic import signals, supported_patterns
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
    ids = [e.id for e in evidence]
    kind = (supported_patterns(signals(evidence)) or ["none"])[0]
    return Report(verdict="legitimate" if kind == "none" else "fraud", fraud_type=kind,
                  confidence="medium", evidence_ids=ids, summary=f"Rules: {kind}.")  # fmt: skip
