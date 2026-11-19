"""What flows through the graph: plan, evidence, report, critique, and the final result."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

FraudType = Literal["high_amount", "card_testing", "impossible_travel", "none"]
Check = Literal["card_profile", "recent_activity", "last_location", "model_score"]
ALL_CHECKS: tuple[Check, ...] = ("card_profile", "recent_activity", "last_location", "model_score")


class Plan(BaseModel):
    """Planner output (LLM, schema-constrained)."""

    hypotheses: list[FraudType] = Field(
        description="Fraud patterns worth testing for this alert, most likely first"
    )
    checks: list[Check] = Field(description="Which lookups to run")
    rationale: str = Field(description="One or two sentences on why")


class Evidence(BaseModel):
    """One fact the Researcher established, with the numbers behind it."""

    id: str = Field(description="E1, E2, ...")
    check: str
    fact: str
    data: dict[str, Any]


class Report(BaseModel):
    """Reporter output (LLM, schema-constrained)."""

    verdict: Literal["fraud", "legitimate"]
    fraud_type: FraudType
    confidence: Literal["low", "medium", "high"]
    evidence_ids: list[str] = Field(description="Evidence ids that support the verdict")
    summary: str = Field(description="Three sentences at most, citing evidence ids like [E2]")


class Critique(BaseModel):
    approved: bool
    problems: list[str] = Field(default_factory=list)
    missing_checks: list[Check] = Field(default_factory=list)


class Step(BaseModel):
    node: str
    seconds: float
    detail: str


class Investigation(BaseModel):
    transaction_id: str
    plan: Plan | None
    evidence: list[Evidence]
    report: Report
    critiques: list[Critique]
    approved: bool = Field(description="False if the Critic still objected after the last round")
    rounds: int
    trace: list[Step]
    llm_calls: int
    seconds: float
