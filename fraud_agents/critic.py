"""The Critic: checks a report against the evidence and says what to redo.

It is rules, not a model. Its job is to catch reports the evidence does not support, and an
LLM judging another LLM's report would share its blind spots (ADR 0002). Each rule names the
evidence it relies on, so a rejection is specific enough for the Reporter to act on.
"""

from __future__ import annotations

from typing import Any

from fraud_agents.schema import Check, Critique, Evidence, Report

MIN_MICRO_CHARGES = 3  # card testing: at least this many charges <= 10 BRL in 5 minutes
HIGH_AMOUNT_RATIO = 3.0  # high amount: at least 3x the card's 95th-percentile purchase
HIGH_AMOUNT_FLOOR = 2000.0  # ...and at least this many BRL

REQUIRED_CHECKS: dict[str, tuple[Check, ...]] = {
    "card_testing": ("recent_activity",),
    "high_amount": ("card_profile",),
    "impossible_travel": ("card_profile", "model_score"),
}
ALWAYS_REQUIRED: tuple[Check, ...] = ("card_profile", "model_score")


def signals(evidence: list[Evidence]) -> dict[str, Any]:
    """Pattern signals computed from the evidence data (not from the model's words)."""
    by_check = {e.check: e.data for e in evidence}
    alert = by_check.get("alert", {})
    profile = by_check.get("card_profile")
    recent = by_check.get("recent_activity")
    out: dict[str, Any] = {"checks": set(by_check)}
    if recent is not None:
        out["micro_burst"] = recent["micro_charges"] >= MIN_MICRO_CHARGES
    if profile is not None:
        p95 = profile.get("p95_amount")
        amount = alert.get("amount", 0.0)
        out["high_amount"] = bool(
            p95 and amount >= HIGH_AMOUNT_RATIO * p95 and amount >= HIGH_AMOUNT_FLOOR
        )
        home = profile.get("home_city")
        out["away_from_home"] = bool(home) and alert.get("city") != home
    if "model_score" in by_check:
        out["model_flagged"] = by_check["model_score"]["flagged"]
    return out


def supported_patterns(sig: dict[str, Any]) -> list[str]:
    """Patterns the evidence supports on its own, strongest first."""
    out = []
    if sig.get("micro_burst"):
        out.append("card_testing")
    if sig.get("high_amount"):
        out.append("high_amount")
    if sig.get("away_from_home") and sig.get("model_flagged"):
        out.append("impossible_travel")
    return out


def critique(report: Report, evidence: list[Evidence]) -> Critique:
    ids = {e.id: e for e in evidence}
    sig = signals(evidence)
    problems: list[str] = []
    missing: list[Check] = []

    unknown = [i for i in report.evidence_ids if i not in ids]
    if unknown:
        problems.append(f"Cited evidence that does not exist: {', '.join(unknown)}.")
    if not [i for i in report.evidence_ids if i in ids]:
        problems.append("The report cites no evidence; cite the evidence ids that support it.")

    if report.verdict == "legitimate" and report.fraud_type != "none":
        problems.append("A legitimate verdict must have fraud_type 'none'.")
    if report.verdict == "fraud" and report.fraud_type == "none":
        problems.append("A fraud verdict must name the pattern (fraud_type).")

    for check in REQUIRED_CHECKS.get(report.fraud_type, ()):
        if check not in sig["checks"]:
            missing.append(check)
    if report.fraud_type == "card_testing" and sig.get("micro_burst") is False:
        problems.append(
            f"card_testing needs at least {MIN_MICRO_CHARGES} micro-charges in the previous 5 "
            "minutes; the recent-activity evidence does not show that."
        )
    if report.fraud_type == "high_amount" and sig.get("high_amount") is False:
        problems.append(
            f"high_amount needs an amount of at least {HIGH_AMOUNT_RATIO:g}x the card's "
            f"95th-percentile purchase and {HIGH_AMOUNT_FLOOR:,.0f} BRL; the profile evidence "
            "does not show that."
        )
    if report.fraud_type == "impossible_travel" and sig.get("away_from_home") is False:
        problems.append("impossible_travel needs the purchase to be outside the home city.")
    if (report.fraud_type == "impossible_travel" and sig.get("away_from_home")
            and sig.get("model_flagged") is False):  # fmt: skip
        # In this data many legitimate purchases are in another city; being away is not enough
        # (eval run 1, case c28).
        problems.append("The purchase is outside the home city, but the fraud model does not "
                        "flag it, so nothing makes the travel implausible.")  # fmt: skip

    # A legitimate verdict cannot ignore a pattern the evidence clearly shows.
    if report.verdict == "legitimate":
        if sig.get("micro_burst"):
            problems.append("The recent activity shows a burst of micro-charges (card testing).")
        if sig.get("high_amount"):
            problems.append("The amount is far above anything this card usually spends.")
    # The strongest signal wins: a burst of micro-charges is card testing even when the city
    # is also unusual.
    if report.fraud_type in {"impossible_travel", "high_amount"} and sig.get("micro_burst"):
        problems.append("The micro-charge burst points to card_testing, not "
                        f"{report.fraud_type}.")  # fmt: skip

    for check in ALWAYS_REQUIRED:  # no verdict without the card's normal behaviour and the model
        if check not in sig["checks"]:
            missing.append(check)

    # When the claimed pattern fails and the evidence supports none, say so: in eval run 1 the
    # reporter kept swapping one unsupported pattern for another until it ran out of rounds.
    if report.verdict == "fraud" and problems and not missing and not supported_patterns(sig):
        problems.append(
            "No evidence supports any fraud pattern (no micro-charge burst, no "
            "extreme amount, no flagged purchase away from home). Unless you can "
            "cite evidence for one, the verdict is legitimate with fraud_type none."
        )
    return Critique(approved=not problems and not missing, problems=problems,
                    missing_checks=sorted(set(missing)))  # fmt: skip
