"""Every Critic rule, on hand-built evidence."""

import pytest

from fraud_agents.baselines import rules_only
from fraud_agents.critic import critique, signals
from fraud_agents.schema import Evidence, Report


def ev(amount=60.0, city="Recife", home="Recife", p95=100.0, micro=0, flagged=False,
       checks=("card_profile", "recent_activity", "model_score")) -> list[Evidence]:  # fmt: skip
    out = [Evidence(id="E1", check="alert", fact="alert", data={"amount": amount, "city": city})]
    data = {
        "card_profile": {"home_city": home, "p95_amount": p95},
        "recent_activity": {"micro_charges": micro, "count": micro},
        "model_score": {"flagged": flagged, "probability": 0.9 if flagged else 0.01},
    }
    for c in checks:
        out.append(Evidence(id=f"E{len(out) + 1}", check=c, fact=c, data=data[c]))
    return out


def report(verdict="fraud", kind="card_testing", ids=("E1", "E3")) -> Report:
    return Report(verdict=verdict, fraud_type=kind, confidence="high", evidence_ids=list(ids),
                  summary="s")  # fmt: skip


def test_supported_reports_are_approved():
    assert critique(report(), ev(micro=5)).approved
    assert critique(report(kind="high_amount", ids=["E2"]), ev(amount=9000)).approved
    assert critique(
        report(kind="impossible_travel", ids=["E2"]), ev(city="Manaus", flagged=True)
    ).approved
    assert critique(report("legitimate", "none", ["E2"]), ev()).approved


@pytest.mark.parametrize(("rep", "evidence", "phrase"), [
    (report(ids=["E9"]), ev(micro=5), "does not exist"),
    (report(ids=[]), ev(micro=5), "cites no evidence"),
    (report("legitimate", "card_testing"), ev(micro=5), "must have fraud_type 'none'"),
    (report("fraud", "none"), ev(), "must name the pattern"),
    (report(), ev(micro=1), "micro-charges"),
    (report(kind="high_amount"), ev(amount=250), "95th-percentile"),
    (report(kind="high_amount"), ev(amount=1500, p95=100), "95th-percentile"),  # under the floor
    (report(kind="impossible_travel"), ev(), "outside the home city"),
    (report(kind="impossible_travel"), ev(city="Manaus"), "does not flag it"),
    (report(kind="high_amount"), ev(amount=250), "No evidence supports any fraud pattern"),
    (report("legitimate", "none"), ev(micro=4), "burst of micro-charges"),
    (report("legitimate", "none"), ev(amount=9000), "far above"),
    (report(kind="impossible_travel"), ev(city="Manaus", micro=4), "points to card_testing"),
])  # fmt: skip
def test_each_rule_rejects_with_a_specific_reason(rep, evidence, phrase):
    c = critique(rep, evidence)
    assert not c.approved
    assert any(phrase in p for p in c.problems), c.problems


def test_missing_checks_are_requested():
    c = critique(report(), ev(micro=5, checks=("model_score",)))
    assert not c.approved and c.missing_checks == ["card_profile", "recent_activity"]
    c = critique(report("legitimate", "none", ["E2"]), ev(checks=("model_score",)))
    assert c.missing_checks == ["card_profile"]  # no verdict without the card's normal behaviour


def test_signals_are_computed_from_data_not_words():
    s = signals(ev(amount=9000, city="Manaus", micro=3, flagged=True))
    assert s["micro_burst"] and s["high_amount"] and s["away_from_home"] and s["model_flagged"]


def test_rules_baseline_order():
    assert rules_only(ev(micro=5, amount=9000)).fraud_type == "card_testing"
    assert rules_only(ev(amount=9000)).fraud_type == "high_amount"
    assert rules_only(ev(city="Manaus", flagged=True)).fraud_type == "impossible_travel"
    assert rules_only(ev(city="Manaus")).verdict == "legitimate"  # away but not flagged


def test_the_model_score_is_always_required():
    c = critique(report("legitimate", "none", ["E2"]), ev(checks=("card_profile",)))
    assert c.missing_checks == ["model_score"]
