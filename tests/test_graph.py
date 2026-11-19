"""The graph end to end with a scripted model, against both real MCP servers (in memory)."""

import json

import pytest
from mcp.client import Client
from sec_mcp_agent.agent.llm import ChatReply, ScriptedChat
from sec_mcp_agent.fraud.server import build_server as fraud_server

from fraud_agents.activity_server import build_server as activity_server
from fraud_agents.graph import Investigator, mermaid, single_agent

pytestmark = pytest.mark.anyio


def plan(checks, hypotheses=("card_testing",)) -> ChatReply:
    return ChatReply(content=json.dumps({"hypotheses": list(hypotheses), "checks": list(checks),
                                         "rationale": "r"}))  # fmt: skip


def rep(verdict, kind, ids) -> ChatReply:
    return ChatReply(content=json.dumps({"verdict": verdict, "fraud_type": kind,
                                         "confidence": "high", "evidence_ids": ids,
                                         "summary": "See [E2]."}))  # fmt: skip


@pytest.fixture
def investigator(store, fraud_model):
    def make(steps):
        llm = ScriptedChat(steps)
        return Investigator(llm, activity_server(store), fraud_server(fraud_model)), llm

    return make


async def test_approved_on_the_first_round(investigator):
    inv, llm = investigator([plan(["recent_activity", "card_profile", "model_score"]),
                             rep("fraud", "card_testing", ["E1", "E2"])])  # fmt: skip
    result = await inv.investigate("alert-testing")
    assert result.approved and result.rounds == 1 and result.llm_calls == 2
    assert [s.node for s in result.trace] == ["intake", "planner", "researcher", "reporter",
                                              "critic"]  # fmt: skip
    assert [e.check for e in result.evidence] == [
        "alert",
        "recent_activity",
        "card_profile",
        "model_score",
    ]
    assert "5 purchases, 5 of them micro-charges" in result.evidence[1].fact
    assert llm.calls[0]["schema"]["title"] == "Plan"
    assert llm.calls[1]["schema"]["title"] == "Report"


async def test_critic_rejection_goes_back_to_the_reporter_with_the_reason(investigator):
    inv, llm = investigator([
        plan(["card_profile", "recent_activity", "model_score"]),
        rep("legitimate", "none", ["E2"]),  # ignores the burst
        rep("fraud", "card_testing", ["E3"]),
    ])  # fmt: skip
    result = await inv.investigate("alert-testing")
    assert result.approved and result.rounds == 2
    assert not result.critiques[0].approved
    feedback = llm.calls[-1]["messages"][-1]["content"]
    assert "burst of micro-charges" in feedback
    assert [s.node for s in result.trace][-4:] == ["reporter", "critic", "reporter", "critic"]


async def test_missing_check_sends_work_back_to_the_researcher(investigator):
    inv, _ = investigator([
        plan(["model_score"]),  # the planner forgot the profile
        rep("fraud", "high_amount", ["E1"]),
        rep("fraud", "high_amount", ["E1", "E3"]),
    ])  # fmt: skip
    result = await inv.investigate("alert-high")
    assert result.critiques[0].missing_checks == ["card_profile"]
    assert [e.check for e in result.evidence] == ["alert", "model_score", "card_profile"]
    assert "researcher" in [s.node for s in result.trace][5:]
    assert result.approved and result.report.fraud_type == "high_amount"


async def test_rounds_are_capped_and_the_disagreement_is_reported(investigator):
    wrong = rep("legitimate", "none", ["E2"])
    inv, _ = investigator([plan(["card_profile", "recent_activity"]), wrong, wrong, wrong])
    result = await inv.investigate("alert-testing")
    assert result.rounds == 3 and not result.approved and len(result.critiques) == 3


async def test_model_score_goes_through_the_fraud_scoring_server(investigator):
    inv, _ = investigator([plan(["card_profile", "model_score"], ["impossible_travel"]),
                           rep("fraud", "impossible_travel", ["E2", "E3"])])  # fmt: skip
    result = await inv.investigate("alert-away")
    score = result.evidence[2]
    assert score.check == "model_score" and score.data["flagged"] is True
    assert "distance_km_from_home" in score.fact


async def test_single_agent_baseline_runs_every_check_once(store, fraud_model):
    llm = ScriptedChat([rep("fraud", "high_amount", ["E2"])])
    async with Client(activity_server(store)) as act, Client(fraud_server(fraud_model)) as fr:
        report, evidence, calls = await single_agent(llm, act, fr, "alert-high")
    assert [e.check for e in evidence] == ["alert", "card_profile", "recent_activity",
                                           "last_location", "model_score"]  # fmt: skip
    assert report.fraud_type == "high_amount" and calls == 1


async def test_invalid_json_from_the_model_is_retried_once(investigator):
    inv, _ = investigator([ChatReply(content="not json"), plan(["card_profile", "model_score"]),
                             rep("legitimate", "none", ["E2"])])  # fmt: skip
    result = await inv.investigate("alert-legit")
    assert result.approved and result.llm_calls == 3


def test_mermaid_shows_every_node_and_the_loop():
    text = mermaid()
    for node in ["intake", "planner", "researcher", "reporter", "critic"]:
        assert node in text
    assert "critic -.-> researcher" in text and "critic -.-> reporter" in text


def test_the_diagram_in_docs_matches_the_code():
    from pathlib import Path

    committed = (Path(__file__).parents[1] / "docs" / "graph.mmd").read_text(encoding="utf-8")
    assert committed == mermaid(), "regenerate docs/graph.mmd from fraud_agents.graph.mermaid()"
