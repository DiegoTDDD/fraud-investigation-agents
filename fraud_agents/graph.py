"""The investigation graph: intake -> planner -> researcher -> reporter -> critic -> (loop | end).

intake      code   fetch the alert (E1)
planner     LLM    pick hypotheses and checks                   -> Plan
researcher  code   run the checks through MCP, state evidence   -> E2, E3, ...
reporter    LLM    verdict, pattern, confidence, cited evidence -> Report
critic      rules  check the report against the evidence        -> Critique
  rejected with missing checks -> researcher
  rejected otherwise           -> reporter (with the problems)
  approved, or out of rounds   -> end
"""

from __future__ import annotations

import json
import time
from contextlib import AsyncExitStack
from typing import Any, TypedDict, TypeVar

from langgraph.graph import END, START, StateGraph
from mcp.client import Client
from pydantic import BaseModel, ValidationError
from sec_mcp_agent.agent.llm import ChatModel, Message

from fraud_agents.critic import critique
from fraud_agents.research import alert_evidence, run_check
from fraud_agents.schema import (
    ALL_CHECKS,
    Check,
    Critique,
    Evidence,
    Investigation,
    Plan,
    Report,
    Step,
)

MAX_ROUNDS = 3  # reporter attempts; the critic can send work back twice

PLANNER_PROMPT = """You are the planner of a card-fraud investigation team. Given a flagged \
purchase, choose which fraud patterns to test and which lookups the researcher should run.

Patterns: high_amount (far above what the card usually spends), card_testing (a burst of tiny \
charges), impossible_travel (purchase far from where the card is normally used).
Lookups: card_profile (home city, usual amounts), recent_activity (purchases in the last \
5 minutes), last_location (where the card was last used), model_score (the fraud model)."""

REPORTER_PROMPT = """You are the reporter of a card-fraud investigation team. Decide from the \
evidence alone whether the flagged purchase is fraud, and which pattern.

- card_testing: several micro-charges (at most R$ 10) within minutes.
- high_amount: an amount several times above the card's usual 95th-percentile purchase.
- impossible_travel: a purchase outside the card's home city that the rest of the evidence \
makes implausible.
- legitimate: none of these is supported; use fraud_type "none".
Cite the evidence ids you rely on, in evidence_ids and in the summary like [E2]. Do not invent \
numbers that are not in the evidence."""

M = TypeVar("M", bound=BaseModel)


class State(TypedDict, total=False):
    transaction_id: str
    alert: dict[str, Any]
    plan: Plan
    evidence: list[Evidence]
    pending: list[Check]
    report: Report
    critiques: list[Critique]
    rounds: int
    trace: list[Step]
    llm_calls: int


def ask_json(llm: ChatModel, messages: list[Message], model: type[M]) -> tuple[M, int]:
    """Schema-constrained reply, one retry on invalid JSON. Returns the object and calls made."""
    schema = model.model_json_schema()
    reply = llm.chat(messages, schema=schema)
    try:
        return model.model_validate_json(reply.content), 1
    except ValidationError as exc:
        retry = [*messages, {"role": "assistant", "content": reply.content},
                 {"role": "user", "content": f"Invalid: {exc.errors()[0]['msg']}. Answer again "
                                             "with valid JSON for the schema."}]  # fmt: skip
        reply = llm.chat(retry, schema=schema)
        return model.model_validate_json(reply.content), 2


def evidence_text(evidence: list[Evidence]) -> str:
    return "\n".join(f"[{e.id}] {e.fact}" for e in evidence)


def build_graph(llm: ChatModel, activity: Client, fraud: Client) -> Any:
    def step(state: State, node: str, t0: float, detail: str) -> list[Step]:
        return [
            *state.get("trace", []),
            Step(node=node, seconds=round(time.perf_counter() - t0, 2), detail=detail),
        ]

    async def intake(state: State) -> State:
        t0 = time.perf_counter()
        e1 = await alert_evidence(activity, state["transaction_id"])
        return {"alert": e1.data, "evidence": [e1], "rounds": 0, "critiques": [],
                "llm_calls": 0, "trace": step(state, "intake", t0, e1.fact)}  # fmt: skip

    async def planner(state: State) -> State:
        t0 = time.perf_counter()
        messages = [{"role": "system", "content": PLANNER_PROMPT},
                    {"role": "user", "content": evidence_text(state["evidence"])}]  # fmt: skip
        plan, calls = ask_json(llm, messages, Plan)
        checks = list(dict.fromkeys(plan.checks))  # keep order, drop repeats
        detail = f"hypotheses={plan.hypotheses} checks={checks}"
        return {"plan": plan, "pending": checks, "llm_calls": state["llm_calls"] + calls,
                "trace": step(state, "planner", t0, detail)}  # fmt: skip

    async def researcher(state: State) -> State:
        t0 = time.perf_counter()
        evidence = list(state["evidence"])
        done = {e.check for e in evidence}
        added = []
        for check in state.get("pending", []):
            if check in done:
                continue
            e = await run_check(check, f"E{len(evidence) + 1}", state["alert"], activity, fraud)
            evidence.append(e)
            done.add(check)
            added.append(e.id)
        return {"evidence": evidence, "pending": [],
                "trace": step(state, "researcher", t0, f"added {added or 'nothing'}")}  # fmt: skip

    async def reporter(state: State) -> State:
        t0 = time.perf_counter()
        messages: list[Message] = [
            {"role": "system", "content": REPORTER_PROMPT},
            {"role": "user", "content": evidence_text(state["evidence"])},
        ]
        if state["critiques"] and "report" in state:
            last = state["critiques"][-1]
            messages += [
                {"role": "assistant", "content": state["report"].model_dump_json()},
                {"role": "user", "content": "The critic rejected this report. "
                 + (" ".join(last.problems) or "It asked for more evidence, now added above.")
                 + " Write a corrected report from all the evidence."},
            ]  # fmt: skip
        report, calls = ask_json(llm, messages, Report)
        detail = f"{report.verdict}/{report.fraud_type} ({report.confidence}) {report.evidence_ids}"
        return {"report": report, "rounds": state["rounds"] + 1,
                "llm_calls": state["llm_calls"] + calls,
                "trace": step(state, "reporter", t0, detail)}  # fmt: skip

    async def critic(state: State) -> State:
        t0 = time.perf_counter()
        c = critique(state["report"], state["evidence"])
        detail = "approved" if c.approved else f"rejected: {c.problems} missing={c.missing_checks}"
        return {"critiques": [*state["critiques"], c], "pending": c.missing_checks,
                "trace": step(state, "critic", t0, detail)}  # fmt: skip

    def after_critic(state: State) -> str:
        last = state["critiques"][-1]
        if last.approved or state["rounds"] >= MAX_ROUNDS:
            return END
        return "researcher" if last.missing_checks else "reporter"

    g = StateGraph(State)
    for name, fn in [("intake", intake), ("planner", planner), ("researcher", researcher),
                     ("reporter", reporter), ("critic", critic)]:  # fmt: skip
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_edge("intake", "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "reporter")
    g.add_edge("reporter", "critic")
    g.add_conditional_edges("critic", after_critic, ["researcher", "reporter", END])
    return g.compile()


class Investigator:
    """Opens the MCP connections and runs one investigation per alert."""

    def __init__(self, llm: ChatModel, activity_target: Any, fraud_target: Any) -> None:
        self.llm = llm
        self.activity_target = activity_target
        self.fraud_target = fraud_target

    async def investigate(self, transaction_id: str) -> Investigation:
        t0 = time.perf_counter()
        async with AsyncExitStack() as stack:
            activity = await stack.enter_async_context(Client(self.activity_target))
            fraud = await stack.enter_async_context(Client(self.fraud_target))
            graph = build_graph(self.llm, activity, fraud)
            state = await graph.ainvoke({"transaction_id": transaction_id},
                                        {"recursion_limit": 25})  # fmt: skip
        return Investigation(
            transaction_id=transaction_id, plan=state.get("plan"), evidence=state["evidence"],
            report=state["report"], critiques=state["critiques"],
            approved=state["critiques"][-1].approved, rounds=state["rounds"],
            trace=state["trace"], llm_calls=state["llm_calls"],
            seconds=round(time.perf_counter() - t0, 2),
        )  # fmt: skip


async def single_agent(llm: ChatModel, activity: Client, fraud: Client,
                       transaction_id: str) -> tuple[Report, list[Evidence], int]:  # fmt: skip
    """Baseline: all checks, one reporter call, no planner and no critic."""
    evidence = [await alert_evidence(activity, transaction_id)]
    for check in ALL_CHECKS:
        evidence.append(await run_check(check, f"E{len(evidence) + 1}", evidence[0].data,
                                        activity, fraud))  # fmt: skip
    messages = [{"role": "system", "content": REPORTER_PROMPT},
                {"role": "user", "content": evidence_text(evidence)}]  # fmt: skip
    report, calls = ask_json(llm, messages, Report)
    return report, evidence, calls


def mermaid() -> str:
    """The compiled graph as Mermaid, for the README."""

    class _Null:
        name = "null"

        def chat(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - never called
            raise NotImplementedError

    return build_graph(_Null(), None, None).get_graph().draw_mermaid()  # type: ignore[arg-type]


def to_json(inv: Investigation) -> str:
    return json.dumps(inv.model_dump(), indent=2, default=str)
