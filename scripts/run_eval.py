"""Investigate the 30 cases with four systems and score them against the known answers.

    model_only    the fraud model's flag (verdict only, no pattern)
    rules_only    the Critic's signals in a fixed order, no language model
    single_agent  every check + one LLM report, no planner, no critic
    graph         planner -> researcher -> reporter -> critic loop

Writes eval/results.md, eval/results.json and eval/transcripts/<case>.md.

    python scripts/run_eval.py --model qwen2.5:3b
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import anyio
from mcp.client import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sec_mcp_agent.agent.llm import OllamaChat  # noqa: E402
from sec_mcp_agent.fraud.model import FraudModel  # noqa: E402
from sec_mcp_agent.fraud.server import build_server as fraud_server  # noqa: E402

from fraud_agents.activity_server import build_server as activity_server  # noqa: E402
from fraud_agents.activity_server import load_store  # noqa: E402
from fraud_agents.baselines import model_only, rules_only  # noqa: E402
from fraud_agents.graph import Investigator, single_agent  # noqa: E402
from fraud_agents.schema import Report  # noqa: E402

SYSTEMS = ["model_only", "rules_only", "single_agent", "graph"]
EVAL = ROOT / "eval"


def score(case: dict, report: Report | None) -> dict[str, bool]:
    if report is None:
        return {"verdict": False, "pattern": False}
    verdict = report.verdict == case["expected_verdict"]
    return {"verdict": verdict, "pattern": verdict and report.fraud_type == case["expected_type"]}


def summarize(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for system in SYSTEMS:
        s = [r["systems"][system] for r in rows]
        out[system] = {
            "verdict_accuracy": round(sum(x["score"]["verdict"] for x in s) / len(s), 3),
            "pattern_accuracy": round(sum(x["score"]["pattern"] for x in s) / len(s), 3),
            "errors": sum(x.get("error") is not None for x in s),
            "median_seconds": round(statistics.median(x["seconds"] for x in s), 2),
        }
    by_kind: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        kind = r["expected_type"] + (" (hard)" if r["difficulty"].startswith("hard") else "")
        for system in SYSTEMS:
            by_kind[kind][system].append(r["systems"][system]["score"]["pattern"])
    out["by_case_type"] = {
        k: {"cases": len(v["graph"]), **{s: sum(v[s]) for s in SYSTEMS}} for k, v in by_kind.items()
    }
    graph = [r["graph"] for r in rows if r["graph"]]
    out["critic_loop"] = {
        "approved_first_round": sum(g["rounds"] == 1 and g["approved"] for g in graph),
        "approved_after_rework": sum(g["rounds"] > 1 and g["approved"] for g in graph),
        "still_rejected": sum(not g["approved"] for g in graph),
        "critic_rejections": sum(sum(not c["approved"] for c in g["critiques"]) for g in graph),
        "extra_checks_requested": sum(
            sum(len(c["missing_checks"]) for c in g["critiques"]) for g in graph
        ),
        "mean_llm_calls": round(statistics.mean(g["llm_calls"] for g in graph), 2) if graph else 0,
        "fixed_by_rework": sum(r["fixed_by_rework"] for r in rows),
        "broken_by_rework": sum(r["broken_by_rework"] for r in rows),
    }
    return out


def write_markdown(summary: dict, rows: list[dict], model: str) -> None:
    pct = lambda v: f"{100 * v:.0f}%"  # noqa: E731
    g = summary["critic_loop"]
    lines = [
        "# Investigation evaluation", "",
        f"{len(rows)} alerts from the fraud model's test period with known answers. Language model "
        f"`{model}` served by Ollama on a CPU-only GitHub Actions runner, temperature 0.", "",
        "**Verdict**: fraud or legitimate. **Pattern**: the verdict and the fraud type "
        "(high_amount, card_testing, impossible_travel, or none for legitimate) are both right.", "",
        "| System | Verdict | Pattern | Median seconds per case | Errors |",
        "| --- | --- | --- | --- | --- |",
    ]  # fmt: skip
    for s in SYSTEMS:
        x = summary[s]
        pattern = "n/a (no pattern)" if s == "model_only" else pct(x["pattern_accuracy"])
        lines.append(f"| {s} | {pct(x['verdict_accuracy'])} | {pattern} | "
                     f"{x['median_seconds']} | {x['errors']} |")  # fmt: skip
    lines += ["", "## Pattern accuracy by case type", "",
              "| Case type | Cases | " + " | ".join(SYSTEMS[1:]) + " |",
              "| --- | --- | " + " | ".join("---" for _ in SYSTEMS[1:]) + " |"]  # fmt: skip
    for kind, v in sorted(summary["by_case_type"].items()):
        lines.append(
            f"| {kind} | {v['cases']} | " + " | ".join(str(v[s]) for s in SYSTEMS[1:]) + " |"
        )
    lines += [
        "", "## The critic loop", "",
        f"- Approved on the first round: {g['approved_first_round']}",
        f"- Approved after the critic sent work back: {g['approved_after_rework']}",
        f"- Still rejected after 3 rounds: {g['still_rejected']}",
        f"- Critic rejections in total: {g['critic_rejections']}; extra checks it requested: "
        f"{g['extra_checks_requested']}",
        f"- Cases the rework turned from wrong to right: {g['fixed_by_rework']}; from right to "
        f"wrong: {g['broken_by_rework']}",
        f"- Mean LLM calls per case: {g['mean_llm_calls']}",
        "", "## Per case", "",
        "| Case | Expected | Difficulty | rules | single agent | graph | Rounds |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]  # fmt: skip
    for r in rows:
        cell = {
            s: r["systems"][s]["label"] + (" ✓" if r["systems"][s]["score"]["pattern"] else " ✗")
            for s in SYSTEMS
        }
        rounds = r["graph"]["rounds"] if r["graph"] else "-"
        lines.append(f"| [{r['id']}](transcripts/{r['id']}.md) | {r['expected_type']} | "
                     f"{r['difficulty']} | {cell['rules_only']} | {cell['single_agent']} | "
                     f"{cell['graph']} | {rounds} |")  # fmt: skip
    (EVAL / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def transcript(case: dict, row: dict) -> str:
    g = row["graph"]
    out = [f"# {case['id']}: expected {case['expected_verdict']} / {case['expected_type']}",
           f"Difficulty: {case['difficulty']}", ""]  # fmt: skip
    if not g:
        return "\n".join([*out, f"Graph failed: {row['systems']['graph'].get('error')}"]) + "\n"
    if g["plan"]:
        out += ["## Plan", f"Hypotheses {g['plan']['hypotheses']}, checks {g['plan']['checks']}",
                f"> {g['plan']['rationale']}", ""]  # fmt: skip
    out += ["## Evidence", *[f"- **{e['id']}** {e['fact']}" for e in g["evidence"]], "",
            "## Trace", *[f"- `{s['node']}` ({s['seconds']} s): {s['detail']}" for s in g["trace"]],
            "", "## Final report", "```json", json.dumps(g["report"], indent=1), "```"]  # fmt: skip
    return "\n".join(out) + "\n"


async def run(args: argparse.Namespace) -> None:
    cases = json.loads((ROOT / "cases" / "cases.json").read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in args.only.split(",")]
    store = load_store(ROOT / "data")
    fraud = FraudModel.load(ROOT / "models" / "fraud")
    llm = OllamaChat(args.model)
    investigator = Investigator(llm, activity_server(store), fraud_server(fraud))
    (EVAL / "transcripts").mkdir(parents=True, exist_ok=True)
    rows = []
    async with Client(activity_server(store)) as act, Client(fraud_server(fraud)) as fr:
        for case in cases:
            row: dict[str, Any] = {**case, "systems": {}, "graph": None}
            tid = store.find(case["card_id"], case["timestamp"])
            t0 = time.perf_counter()
            try:
                report, evidence, _ = await single_agent(llm, act, fr, tid)
                error = None
            except Exception as exc:  # a model timeout fails one case, not the run
                report, error = None, f"{type(exc).__name__}: {exc}"
                from fraud_agents.research import alert_evidence, run_check
                from fraud_agents.schema import ALL_CHECKS

                evidence = [await alert_evidence(act, tid)]
                for c in ALL_CHECKS:
                    evidence.append(await run_check(c, f"E{len(evidence) + 1}", evidence[0].data,
                                                    act, fr))  # fmt: skip
            single_s = time.perf_counter() - t0
            for name, rep, secs, err in [
                ("model_only", model_only(evidence), 0.0, None),
                ("rules_only", rules_only(evidence), 0.0, None),
                ("single_agent", report, single_s, error),
            ]:
                row["systems"][name] = {
                    "label": f"{rep.verdict}/{rep.fraud_type}" if rep else "error",
                    "score": score(case, rep), "seconds": round(secs, 2), "error": err,
                }  # fmt: skip
            t0 = time.perf_counter()
            try:
                inv = await investigator.investigate(tid)
                first = inv.critiques and _first_report(inv)
                row["graph"] = json.loads(inv.model_dump_json())
                final = score(case, inv.report)
                row["systems"]["graph"] = {
                    "label": f"{inv.report.verdict}/{inv.report.fraud_type}", "score": final,
                    "seconds": round(time.perf_counter() - t0, 2), "error": None,
                }  # fmt: skip
                first_ok = score(case, first)["pattern"] if first else final["pattern"]
                row["fixed_by_rework"] = final["pattern"] and not first_ok
                row["broken_by_rework"] = first_ok and not final["pattern"]
            except Exception as exc:
                row["systems"]["graph"] = {"label": "error", "score": score(case, None),
                                           "seconds": round(time.perf_counter() - t0, 2),
                                           "error": f"{type(exc).__name__}: {exc}"}  # fmt: skip
                row["fixed_by_rework"] = row["broken_by_rework"] = False
            rows.append(row)
            (EVAL / "transcripts" / f"{case['id']}.md").write_text(transcript(case, row),
                                                                  encoding="utf-8")  # fmt: skip
            print(case["id"], {s: row["systems"][s]["label"] for s in SYSTEMS},
                  "expected", case["expected_type"], flush=True)  # fmt: skip
    summary = summarize(rows)
    (EVAL / "results.json").write_text(
        json.dumps({"model": llm.name, "summary": summary, "rows": rows}, indent=1, default=str),
        encoding="utf-8",
    )
    write_markdown(summary, rows, llm.name)
    print(json.dumps(summary, indent=1))


def _first_report(inv: Any) -> Report | None:
    """The reporter's first answer, recovered from the trace's first reporter step."""
    step = next((s for s in inv.trace if s.node == "reporter"), None)
    if step is None:
        return None
    verdict_type = step.detail.split(" ")[0]
    verdict, kind = verdict_type.split("/")
    return Report(verdict=verdict, fraud_type=kind, confidence="low", evidence_ids=[], summary="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--only", default=None, help="Comma-separated case ids")
    anyio.run(run, parser.parse_args())


if __name__ == "__main__":
    main()
