# fraud-investigation-agents

[![tests](https://github.com/DiegoTDDD/fraud-investigation-agents/actions/workflows/tests.yml/badge.svg)](https://github.com/DiegoTDDD/fraud-investigation-agents/actions/workflows/tests.yml)
[![eval](https://github.com/DiegoTDDD/fraud-investigation-agents/actions/workflows/eval.yml/badge.svg)](https://github.com/DiegoTDDD/fraud-investigation-agents/actions/workflows/eval.yml)

A LangGraph team that investigates card-fraud alerts: a **Planner** chooses what to check, a
**Researcher** runs the checks through two MCP servers, a **Reporter** writes a verdict that
cites the evidence, and a **Critic** checks it and sends work back when the evidence does not
support it. Measured on 30 alerts with known answers against the fraud model alone, plain
rules and a single LLM call. Everything runs free: a local 3B model through Ollama on a CPU.

![Demo: the Critic rejects a report the evidence does not support and the graph corrects it](docs/demo.gif)

Two investigations recorded in the evaluation run, not staged: in both, the Critic rejects the
first report with the rule it breaks and the second round reaches the right verdict
([`scripts/render_demo.py`](scripts/render_demo.py); [MP4](docs/demo.mp4)).

## Problem

A fraud model can flag a transaction, but an analyst still has to find out *what* happened
(card testing? an unusual amount? a purchase far from home?) and write it up with evidence.
Agents that do this are easy to demo and hard to trust: they can name a pattern the data does not
show, and nobody checks.

## Solution

The compiled graph (generated from the code into [`docs/graph.mmd`](docs/graph.mmd), and a test
keeps it in sync):

```mermaid
graph TD;
	__start__([start]) --> intake;
	intake --> planner;
	planner --> researcher;
	researcher --> reporter;
	reporter --> critic;
	critic -. approved or out of rounds .-> __end__([end]);
	critic -. problems in the report .-> reporter;
	critic -. evidence missing .-> researcher;
```

| Node | Kind | Does |
| --- | --- | --- |
| intake | code | fetches the alert (evidence E1) |
| planner | LLM | picks hypotheses and lookups (`Plan`, JSON schema) |
| researcher | code + MCP | runs the lookups and states each result as numbered evidence |
| reporter | LLM | verdict, pattern, confidence and cited evidence ids (`Report`, JSON schema) |
| critic | rules | checks the report against the evidence; sends it back with reasons or for more evidence |

Tools, all through MCP:

- **`card-activity`** (this repo): `get_alert`, `get_card_profile`, `get_recent_activity`,
  `get_last_location`. Each one only sees transactions before the alert.
- **`fraud-scoring`** (from [sec-mcp-agent](https://github.com/DiegoTDDD/sec-mcp-agent),
  installed as a pinned dependency): the XGBoost model from
  [realtime-fraud-detection-ml](https://github.com/DiegoTDDD/realtime-fraud-detection-ml) with
  per-feature reasons.

The 30 cases come from the fraud model's test period: 8 high amount, 8 card testing (2 of them
the first charge of a burst, before any pattern exists), 7 impossible travel, 7 legitimate (3
of them outside the home city).

## Results

<!-- EVAL:START -->
Eval run [44537432010](https://github.com/DiegoTDDD/fraud-investigation-agents/actions/runs/44537432010),
qwen2.5:3b on a CPU-only GitHub runner, temperature 0. Full table and one transcript per case
in [`eval/results.md`](eval/results.md).

| System | Verdict right | Verdict and pattern right | Median seconds per case |
| --- | --- | --- | --- |
| Fraud model alone | **97%** | n/a (gives no pattern) | 0 |
| Rules alone (the Critic's signals) | 83% | **80%** | 0 |
| Single agent (all checks, one LLM call, no Critic) | 80% | 47% | 24 |
| **Graph** (Planner, Researcher, Reporter, Critic) | 83% | **77%** | 51 |

What this says:

- **The Critic loop is what makes the language model useful: +30 points** over the same model
  with the same evidence in a single call (47% → 77%). It sent 21 of 30 reports back at least
  once; the rework turned 12 wrong answers right and 3 right answers wrong.
- **The graph matches the rules on 29 of 30 cases** and differs on one (c22, an impossible-travel
  case it labelled high amount). What it adds is the investigation itself: a plan, numbered
  evidence and a report that cites it, checked by the Critic.
- **The fraud model is the better detector** (97% of verdicts). The agents explain and classify;
  they should not replace the model's decision.
- **Hard cases** stay hard for every system: the first charge of a card-testing burst (0 of 2,
  no pattern exists yet) and legitimate purchases away from home (the graph gets 3 of 3, the
  single agent 0 of 3).

By case type (pattern right):

| Case type | Cases | Rules | Single agent | Graph |
| --- | --- | --- | --- | --- |
| card testing | 6 | 6 | 0 | 6 |
| card testing, first charge of a burst (hard) | 2 | 0 | 0 | 0 |
| high amount | 8 | 7 | 8 | 7 |
| impossible travel | 7 | 5 | 2 | 4 |
| legitimate | 4 | 3 | 4 | 3 |
| legitimate outside the home city (hard) | 3 | 3 | 0 | 3 |

### Run 1 → run 2: a regression found in the transcripts and fixed

Run 1 ([`eval/history/run1`](eval/history/run1/results.md)) scored 67% for the graph and only 2
of 7 legitimate cases. The transcripts showed three causes, fixed in
[4920573](https://github.com/DiegoTDDD/fraud-investigation-agents/commit/4920573):

| Cause seen in run 1 | Fix |
| --- | --- |
| The alert was worded "Flagged purchase"; the model called fraud even at model probability 0.000 | Neutral wording ("Purchase under review") and a base-rate reminder in the Reporter prompt |
| The Critic rejected an unsupported pattern without saying that none was supported, so the Reporter swapped one wrong pattern for another until rounds ran out (8 cases) | The Critic now says when the evidence supports no pattern |
| Impossible travel approved for a purchase merely outside the home city (c28), common for legitimate purchases here | Impossible travel needs the fraud model to flag the purchase; the model score is required for every verdict |

| | Run 1 | Run 2 |
| --- | --- | --- |
| Graph, pattern right | 67% | **77%** |
| Legitimate cases right (graph) | 2 of 7 | **6 of 7** |
| Reports still rejected after 3 rounds | 8 | **1** |
<!-- EVAL:END -->

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
bash scripts/build_data.sh          # transactions + fraud model at a pinned commit; checks the cases
ollama pull qwen2.5:3b
python scripts/run_eval.py          # eval/results.md and one transcript per case
```

## Tests

`pytest -q -p anyio`: 34 tests. Card facts never look past the alert; every Critic rule
rejects with a specific reason; the graph runs end to end against both MCP servers with a
scripted model through approval, rejection back to the Reporter, missing evidence back to the
Researcher, the round cap, and invalid JSON; the diagram in `docs/graph.mmd` matches the code.

## Design decisions

- [ADR 0001](docs/adr/0001-models-decide-code-fetches.md): language models decide; code fetches and states the numbers.
- [ADR 0002](docs/adr/0002-rule-based-critic.md): the Critic is rules over the evidence, not another model.
- [ADR 0003](docs/adr/0003-reuse-and-pin.md): reuse the other projects, pinned, and rebuild the data in CI.
- [ADR 0004](docs/adr/0004-measure-against-baselines.md): measure the agents against the model and plain rules.
- [ADR 0005](docs/adr/0005-what-the-data-can-and-cannot-show.md): limits of the synthetic data.

## Limits

- The data is synthetic and dense by design (200 cards at the live stream's 10 purchases per
  second, so 4.2 hours), so purchase counts and travel speed carry little information; the Critic
  does not use them (ADR 0005).
- Part of "impossible travel" cannot be told apart from legitimate travel with this data.
- The Critic checks the verdict, the fraud type and the cited evidence against rules, not that
  the investigation was complete, and it does not read the free-text summary: an approved report
  can still misstate the evidence in prose (case c18's summary mentions micro-charges that
  evidence E3 rules out).

## Stack

Python 3.11, LangGraph, MCP Python SDK 2.x, Pydantic, Ollama (qwen2.5:3b), XGBoost, pandas,
pytest, ruff, GitHub Actions.

## License

MIT
