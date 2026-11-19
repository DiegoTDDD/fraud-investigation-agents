# ADR 0001: Language models decide; code fetches and states the numbers

**Status:** accepted

## Context
In sec-mcp-agent, a 3B model filling tool arguments was the largest source of errors
(eval run 1: most tool calls failed validation; one fraud question dropped a timestamp).
An investigation chains several lookups whose arguments are all known once the alert is known.

## Decision
- The **Planner** (LLM) chooses hypotheses and which lookups to run.
- The **Researcher** is code: it runs those lookups through MCP and writes each result as a
  numbered fact (E1, E2…) with its data attached.
- The **Reporter** (LLM) decides the verdict and the pattern from the evidence and cites ids.
- The **Critic** checks the report against the evidence (ADR 0002).

## Consequences
- Evidence never contains a misread figure or a malformed call; every number in a report can be
  traced to an evidence id and the tool result behind it.
- The models' freedom is where judgement is needed: what to check and what it means.
- Tool use is still real MCP: the Researcher talks to two servers (`card-activity` here and
  `fraud-scoring` from sec-mcp-agent) exactly as an external client would.
