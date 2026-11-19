# ADR 0003: Reuse the other projects, pinned, and rebuild the data in CI

**Status:** accepted

## Context
The investigation needs the fraud model, its scoring server, the chat-model interface and the
transaction history. All already exist in other projects.

## Decision
- `sec-mcp-agent` is a dependency pinned to a commit: its `fraud-scoring` MCP server scores
  transactions and its `ChatModel` interface (Ollama or scripted) drives the LLM nodes.
- `scripts/build_data.sh` generates the transactions and trains the model with
  realtime-fraud-detection-ml's own code at a pinned commit, then rebuilds the 30 cases and
  fails if they differ from `cases/cases.json`.
- Cases come only from the model's test period (the last 20% of the timeline).

## Consequences
- Nothing is copied between projects, and every number can be reproduced from three commits.
- Moving a pin is a deliberate change that the case check will surface.
