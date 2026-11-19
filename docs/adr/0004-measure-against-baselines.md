# ADR 0004: Measure the agents against the model and against plain rules

**Status:** accepted

## Context
A multi-agent system is only worth its cost if it beats simpler options on the same cases. Before
any LLM ran, the baselines showed the fraud model alone gets 29 of 30 verdicts right on these
cases, and the Critic's rules alone name the right pattern in 24 of 30.

## Decision
Every case is scored for four systems: `model_only`, `rules_only`, `single_agent` (all checks,
one LLM call, no Critic) and `graph`. Two scores: the verdict, and the verdict plus the pattern.
Results are broken down by case type, including deliberately hard cases.

## Consequences
- The question the eval answers is not "can agents detect fraud" (the model already does) but
  whether they name the pattern and justify it from evidence better than rules or a single call.
- If they do not, the results say so; that is a finding, not a failure of the project.
