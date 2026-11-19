# ADR 0002: The Critic is rules over the evidence, not another model

**Status:** accepted

## Context
The Critic must be able to reject a report and send work back. A second LLM reviewing the first
shares its blind spots and adds latency on a CPU, and its objections would be hard to test.

## Decision
The Critic computes signals from the evidence data (micro-charge burst, amount against the
card's 95th percentile, purchase outside the home city, model flag) and rejects a report when:
- it cites evidence that does not exist, or none;
- verdict and pattern disagree (legitimate with a pattern, fraud without one);
- the claimed pattern is not supported by the signals, or a check it needs was not run;
- a legitimate verdict ignores a burst or an extreme amount.
Missing checks send the graph back to the Researcher; other problems back to the Reporter with
the reasons. At most three report rounds; a report still rejected is returned with `approved:
false` and the Critic's objections.

## Consequences
- Every rejection is specific and tested (one test per rule).
- The Critic encodes the same signals a rules-only system uses, so the evaluation includes that
  system as a baseline: the agents must justify themselves against it (ADR 0004).
