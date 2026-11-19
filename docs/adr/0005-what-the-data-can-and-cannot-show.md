# ADR 0005: Limits of the synthetic data, and what the Critic ignores because of them

**Status:** accepted

## Context
The transactions come from realtime-fraud-detection-ml's generator. Two properties matter here:
- Its comment describes a 30-day window, but timestamps advance 0.1 s per event, so 150,000
  events cover 4.2 hours. Every card buys every ~20 seconds; "purchases in the last 5 minutes" is
  a dozen or more for any card, and the speed implied between consecutive purchases in different cities
  is in the hundreds of thousands of km/h.
- "Impossible travel" fraud is a purchase in another city for 100–1,500 BRL, while 15% of
  legitimate purchases are also in another city. Part of that pattern cannot be told apart from
  legitimate travel with the data available, which matches the model's 0.65 recall on it.

## Decision
- The Critic uses micro-charges (not purchase counts) for card testing, and never uses implied
  travel speed. `get_last_location` stays available, and its fact is reported, but no rule
  depends on it.
- Hard cases are in the set on purpose: the first charge of a card-testing burst (no pattern yet)
  and legitimate purchases outside the home city.
- The 30-day/4.2-hour mismatch is recorded as an item for realtime-fraud-detection-ml.

## Consequences
Accuracy on impossible travel and on the hard cases has a ceiling set by the data, and the
per-case-type table shows it.
