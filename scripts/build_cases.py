"""Pick the 30 evaluation cases from the fraud model's test period and write cases/cases.json.

Only the last 20% of the timeline is used (the period the fraud model never trained on). The
mix is fixed and includes hard cases on purpose:

    high_amount 8 · card_testing 8 (2 of them the first charge of a burst, before any pattern
    is visible) · impossible_travel 7 · legitimate 7 (3 of them outside the home city)

    python scripts/build_cases.py data/transactions.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 7


def main() -> None:
    df = pd.read_parquet(
        sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "transactions.parquet"
    )
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    cutoff = df["ts"].iloc[int(len(df) * 0.8)]
    home = json.loads((ROOT / "data" / "card_home_city.json").read_text(encoding="utf-8"))
    df["home"] = df["card_id"].map(home)

    # Position of each card-testing charge inside its burst (consecutive charges on one card).
    df = df.sort_values(["card_id", "ts"], kind="stable")
    prev_ct = df.groupby("card_id")["fraud_type"].shift(1).eq("card_testing")
    gap = df.groupby("card_id")["ts"].diff().dt.total_seconds().fillna(1e9)
    start = (df["fraud_type"] == "card_testing") & ~(prev_ct & (gap < 300))
    df["burst_id"] = start.cumsum().where(df["fraud_type"] == "card_testing")
    df["burst_pos"] = df.groupby("burst_id").cumcount().where(df["fraud_type"] == "card_testing")
    df = df.sort_values("ts", kind="stable")
    test = df[df["ts"] >= cutoff]

    def pick(frame: pd.DataFrame, n: int, label: str, difficulty: str) -> list[dict]:
        rows = frame.sample(n, random_state=SEED) if len(frame) >= n else frame
        return [{"transaction_id": r.transaction_id, "expected_verdict":
                 "legitimate" if label == "none" else "fraud", "expected_type": label,
                 "difficulty": difficulty} for r in rows.itertuples()]  # fmt: skip

    cases = (
        pick(test[test["fraud_type"] == "high_amount"], 8, "high_amount", "normal")
        + pick(test[(test["fraud_type"] == "card_testing") & (test["burst_pos"] >= 3)], 6,
               "card_testing", "normal")
        + pick(test[(test["fraud_type"] == "card_testing") & (test["burst_pos"] == 0)], 2,
               "card_testing", "hard: first charge of a burst")
        + pick(test[test["fraud_type"] == "impossible_travel"], 7, "impossible_travel", "normal")
        + pick(test[~test["is_fraud"] & (test["city"] == test["home"])], 4, "none", "normal")
        + pick(test[~test["is_fraud"] & (test["city"] != test["home"])], 3, "none",
               "hard: legitimate purchase outside the home city")
    )  # fmt: skip
    for i, c in enumerate(cases, 1):
        c["id"] = f"c{i:02d}"
    out = ROOT / "cases" / "cases.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([{"id": c.pop("id"), **c} for c in cases], indent=1) + "\n",
                   encoding="utf-8")  # fmt: skip
    print(f"{len(cases)} cases from the test period (after {cutoff}) -> {out}")


if __name__ == "__main__":
    main()
