"""A small card history with one case of each pattern, and a fraud model with the real layout."""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sec_mcp_agent.fraud.model import CITY_COORDS, FraudModel
from xgboost import XGBClassifier

from fraud_agents.activity import ActivityStore

T0 = datetime(2026, 1, 1, 12, 0, 0)
HOME = {"card_a": "Recife", "card_b": "Curitiba", "card_c": "Sao Paulo", "card_d": "Manaus"}
CATEGORIES = ["entertainment", "groceries", "electronics", "travel"]


def history() -> pd.DataFrame:
    """Each card buys groceries at home every 30 minutes for a day, then:
    card_a: 5 micro-charges in 2 minutes, then the alert (card testing)
    card_b: a 9,000 BRL purchase at home (high amount)
    card_c: an 800 BRL purchase in Manaus (away from home)
    card_d: an ordinary purchase at home (legitimate)
    """
    rng = random.Random(1)
    rows = []

    def add(tid: str, card: str, amount: float, category: str, city: str, ts: datetime) -> None:
        rows.append({"transaction_id": tid, "card_id": card, "amount": amount,
                     "merchant": "Shop", "category": category, "city": city,
                     "timestamp": ts.isoformat() + "+00:00"})  # fmt: skip

    for card, home in HOME.items():
        for i in range(48):
            ts = T0 - timedelta(days=1) + timedelta(minutes=30 * i)
            add(f"{card}-{i}", card, round(rng.uniform(20, 120), 2), "groceries", home, ts)
    for i in range(5):
        add(
            f"burst-{i}",
            "card_a",
            1.5 + i,
            "entertainment",
            "Recife",
            T0 + timedelta(seconds=20 * i),
        )
    add("alert-testing", "card_a", 2.0, "entertainment", "Recife", T0 + timedelta(minutes=2))
    add("alert-high", "card_b", 9000.0, "electronics", "Curitiba", T0)
    add("alert-away", "card_c", 800.0, "travel", "Manaus", T0)
    add("alert-legit", "card_d", 60.0, "groceries", "Manaus", T0)
    add("after-alert", "card_d", 7000.0, "electronics", "Recife", T0 + timedelta(hours=1))
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def store() -> ActivityStore:
    return ActivityStore(history(), HOME)


@pytest.fixture(scope="session")
def fraud_model(tmp_path_factory: pytest.TempPathFactory) -> FraudModel:
    """Fraud = a large amount or a purchase far from home (same artifact layout as the real one)."""
    rng = random.Random(0)
    rows = []
    for _ in range(2000):
        kind = rng.choices(["legit", "amount", "away"], weights=[80, 10, 10])[0]
        rows.append({
            "amount": rng.uniform(3000, 9000) if kind == "amount" else rng.uniform(5, 300),
            "distance_km_from_home": 2000.0 if kind == "away" else 0.0,
            "tx_count_last_5min": rng.randint(0, 2), "time_since_last_tx": rng.uniform(30, 3000),
            "hour_of_day": rng.randint(0, 23), "category": rng.choice(CATEGORIES),
            "is_fraud": int(kind != "legit"),
        })  # fmt: skip
    df = pd.DataFrame(rows)
    dummies = pd.get_dummies(df["category"], prefix="category").astype(int)
    cols = ["amount", "distance_km_from_home", "tx_count_last_5min", "time_since_last_tx",
            "hour_of_day", *dummies.columns]  # fmt: skip
    x = pd.concat([df, dummies], axis=1)[cols]
    model = XGBClassifier(n_estimators=30, max_depth=3, random_state=0).fit(x, df["is_fraud"])
    out = tmp_path_factory.mktemp("fraud")
    joblib.dump({"model": model, "feature_cols": cols, "threshold": 0.5}, out / "model.joblib")
    (out / "card_home_city.json").write_text(json.dumps(HOME), encoding="utf-8")
    assert set(HOME.values()) <= set(CITY_COORDS)
    return FraudModel.load(out)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    history().to_parquet(tmp_path / "transactions.parquet")
    (tmp_path / "card_home_city.json").write_text(json.dumps(HOME), encoding="utf-8")
    return tmp_path
