"""Card activity: the facts an investigator can look up about a card, as of a moment in time.

Everything is computed from transactions strictly *before* the alert, so an investigation never
sees the future (the same rule the fraud model's features follow).
"""

from __future__ import annotations

import bisect
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
from sec_mcp_agent.fraud.model import CITY_COORDS, haversine_km

MICRO_AMOUNT = 10.0  # BRL; card-testing charges are tiny


def to_utc_naive(value: datetime | str) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(UTC).tz_localize(None)
    return ts.to_pydatetime()


class Txn(BaseModel):
    transaction_id: str
    card_id: str
    amount: float
    merchant: str
    category: str
    city: str
    timestamp: datetime


class CardProfile(BaseModel):
    card_id: str
    home_city: str | None = Field(description="Most frequent city in the model's training period")
    transactions_before: int
    median_amount: float | None
    p95_amount: float | None
    share_in_home_city: float | None
    top_categories: list[str]


class RecentActivity(BaseModel):
    card_id: str
    window_minutes: float
    count: int
    micro_charges: int = Field(description=f"Transactions of at most {MICRO_AMOUNT:g} BRL")
    distinct_cities: list[str]
    total_amount: float
    transactions: list[Txn] = Field(description="Most recent first, at most 20")


class LastLocation(BaseModel):
    card_id: str
    previous_city: str | None
    previous_timestamp: datetime | None
    minutes_before: float | None
    distance_km: float | None
    implied_speed_kmh: float | None = Field(
        description="Distance divided by time between the two purchases"
    )


class ActivityStore:
    def __init__(self, transactions: pd.DataFrame, home_city: dict[str, str]) -> None:
        df = transactions.copy()
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], utc=True, format="ISO8601"
        ).dt.tz_localize(None)
        df = df.sort_values(["card_id", "timestamp"], kind="stable").reset_index(drop=True)
        self.df = df
        self.home_city = home_city
        self.by_id = {row.transaction_id: i for i, row in enumerate(df.itertuples())}
        self.cards: dict[str, pd.DataFrame] = {c: g for c, g in df.groupby("card_id", sort=False)}
        self._times = {c: g["timestamp"].tolist() for c, g in self.cards.items()}

    @classmethod
    def load(cls, parquet: str | Path, home_city: dict[str, str]) -> ActivityStore:
        return cls(pd.read_parquet(parquet), home_city)

    def transaction(self, transaction_id: str) -> Txn:
        if transaction_id not in self.by_id:
            raise KeyError(f"No transaction {transaction_id!r}")
        row = self.df.iloc[self.by_id[transaction_id]]
        return _txn(row)

    def find(self, card_id: str, timestamp: datetime | str) -> str:
        """The transaction_id of a card's purchase at an exact time (how cases are stored)."""
        ts = to_utc_naive(timestamp)
        times = self._times.get(card_id, [])
        i = bisect.bisect_left(times, ts)
        if i == len(times) or times[i] != ts:
            raise KeyError(f"No purchase on {card_id} at {ts.isoformat()}")
        return self.cards[card_id].iloc[i]["transaction_id"]

    def _before(self, card_id: str, before: datetime) -> pd.DataFrame:
        if card_id not in self.cards:
            raise KeyError(f"No card {card_id!r}")
        idx = bisect.bisect_left(self._times[card_id], before)
        return self.cards[card_id].iloc[:idx]

    def profile(self, card_id: str, before: datetime) -> CardProfile:
        hist = self._before(card_id, before)
        home = self.home_city.get(card_id)
        if hist.empty:
            return CardProfile(card_id=card_id, home_city=home, transactions_before=0,
                               median_amount=None, p95_amount=None, share_in_home_city=None,
                               top_categories=[])  # fmt: skip
        return CardProfile(
            card_id=card_id,
            home_city=home,
            transactions_before=len(hist),
            median_amount=round(float(hist["amount"].median()), 2),
            p95_amount=round(float(hist["amount"].quantile(0.95)), 2),
            share_in_home_city=round(float((hist["city"] == home).mean()), 3) if home else None,
            top_categories=hist["category"].value_counts().head(3).index.tolist(),
        )

    def recent(self, card_id: str, before: datetime, minutes: float = 5.0) -> RecentActivity:
        hist = self._before(card_id, before)
        window = hist[hist["timestamp"] >= before - pd.Timedelta(minutes=minutes)]
        rows = window.iloc[::-1]
        return RecentActivity(
            card_id=card_id,
            window_minutes=minutes,
            count=len(window),
            micro_charges=int((window["amount"] <= MICRO_AMOUNT).sum()),
            distinct_cities=sorted(window["city"].unique().tolist()),
            total_amount=round(float(window["amount"].sum()), 2),
            transactions=[_txn(r) for _, r in rows.head(20).iterrows()],
        )

    def last_location(self, card_id: str, before: datetime, city: str) -> LastLocation:
        hist = self._before(card_id, before)
        if hist.empty:
            return LastLocation(
                card_id=card_id,
                previous_city=None,
                previous_timestamp=None,
                minutes_before=None,
                distance_km=None,
                implied_speed_kmh=None,
            )
        last = hist.iloc[-1]
        minutes = (before - last["timestamp"]).total_seconds() / 60
        distance = haversine_km(city, last["city"]) if city in CITY_COORDS else None
        speed = distance / (minutes / 60) if distance is not None and minutes > 0 else None
        if distance and minutes == 0:
            speed = float("inf")
        return LastLocation(
            card_id=card_id,
            previous_city=last["city"],
            previous_timestamp=last["timestamp"].to_pydatetime(),
            minutes_before=round(minutes, 2),
            distance_km=round(distance, 1) if distance is not None else None,
            implied_speed_kmh=None if speed is None else round(min(speed, 1e7), 1),
        )

    def previous_timestamps(self, card_id: str, before: datetime, minutes: float = 10) -> list[str]:
        """Earlier purchase times the fraud model needs for its velocity features."""
        hist = self._before(card_id, before)
        window = hist[hist["timestamp"] >= before - pd.Timedelta(minutes=minutes)]
        return [t.isoformat() for t in window["timestamp"]] or (
            [hist["timestamp"].iloc[-1].isoformat()] if not hist.empty else []
        )


def _txn(row: pd.Series) -> Txn:
    return Txn(
        transaction_id=row["transaction_id"],
        card_id=row["card_id"],
        amount=float(row["amount"]),
        merchant=row["merchant"],
        category=row["category"],
        city=row["city"],
        timestamp=row["timestamp"].to_pydatetime(),
    )
