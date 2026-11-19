"""The Researcher: runs the planned checks through MCP and states each result as evidence.

The model decides *which* checks to run (Planner) and *what they mean* (Reporter); code does
the lookups and writes the numbers, so evidence never contains a misread figure (ADR 0001).
"""

from __future__ import annotations

from typing import Any

from mcp.client import Client

from fraud_agents.schema import Check, Evidence


async def call(client: Client, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await client.call_tool(tool, args)
    if result.is_error:
        text = " ".join(getattr(c, "text", "") for c in result.content)
        raise RuntimeError(f"{tool} failed: {text}")
    data = result.structured_content or {}
    return data.get("result", data) if isinstance(data, dict) else data


def money(x: float | None) -> str:
    return "n/a" if x is None else f"R$ {x:,.2f}"


async def alert_evidence(activity: Client, transaction_id: str) -> Evidence:
    tx = await call(activity, "get_alert", {"transaction_id": transaction_id})
    fact = (f"Flagged purchase: {money(tx['amount'])} on {tx['category']} in {tx['city']} at "
            f"{tx['timestamp']} (card {tx['card_id']}, merchant {tx['merchant']}).")  # fmt: skip
    return Evidence(id="E1", check="alert", fact=fact, data=tx)


async def previous_times(activity: Client, transaction_id: str) -> list[str]:
    """The card's purchase times the fraud model's velocity features need: the last 10 minutes,
    or at least the previous purchase."""
    recent = await call(activity, "get_recent_activity",
                        {"transaction_id": transaction_id, "minutes": 10})  # fmt: skip
    times = [t["timestamp"] for t in recent["transactions"]]
    if not times:
        last = await call(activity, "get_last_location", {"transaction_id": transaction_id})
        times = [last["previous_timestamp"]] if last["previous_timestamp"] else []
    return sorted(times)


async def run_check(check: Check, evidence_id: str, alert: dict[str, Any], activity: Client,
                    fraud: Client) -> Evidence:  # fmt: skip
    tid = alert["transaction_id"]
    if check == "card_profile":
        p = await call(activity, "get_card_profile", {"transaction_id": tid})
        ratio = alert["amount"] / p["p95_amount"] if p.get("p95_amount") else None
        p["amount_vs_p95"] = round(ratio, 2) if ratio is not None else None
        p["alert_in_home_city"] = alert["city"] == p.get("home_city")
        fact = (
            f"Card profile before the alert ({p['transactions_before']} purchases): home city "
            f"{p['home_city']}, {100 * (p['share_in_home_city'] or 0):.0f}% of purchases there; "
            f"median {money(p['median_amount'])}, 95th percentile {money(p['p95_amount'])}. "
            f"This purchase is {p['amount_vs_p95'] if ratio is not None else 'n/a'}x the 95th "
            f"percentile and {'in' if p['alert_in_home_city'] else 'outside'} the home city."
        )
    elif check == "recent_activity":
        p = await call(activity, "get_recent_activity", {"transaction_id": tid, "minutes": 5})
        p.pop("transactions", None)
        fact = (
            f"In the 5 minutes before the alert the card made {p['count']} purchases, "
            f"{p['micro_charges']} of them micro-charges of at most R$ 10, in "
            f"{', '.join(p['distinct_cities']) or 'no city'} (total {money(p['total_amount'])})."
        )
    elif check == "last_location":
        p = await call(activity, "get_last_location", {"transaction_id": tid})
        if p["previous_city"] is None:
            fact = "The card has no purchase before this one."
        else:
            speed = p["implied_speed_kmh"]
            fact = (
                f"Previous purchase was in {p['previous_city']}, {p['minutes_before']:.1f} minutes "
                f"earlier, {p['distance_km']:.0f} km from {alert['city']}"
                + (f" (implied travel speed {speed:,.0f} km/h)." if speed else ".")
            )
    else:  # model_score, through the fraud-scoring server of sec-mcp-agent
        args = {k: alert[k] for k in ("card_id", "amount", "category", "city")}
        args |= {"timestamp": alert["timestamp"],
                 "previous_timestamps": await previous_times(activity, tid)}  # fmt: skip
        p = await call(fraud, "score_transaction", args)
        reasons = ", ".join(f"{r['feature']} ({r['log_odds']:+.2f})" for r in p["top_reasons"])
        fact = (
            f"Fraud model: probability {p['probability']:.3f} against an alert threshold of "
            f"{p['threshold']:.3f}, so {'flagged' if p['flagged'] else 'not flagged'}; strongest "
            f"factors {reasons}."
        )
    return Evidence(id=evidence_id, check=check, fact=fact, data=p)
