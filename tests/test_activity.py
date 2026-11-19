"""Card activity facts (never looking past the alert) and the card-activity MCP server."""

from datetime import timedelta

import pytest
from conftest import T0
from mcp.client import Client

from fraud_agents.activity_server import build_server, load_store

pytestmark = pytest.mark.anyio


def test_profile_only_uses_purchases_before_the_alert(store):
    tx = store.transaction("alert-legit")
    p = store.profile(tx.card_id, tx.timestamp)
    assert p.transactions_before == 48  # the 7,000 BRL purchase an hour later is not seen
    assert p.p95_amount < 130 and p.home_city == "Manaus" and p.share_in_home_city == 1.0


def test_recent_activity_counts_micro_charges_in_the_window(store):
    tx = store.transaction("alert-testing")
    recent = store.recent(tx.card_id, tx.timestamp, minutes=5)
    assert recent.count == 5 and recent.micro_charges == 5
    assert [t.transaction_id for t in recent.transactions][:2] == ["burst-4", "burst-3"]
    assert store.recent(tx.card_id, tx.timestamp - timedelta(minutes=10)).count == 0


def test_last_location_and_implied_speed(store):
    tx = store.transaction("alert-away")
    last = store.last_location(tx.card_id, tx.timestamp, tx.city)
    assert last.previous_city == "Sao Paulo" and last.minutes_before == 30.0
    assert last.distance_km == pytest.approx(2689, abs=10)
    assert last.implied_speed_kmh == pytest.approx(2689 * 2, abs=20)


def test_previous_times_for_the_model(store):
    tx = store.transaction("alert-testing")
    times = store.previous_timestamps(tx.card_id, tx.timestamp)
    assert len(times) == 5  # the burst; the last grocery purchase was 30 minutes before T0
    assert all(t >= (T0 - timedelta(minutes=10)).isoformat() for t in times)
    # Outside any burst, the model still gets the previous purchase for time_since_last_tx.
    legit = store.transaction("alert-legit")
    assert store.previous_timestamps(legit.card_id, legit.timestamp) == [
        (T0 - timedelta(minutes=30)).isoformat()
    ]


def test_unknown_transaction(store):
    with pytest.raises(KeyError):
        store.transaction("nope")


async def test_server_tools_and_errors(data_dir):
    async with Client(build_server(load_store(data_dir))) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert tools == {"get_alert", "get_card_profile", "get_recent_activity",
                         "get_last_location"}  # fmt: skip
        alert = await client.call_tool("get_alert", {"transaction_id": "alert-high"})
        assert alert.structured_content["amount"] == 9000.0
        recent = await client.call_tool(
            "get_recent_activity", {"transaction_id": "alert-testing", "minutes": 5}
        )
        assert recent.structured_content["micro_charges"] == 5
        missing = await client.call_tool("get_alert", {"transaction_id": "nope"})
        assert missing.is_error and "No transaction" in missing.content[0].text


def test_find_a_purchase_by_card_and_time(store):
    assert store.find("card_b", T0.isoformat() + "+00:00") == "alert-high"
    assert store.find("card_b", T0) == "alert-high"
    with pytest.raises(KeyError):
        store.find("card_b", T0 + timedelta(seconds=1))
