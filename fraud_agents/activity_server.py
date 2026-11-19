"""MCP server `card-activity`: what an investigator can look up about a card.

Tools  get_alert, get_card_profile, get_recent_activity, get_last_location
All are read-only and only see transactions before the alert's timestamp.

    python -m fraud_agents.activity_server   # FRAUD_AGENTS_DATA=data (parquet + home cities)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

from fraud_agents.activity import (
    ActivityStore,
    CardProfile,
    LastLocation,
    RecentActivity,
    Txn,
)

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def build_server(store: ActivityStore) -> MCPServer:
    server = MCPServer(
        "card-activity",
        instructions="Look up a flagged transaction and the card's activity before it.",
    )

    def alert(transaction_id: str) -> Txn:
        try:
            return store.transaction(transaction_id.strip())
        except KeyError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(annotations=READ_ONLY)
    def get_alert(
        transaction_id: Annotated[str, Field(description="The flagged transaction's id")],
    ) -> Txn:
        """The flagged transaction: card, amount, category, city, merchant and time."""
        return alert(transaction_id)

    @server.tool(annotations=READ_ONLY)
    def get_card_profile(
        transaction_id: Annotated[str, Field(description="The flagged transaction's id")],
    ) -> CardProfile:
        """The card's usual behaviour before the alert: home city, median and 95th-percentile
        amount, share of purchases in the home city, top categories."""
        tx = alert(transaction_id)
        return store.profile(tx.card_id, tx.timestamp)

    @server.tool(annotations=READ_ONLY)
    def get_recent_activity(
        transaction_id: Annotated[str, Field(description="The flagged transaction's id")],
        minutes: Annotated[float, Field(gt=0, le=1440, description="Look-back window")] = 5,
    ) -> RecentActivity:
        """The card's purchases in the minutes before the alert: count, micro-charges
        (at most 10 BRL), cities and the transactions themselves."""
        tx = alert(transaction_id)
        return store.recent(tx.card_id, tx.timestamp, minutes)

    @server.tool(annotations=READ_ONLY)
    def get_last_location(
        transaction_id: Annotated[str, Field(description="The flagged transaction's id")],
    ) -> LastLocation:
        """Where and when the card was last used before the alert, the distance to the alert's
        city and the speed that travel would imply."""
        tx = alert(transaction_id)
        return store.last_location(tx.card_id, tx.timestamp, tx.city)

    return server


def load_store(directory: str | Path) -> ActivityStore:
    directory = Path(directory)
    home = json.loads((directory / "card_home_city.json").read_text(encoding="utf-8"))
    return ActivityStore.load(directory / "transactions.parquet", home)


def main() -> None:
    build_server(load_store(os.environ.get("FRAUD_AGENTS_DATA", "data"))).run("stdio")


if __name__ == "__main__":
    main()
