import logging
import sys

import pytest

import main as main_module
from main import parse_args
from src.logging_config import setup_logging


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


class LoggingAgent:
    def __init__(self, api_client, chunk_size):
        self.api_client = api_client
        self.chunk_size = chunk_size

    def run(self, query, limit=None):
        logger = logging.getLogger("order_agent.agent")
        logger.info("fake CLI agent started")
        logger.info("fake CLI agent finished")
        return FakeResponse({"orders": []})


def test_cli_query_mode_requires_query():
    with pytest.raises(SystemExit):
        parse_args([])


def test_cli_query_mode_accepts_query():
    args = parse_args(["--query", "Show all orders in Ohio"])

    assert args.query == "Show all orders in Ohio"
    assert args.ui is False
    assert args.predict_total_for_items is None


def test_cli_accepts_regression_prediction_flag():
    args = parse_args(
        [
            "--query",
            "Show all orders",
            "--predict-total-for-items",
            "2",
        ]
    )

    assert args.predict_total_for_items == 2


def test_cli_rejects_non_positive_regression_item_count():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--query",
                "Show all orders",
                "--predict-total-for-items",
                "0",
            ]
        )


def test_ui_mode_accepts_missing_query():
    args = parse_args(["--ui"])

    assert args.ui is True
    assert args.query is None
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_cli_query_assigns_one_request_id_to_agent_logs(
    monkeypatch,
    caplog,
    capsys,
):
    setup_logging("INFO")
    caplog.set_level(logging.INFO, logger="order_agent.agent")
    monkeypatch.setattr(main_module, "OrderAgent", LoggingAgent)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--query", "Show all orders"],
    )

    result = main_module.main()

    assert result == 0
    assert '"orders": []' in capsys.readouterr().out
    agent_records = [
        record
        for record in caplog.records
        if record.name == "order_agent.agent"
        and record.getMessage().startswith("fake CLI agent")
    ]
    assert [record.getMessage() for record in agent_records] == [
        "fake CLI agent started",
        "fake CLI agent finished",
    ]
    request_ids = {record.request_id for record in agent_records}
    assert len(request_ids) == 1
    assert "-" not in request_ids
