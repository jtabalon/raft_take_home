import pytest

from main import parse_args


def test_cli_query_mode_requires_query():
    with pytest.raises(SystemExit):
        parse_args([])


def test_cli_query_mode_accepts_query():
    args = parse_args(["--query", "Show all orders in Ohio"])

    assert args.query == "Show all orders in Ohio"
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
