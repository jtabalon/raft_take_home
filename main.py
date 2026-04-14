import argparse
import json
import logging
import os
import sys
from typing import Optional, Sequence

from src.agent import OrderAgent
from src.api_client import CustomerAPIClient
from src.env_loader import load_env_files
from src.logging_config import setup_logging


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a whole number") from exc

    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse customer order text into deterministic JSON."
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Natural language query to run against the order API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of records to request from the customer API.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Maximum number of raw orders to process in one chunk.",
    )
    parser.add_argument(
        "--predict-total-for-items",
        type=positive_int,
        default=None,
        help=(
            "Train a tiny sklearn LinearRegression baseline on parsed orders and "
            "predict total for this item count."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Python logging level. Defaults to LOG_LEVEL or INFO.",
    )
    args = parser.parse_args(argv)
    if not args.query:
        parser.error("--query is required")
    return args


def main() -> int:
    load_env_files((".env.local", ".env"))
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("order_agent.cli")

    api_base_url = os.getenv("ORDER_API_BASE_URL", "http://localhost:5001")
    api_client = CustomerAPIClient(base_url=api_base_url)
    agent = OrderAgent(
        api_client=api_client,
        chunk_size=args.chunk_size,
    )

    try:
        if args.predict_total_for_items is None:
            response = agent.run(query=args.query, limit=args.limit)
            payload = response.to_dict()
        else:
            from src.regression import (
                InsufficientRegressionData,
                predict_total_for_item_count,
                regression_error_response,
            )

            response, parsed_orders = agent.run_with_records(
                query=args.query,
                limit=args.limit,
            )
            payload = response.to_dict()
            try:
                regression = predict_total_for_item_count(
                    parsed_orders,
                    args.predict_total_for_items,
                )
            except InsufficientRegressionData as exc:
                regression = regression_error_response(exc)
            payload["regression"] = regression.to_dict()
    except Exception as exc:  # pragma: no cover - exercised via CLI only.
        logger.exception("Failed to process query: %s", exc)
        print(
            json.dumps(
                {
                    "orders": [],
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
