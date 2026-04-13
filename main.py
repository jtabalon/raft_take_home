import argparse
import json
import logging
import os
import sys

from src.agent import OrderAgent
from src.api_client import CustomerAPIClient
from src.env_loader import load_env_files
from src.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse customer order text into deterministic JSON."
    )
    parser.add_argument(
        "--query",
        required=True,
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
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Python logging level. Defaults to LOG_LEVEL or INFO.",
    )
    return parser.parse_args()


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
        response = agent.run(query=args.query, limit=args.limit)
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

    print(json.dumps(response.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
