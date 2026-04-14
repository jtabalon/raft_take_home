from typing import Any, Dict, Optional

from src.agent import OpenRouterLLMClient, OrderAgent
from src.api_client import APIResponseError, CustomerAPIClient
from src.models import OrderQuerySpec, OrderRecord, validate_model


class FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("HTTP error")

    def json(self) -> Dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10):
        return FakeResponse(self.payload)


class FakeLLMClient:
    def __init__(self):
        self.repair_calls = []

    def extract_query_spec(self, query: str) -> OrderQuerySpec:
        return validate_model(OrderQuerySpec, {})

    def repair_order(self, raw_order: str) -> Optional[OrderRecord]:
        self.repair_calls.append(raw_order)
        if "Rachel Kim" not in raw_order:
            return None
        return validate_model(
            OrderRecord,
            {
                "orderId": "1004",
                "buyer": "Rachel Kim",
                "city": "Seattle",
                "state": "WA",
                "total": 89.50,
                "items": ["coffee maker"],
                "source": "llm",
            },
        )


class ConflictingTotalLLMClient(FakeLLMClient):
    def extract_query_spec(self, query: str) -> OrderQuerySpec:
        return validate_model(OrderQuerySpec, {"max_total": 1000})


class FakeOpenRouterLLMClient(OpenRouterLLMClient):
    def __init__(self, responses):
        super().__init__(api_key="test-key")
        self.responses = list(responses)

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        return self.responses.pop(0)


def test_end_to_end_filters_ohio_orders_above_500():
    session = FakeSession(
        {
            "status": "ok",
            "raw_orders": [
                "Order 1005: Buyer=Chris Myers, Location=Cincinnati, OH, Total=$512.00, Items: monitor, desk lamp",
                "Order 1002: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55, Items: headphones",
                "Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99, Items: gaming pc, mouse",
                "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable",
            ],
        }
    )
    api_client = CustomerAPIClient(base_url="http://example.test", session=session)
    agent = OrderAgent(api_client=api_client, llm_client=FakeLLMClient(), chunk_size=2)

    response = agent.run(
        query="Show me all orders where the buyer was located in Ohio and total value was over 500."
    )

    assert response.to_dict() == {
        "orders": [
            {"orderId": "1001", "buyer": "John Davis", "state": "OH", "total": 742.1},
            {"orderId": "1003", "buyer": "Mike Turner", "state": "OH", "total": 1299.99},
            {"orderId": "1005", "buyer": "Chris Myers", "state": "OH", "total": 512.0},
        ]
    }


def test_end_to_end_filters_order_ids_below_threshold():
    session = FakeSession(
        {
            "status": "ok",
            "raw_orders": [
                "Order 999: Buyer=Taylor Reed, Location=Portland, OR, Total=$42.00, Items: cable",
                "Order 1000: Buyer=Jordan Lee, Location=Denver, CO, Total=$55.00, Items: stand",
                "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable",
            ],
        }
    )
    api_client = CustomerAPIClient(base_url="http://example.test", session=session)
    agent = OrderAgent(api_client=api_client, llm_client=FakeLLMClient(), chunk_size=2)

    response = agent.run(query="show me all orders_ids less than 1000")

    assert response.to_dict() == {
        "orders": [
            {"orderId": "999", "buyer": "Taylor Reed", "state": "OR", "total": 42.0},
        ]
    }


def test_order_id_range_ignores_llm_total_range_for_same_threshold():
    session = FakeSession(
        {
            "status": "ok",
            "raw_orders": [
                "Order 999: Buyer=Taylor Reed, Location=Portland, OR, Total=$2000.00, Items: cable",
                "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$42.00, Items: laptop",
            ],
        }
    )
    api_client = CustomerAPIClient(base_url="http://example.test", session=session)
    agent = OrderAgent(
        api_client=api_client,
        llm_client=ConflictingTotalLLMClient(),
        chunk_size=2,
    )

    response = agent.run(query="show me all orders_ids less than 1000")

    assert response.to_dict() == {
        "orders": [
            {"orderId": "999", "buyer": "Taylor Reed", "state": "OR", "total": 2000.0},
        ]
    }


def test_llm_fallback_repairs_unstructured_order():
    session = FakeSession(
        {
            "status": "ok",
            "raw_orders": [
                "Order 1004 Buyer Rachel Kim from Seattle, WA spent $89.50 on coffee maker",
            ],
        }
    )
    llm = FakeLLMClient()
    api_client = CustomerAPIClient(base_url="http://example.test", session=session)
    agent = OrderAgent(api_client=api_client, llm_client=llm, chunk_size=1)

    response = agent.run(query="Show all orders in Washington")

    assert response.to_dict() == {
        "orders": [
            {"orderId": "1004", "buyer": "Rachel Kim", "state": "WA", "total": 89.5}
        ]
    }
    assert llm.repair_calls


def test_customer_api_client_handles_schema_errors():
    session = FakeSession({"status": "ok"})
    api_client = CustomerAPIClient(base_url="http://example.test", session=session)

    try:
        api_client.fetch_orders()
    except APIResponseError as exc:
        assert "orders list" in str(exc)
    else:  # pragma: no cover - defensive assertion.
        raise AssertionError("Expected APIResponseError")


def test_openrouter_query_extraction_accepts_null_order_ids():
    llm = FakeOpenRouterLLMClient(
        [
            """
            {
              "state": "OH",
              "min_total": 500,
              "max_total": null,
              "order_ids": null,
              "buyer_name": null,
              "reason": "Ohio orders over 500"
            }
            """
        ]
    )

    spec = llm.extract_query_spec(
        "Show me all orders where the buyer was located in Ohio and total value was over 500"
    )

    assert spec.state == "OH"
    assert spec.min_total == 500
    assert spec.order_ids == []


def test_openrouter_query_extraction_accepts_order_id_range_fields():
    llm = FakeOpenRouterLLMClient(
        [
            """
            {
              "state": null,
              "min_total": null,
              "max_total": null,
              "order_ids": null,
              "min_order_id": null,
              "max_order_id": 1000,
              "buyer_name": null,
              "reason": "Order IDs below 1000"
            }
            """
        ]
    )

    spec = llm.extract_query_spec("show me all orders_ids less than 1000")

    assert spec.max_order_id == 1000
    assert spec.max_total is None
    assert spec.order_ids == []


def test_openrouter_order_repair_accepts_null_items():
    llm = FakeOpenRouterLLMClient(
        [
            """
            {
              "orderId": "1004",
              "buyer": "Rachel Kim",
              "city": "Seattle",
              "state": "WA",
              "total": 89.50,
              "items": null,
              "reason": null
            }
            """
        ]
    )

    order = llm.repair_order(
        "Order 1004 Buyer Rachel Kim from Seattle, WA spent $89.50"
    )

    assert order is not None
    assert order.items == []
