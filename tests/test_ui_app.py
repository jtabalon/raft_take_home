from html import unescape

from src.models import OrderRecord, validate_model
from src.ui_app import create_app


def make_order(order_id, items, total):
    return validate_model(
        OrderRecord,
        {
            "orderId": str(order_id),
            "buyer": "Test Buyer",
            "city": "Columbus",
            "state": "OH",
            "total": total,
            "items": items,
        },
    )


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


class FakeAgent:
    def __init__(self, payload=None, records=None, error=None):
        self.payload = payload or {"orders": []}
        self.records = records or []
        self.error = error
        self.calls = []

    def run(self, query, limit=None):
        self.calls.append({"query": query, "limit": limit})
        if self.error:
            raise self.error
        return FakeResponse(self.payload)

    def run_with_records(self, query, limit=None):
        self.calls.append({"query": query, "limit": limit, "with_records": True})
        if self.error:
            raise self.error
        return FakeResponse(self.payload), self.records


def test_index_renders_form_and_example_prompt():
    app = create_app(agent_factory=lambda: FakeAgent())

    response = app.test_client().get("/")

    assert response.status_code == 200
    body = unescape(response.get_data(as_text=True))
    assert "Order Parsing Agent" in body
    assert "Show me all orders where the buyer was located in Ohio" in body
    assert "Run query" in body
    assert "Clear prompt" in body
    assert "Predict total" in body
    assert 'id="clear-prompt"' in body
    assert 'id="predict_total_for_items"' in body


def test_query_route_renders_results_table_and_json():
    agent = FakeAgent(
        {
            "orders": [
                {
                    "orderId": "1001",
                    "buyer": "John Davis",
                    "state": "OH",
                    "total": 742.1,
                }
            ]
        }
    )
    app = create_app(agent_factory=lambda: agent)

    response = app.test_client().post(
        "/query",
        data={"query": "Show Ohio orders over 500", "limit": "2"},
    )

    assert response.status_code == 200
    assert agent.calls == [{"query": "Show Ohio orders over 500", "limit": 2}]
    body = unescape(response.get_data(as_text=True))
    assert "<td>1001</td>" in body
    assert "<td>John Davis</td>" in body
    assert '"orderId": "1001"' in body


def test_query_route_renders_regression_prediction_when_requested():
    agent = FakeAgent(
        {
            "orders": [
                {
                    "orderId": "1001",
                    "buyer": "John Davis",
                    "state": "OH",
                    "total": 100.0,
                },
                {
                    "orderId": "1002",
                    "buyer": "Sarah Liu",
                    "state": "TX",
                    "total": 300.0,
                },
            ]
        },
        records=[
            make_order(1001, ["one"], 100.0),
            make_order(1002, ["one", "two", "three"], 300.0),
        ],
    )
    app = create_app(agent_factory=lambda: agent)

    response = app.test_client().post(
        "/query",
        data={
            "query": "Show all orders",
            "predict_total_for_items": "2",
        },
    )

    assert response.status_code == 200
    assert agent.calls == [
        {"query": "Show all orders", "limit": None, "with_records": True}
    ]
    body = unescape(response.get_data(as_text=True))
    assert "Estimated total for 2 items:" in body
    assert "$200.00" in body
    assert "sklearn.linear_model.LinearRegression" in body
    assert '"regression": {' in body
    assert '"predicted_total": 200.0' in body


def test_query_route_rejects_non_positive_regression_item_count():
    app = create_app(agent_factory=lambda: FakeAgent())

    response = app.test_client().post(
        "/query",
        data={
            "query": "Show all orders",
            "predict_total_for_items": "0",
        },
    )

    assert response.status_code == 400
    body = unescape(response.get_data(as_text=True))
    assert "Prediction item count must be at least 1." in body


def test_query_route_renders_agent_errors_without_crashing():
    app = create_app(agent_factory=lambda: FakeAgent(error=RuntimeError("API unavailable")))

    response = app.test_client().post(
        "/query",
        data={"query": "Show all orders"},
    )

    assert response.status_code == 500
    body = unescape(response.get_data(as_text=True))
    assert "API unavailable" in body
    assert '"orders": []' in body
