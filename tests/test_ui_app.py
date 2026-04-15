import logging
from html import unescape

from src.logging_config import setup_logging
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


class LoggingFakeAgent(FakeAgent):
    def run(self, query, limit=None):
        logger = logging.getLogger("order_agent.agent")
        logger.info("fake UI agent started")
        response = super().run(query, limit)
        logger.info("fake UI agent finished")
        return response


def test_index_renders_form_and_example_prompt():
    app = create_app(chunk_size=12, agent_factory=lambda chunk_size: FakeAgent())

    response = app.test_client().get("/")

    assert response.status_code == 200
    body = unescape(response.get_data(as_text=True))
    assert "Order Parsing Agent" in body
    assert "Show me all orders where the buyer was located in Ohio" in body
    assert "Run query" in body
    assert "Clear prompt" in body
    assert "Predict total" in body
    assert "Chunk size" in body
    assert 'id="clear-prompt"' in body
    assert 'id="predict_total_for_items"' in body
    assert 'id="chunk_size"' in body
    assert 'value="12"' in body


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
    app = create_app(agent_factory=lambda chunk_size: agent)

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


def test_query_route_uses_submitted_chunk_size():
    agent = FakeAgent({"orders": []})
    selected_chunk_sizes = []
    app = create_app(
        agent_factory=lambda chunk_size: selected_chunk_sizes.append(chunk_size) or agent
    )

    response = app.test_client().post(
        "/query",
        data={
            "query": "Show all orders",
            "chunk_size": "3",
        },
    )

    assert response.status_code == 200
    assert selected_chunk_sizes == [3]
    body = unescape(response.get_data(as_text=True))
    assert 'id="chunk_size" name="chunk_size" type="number" min="1" value="3"' in body


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
    app = create_app(agent_factory=lambda chunk_size: agent)

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
    app = create_app(agent_factory=lambda chunk_size: FakeAgent())

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


def test_query_route_rejects_non_integer_chunk_size():
    app = create_app(agent_factory=lambda chunk_size: FakeAgent())

    response = app.test_client().post(
        "/query",
        data={
            "query": "Show all orders",
            "chunk_size": "many",
        },
    )

    assert response.status_code == 400
    body = unescape(response.get_data(as_text=True))
    assert "Chunk size must be a whole number." in body
    assert 'id="chunk_size" name="chunk_size" type="number" min="1" value="many"' in body


def test_query_route_rejects_non_positive_chunk_size():
    app = create_app(agent_factory=lambda chunk_size: FakeAgent())

    response = app.test_client().post(
        "/query",
        data={
            "query": "Show all orders",
            "chunk_size": "0",
        },
    )

    assert response.status_code == 400
    body = unescape(response.get_data(as_text=True))
    assert "Chunk size must be at least 1." in body
    assert 'id="chunk_size" name="chunk_size" type="number" min="1" value="0"' in body


def test_query_route_renders_agent_errors_without_crashing():
    app = create_app(
        agent_factory=lambda chunk_size: FakeAgent(error=RuntimeError("API unavailable"))
    )

    response = app.test_client().post(
        "/query",
        data={"query": "Show all orders"},
    )

    assert response.status_code == 500
    body = unescape(response.get_data(as_text=True))
    assert "API unavailable" in body
    assert '"orders": []' in body


def test_query_route_assigns_one_request_id_to_agent_logs(caplog):
    setup_logging("INFO")
    caplog.set_level(logging.INFO, logger="order_agent.agent")
    app = create_app(agent_factory=lambda chunk_size: LoggingFakeAgent())

    response = app.test_client().post(
        "/query",
        data={"query": "Show all orders"},
    )

    assert response.status_code == 200
    agent_records = [
        record
        for record in caplog.records
        if record.name == "order_agent.agent"
        and record.getMessage().startswith("fake UI agent")
    ]
    assert [record.getMessage() for record in agent_records] == [
        "fake UI agent started",
        "fake UI agent finished",
    ]
    request_ids = {record.request_id for record in agent_records}
    assert len(request_ids) == 1
    assert "-" not in request_ids
