import json
import logging
from typing import Callable, Optional, Protocol

from flask import Flask, render_template_string, request

from src.agent import OrderAgent
from src.api_client import CustomerAPIClient
from src.logging_config import new_request_id, request_id_context
from src.regression import (
    InsufficientRegressionData,
    predict_total_for_item_count,
    regression_error_response,
)


class AgentRunner(Protocol):
    def run(self, query: str, limit: Optional[int] = None):
        ...

    def run_with_records(self, query: str, limit: Optional[int] = None):
        ...


EXAMPLE_QUERIES = [
    "Show me all orders where the buyer was located in Ohio and total value was over 500",
    "Show all orders in Texas",
    "Find order 1003",
]


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Order Parsing Agent</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172033;
      --muted: #5d687a;
      --line: #d6dde8;
      --panel: #ffffff;
      --surface: #f5f7fb;
      --accent: #0f766e;
      --accent-dark: #0b5f59;
      --danger: #b42318;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--surface);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.5;
    }

    main {
      width: min(1040px, calc(100% - 32px));
      margin: 0 auto;
      padding: 40px 0;
    }

    h1 {
      margin: 0 0 8px;
      font-size: 2rem;
      line-height: 1.15;
    }

    p {
      margin: 0;
      color: var(--muted);
    }

    form {
      margin-top: 28px;
      padding: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }

    label {
      display: block;
      margin-bottom: 8px;
      font-weight: 700;
    }

    textarea,
    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--ink);
      font: inherit;
      padding: 12px;
    }

    textarea {
      min-height: 108px;
      resize: vertical;
    }

    .form-row {
      display: grid;
      grid-template-columns:
        minmax(120px, 180px)
        minmax(120px, 180px)
        minmax(180px, 240px)
        auto;
      gap: 16px;
      align-items: start;
      margin-top: 16px;
    }

    .help-text {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 0.88rem;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-top: 32px;
    }

    button,
    .secondary-button {
      width: fit-content;
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: white;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
      padding: 12px 18px;
    }

    button:hover,
    .secondary-button:hover {
      background: var(--accent-dark);
    }

    .secondary-button {
      background: white;
      color: var(--accent);
    }

    .secondary-button:hover {
      color: white;
    }

    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }

    .example {
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      padding: 6px 10px;
      font-size: 0.9rem;
    }

    .results {
      margin-top: 28px;
    }

    .notice,
    .error {
      margin-top: 20px;
      padding: 14px 16px;
      border-radius: 8px;
      background: var(--panel);
      border: 1px solid var(--line);
    }

    .error {
      border-color: #f2b8b5;
      color: var(--danger);
    }

    .table-wrap {
      overflow-x: auto;
      margin-top: 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }

    .regression-summary {
      margin-top: 16px;
      padding: 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }

    .prediction {
      margin: 0 0 8px;
      color: var(--ink);
      font-size: 1.2rem;
      font-weight: 700;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 560px;
    }

    th,
    td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }

    tr:last-child td {
      border-bottom: 0;
    }

    th {
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
    }

    pre {
      overflow-x: auto;
      margin: 16px 0 0;
      padding: 16px;
      background: #111827;
      border-radius: 8px;
      color: #f9fafb;
      font-size: 0.94rem;
    }

    @media (max-width: 680px) {
      main {
        width: min(100% - 24px, 1040px);
        padding: 24px 0;
      }

      h1 {
        font-size: 1.65rem;
      }

      form {
        padding: 18px;
      }

      .form-row {
        grid-template-columns: 1fr;
      }

      .actions {
        margin-top: 0;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Order Parsing Agent</h1>
      <p>Ask for customer orders in plain language and get deterministic JSON back.</p>
    </header>

    <form method="post" action="/query">
      <label for="query">Prompt</label>
      <textarea id="query" name="query" required>{{ query }}</textarea>
      <div class="examples" aria-label="Example prompts">
        {% for example in examples %}
          <span class="example">{{ example }}</span>
        {% endfor %}
      </div>

      <div class="form-row">
        <div>
          <label for="limit">Record limit</label>
          <input id="limit" name="limit" type="number" min="1" value="{{ limit or '' }}" placeholder="Optional">
        </div>
        <div>
          <label for="chunk_size">Chunk size</label>
          <input id="chunk_size" name="chunk_size" type="number" min="1" value="{{ chunk_size }}" data-default-value="{{ default_chunk_size }}">
          <span class="help-text">How many raw orders to parse per batch.</span>
        </div>
        <div>
          <label for="predict_total_for_items">Predict total</label>
          <input id="predict_total_for_items" name="predict_total_for_items" type="number" min="1" value="{{ predict_total_for_items or '' }}" placeholder="Item count">
          <span class="help-text">Optional sklearn baseline</span>
        </div>
        <div class="actions">
          <button type="submit">Run query</button>
          <button class="secondary-button" type="button" id="clear-prompt">Clear prompt</button>
        </div>
      </div>
    </form>

    {% if error %}
      <div class="error" role="alert">{{ error }}</div>
    {% endif %}

    {% if has_result %}
      <section class="results" aria-live="polite">
        <h2>Results</h2>
        {% if orders %}
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Order ID</th>
                  <th>Buyer</th>
                  <th>State</th>
                  <th>Total</th>
                </tr>
              </thead>
              <tbody>
                {% for order in orders %}
                  <tr>
                    <td>{{ order.orderId }}</td>
                    <td>{{ order.buyer }}</td>
                    <td>{{ order.state }}</td>
                    <td>${{ "%.2f"|format(order.total) }}</td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        {% else %}
          <div class="notice">No orders matched that query.</div>
        {% endif %}

        {% if regression %}
          <h2>Regression</h2>
          <div class="regression-summary">
            {% if regression.error %}
              <p>{{ regression.error }}</p>
            {% elif regression.prediction %}
              <p class="prediction">
                Estimated total for {{ regression.prediction.item_count }} items:
                ${{ "%.2f"|format(regression.prediction.predicted_total) }}
              </p>
              {% if regression.model %}
                <p>
                  {{ regression.model.type }} using {{ regression.model.feature }}
                  across {{ regression.model.sample_count }} parsed orders.
                </p>
              {% endif %}
            {% endif %}
            {% if regression.note %}
              <p>{{ regression.note }}</p>
            {% endif %}
          </div>
        {% endif %}

        <h2>JSON</h2>
        <pre>{{ response_json }}</pre>
      </section>
    {% endif %}
  </main>
  <script>
    const clearButton = document.getElementById("clear-prompt");
    clearButton.addEventListener("click", () => {
      document.getElementById("query").value = "";
      document.getElementById("limit").value = "";
      const chunkSizeInput = document.getElementById("chunk_size");
      chunkSizeInput.value = chunkSizeInput.dataset.defaultValue;
      document.getElementById("predict_total_for_items").value = "";
      document.getElementById("query").focus();
    });
  </script>
</body>
</html>
"""


def create_app(
    api_base_url: str = "http://localhost:5001",
    chunk_size: int = 50,
    agent_factory: Optional[Callable[[int], AgentRunner]] = None,
) -> Flask:
    app = Flask(__name__)
    logger = logging.getLogger("order_agent.ui")
    default_chunk_size = max(1, chunk_size)

    def build_agent(selected_chunk_size: int) -> AgentRunner:
        if agent_factory is not None:
            return agent_factory(selected_chunk_size)
        api_client = CustomerAPIClient(base_url=api_base_url)
        return OrderAgent(api_client=api_client, chunk_size=selected_chunk_size)

    def render_page(
        query: str = EXAMPLE_QUERIES[0],
        limit: Optional[int] = None,
        chunk_size_value=default_chunk_size,
        predict_total_for_items: Optional[int] = None,
        has_result: bool = False,
        response_payload: Optional[dict] = None,
        error: Optional[str] = None,
    ):
        payload = response_payload or {"orders": []}
        return render_template_string(
            PAGE_TEMPLATE,
            examples=EXAMPLE_QUERIES,
            query=query,
            limit=limit,
            chunk_size=chunk_size_value,
            default_chunk_size=default_chunk_size,
            predict_total_for_items=predict_total_for_items,
            has_result=has_result,
            orders=payload.get("orders", []),
            regression=payload.get("regression"),
            response_json=json.dumps(payload, indent=2),
            error=error,
        )

    @app.get("/")
    def index():
        return render_page()

    @app.post("/query")
    def query_orders():
        query = request.form.get("query", "").strip()
        limit_text = request.form.get("limit", "").strip()
        chunk_size_text = request.form.get("chunk_size", "").strip()
        predict_text = request.form.get("predict_total_for_items", "").strip()
        try:
            limit = int(limit_text) if limit_text else None
        except ValueError:
            return render_page(
                query=query,
                limit=None,
                predict_total_for_items=None,
                has_result=False,
                error="Record limit must be a whole number.",
            ), 400

        try:
            selected_chunk_size = (
                int(chunk_size_text) if chunk_size_text else default_chunk_size
            )
        except ValueError:
            return render_page(
                query=query,
                limit=limit,
                chunk_size_value=chunk_size_text,
                predict_total_for_items=None,
                has_result=False,
                error="Chunk size must be a whole number.",
            ), 400

        try:
            predict_total_for_items = int(predict_text) if predict_text else None
        except ValueError:
            return render_page(
                query=query,
                limit=limit,
                chunk_size_value=selected_chunk_size,
                predict_total_for_items=None,
                has_result=False,
                error="Prediction item count must be a whole number.",
            ), 400

        if selected_chunk_size < 1:
            return render_page(
                query=query,
                limit=limit,
                chunk_size_value=selected_chunk_size,
                predict_total_for_items=predict_total_for_items,
                has_result=False,
                error="Chunk size must be at least 1.",
            ), 400

        if predict_total_for_items is not None and predict_total_for_items < 1:
            return render_page(
                query=query,
                limit=limit,
                chunk_size_value=selected_chunk_size,
                predict_total_for_items=None,
                has_result=False,
                error="Prediction item count must be at least 1.",
            ), 400

        if not query:
            return render_page(
                query=query,
                limit=limit,
                chunk_size_value=selected_chunk_size,
                predict_total_for_items=predict_total_for_items,
                has_result=False,
                error="Enter a prompt before running the agent.",
            ), 400

        with request_id_context(new_request_id()):
            try:
                agent = build_agent(selected_chunk_size)
                if predict_total_for_items is None:
                    response = agent.run(query=query, limit=limit)
                    payload = response.to_dict()
                else:
                    response, parsed_orders = agent.run_with_records(
                        query=query,
                        limit=limit,
                    )
                    payload = response.to_dict()
                    try:
                        regression = predict_total_for_item_count(
                            parsed_orders,
                            predict_total_for_items,
                        )
                    except InsufficientRegressionData as exc:
                        regression = regression_error_response(exc)
                    payload["regression"] = regression.to_dict()
            except Exception as exc:  # pragma: no cover - route-level defensive logging.
                logger.exception("Failed to process UI query: %s", exc)
                payload = {"orders": [], "error": str(exc)}
                return render_page(
                    query=query,
                    limit=limit,
                    chunk_size_value=selected_chunk_size,
                    predict_total_for_items=predict_total_for_items,
                    has_result=True,
                    response_payload=payload,
                    error=str(exc),
                ), 500

        return render_page(
            query=query,
            limit=limit,
            chunk_size_value=selected_chunk_size,
            predict_total_for_items=predict_total_for_items,
            has_result=True,
            response_payload=payload,
        )

    return app
