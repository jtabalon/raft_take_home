from src.parsers import normalize_state, parse_order_text, parse_query_deterministic


def test_parse_order_text_handles_well_formed_orders():
    order, error = parse_order_text(
        "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable"
    )

    assert error is None
    assert order is not None
    assert order.orderId == "1001"
    assert order.buyer == "John Davis"
    assert order.city == "Columbus"
    assert order.state == "OH"
    assert order.total == 742.10
    assert order.items == ["laptop", "hdmi cable"]


def test_parse_order_text_rejects_malformed_orders_without_crashing():
    order, error = parse_order_text("Buyer=Unknown, Total=$10")

    assert order is None
    assert "Missing fields" in error


def test_parse_query_deterministic_extracts_state_and_min_total():
    spec = parse_query_deterministic(
        "Show me all orders where the buyer was located in Ohio and total value was over 500."
    )

    assert spec.state == "OH"
    assert spec.min_total == 500.0


def test_normalize_state_supports_full_names_and_codes():
    assert normalize_state("Ohio") == "OH"
    assert normalize_state("oh") == "OH"
