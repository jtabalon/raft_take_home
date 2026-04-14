from dummy_customer_api import app


def test_dummy_api_root_explains_available_endpoints():
    response = app.test_client().get("/")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["endpoints"]["orders"] == "/api/orders"


def test_dummy_api_health_check():
    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_dummy_api_favicon_request_is_quiet():
    response = app.test_client().get("/favicon.ico")

    assert response.status_code == 204


def test_dummy_api_orders_endpoint_returns_raw_orders():
    response = app.test_client().get("/api/orders?limit=2")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert len(payload["raw_orders"]) == 2
