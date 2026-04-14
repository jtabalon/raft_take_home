import builtins

import pytest

from src.models import OrderRecord, validate_model
from src.regression import InsufficientRegressionData, predict_total_for_item_count


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


def test_predict_total_for_item_count_uses_sklearn_linear_regression():
    orders = [
        make_order(1001, ["one"], 100.0),
        make_order(1002, ["one", "two", "three"], 300.0),
    ]

    result = predict_total_for_item_count(orders, item_count=2).to_dict()

    assert result["model"] == {
        "type": "sklearn.linear_model.LinearRegression",
        "feature": "item_count",
        "sample_count": 2,
        "coefficient": 100.0,
        "intercept": 0.0,
    }
    assert result["prediction"] == {
        "item_count": 2,
        "predicted_total": 200.0,
    }
    assert "Demonstration baseline" in result["note"]
    assert "error" not in result


def test_predict_total_for_item_count_requires_enough_distinct_data():
    orders = [
        make_order(1001, ["one"], 100.0),
        make_order(1002, ["one"], 150.0),
    ]

    with pytest.raises(InsufficientRegressionData):
        predict_total_for_item_count(orders, item_count=2)


def test_predict_total_for_item_count_explains_missing_sklearn(monkeypatch):
    orders = [
        make_order(1001, ["one"], 100.0),
        make_order(1002, ["one", "two", "three"], 300.0),
    ]
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sklearn.linear_model":
            raise ModuleNotFoundError("No module named 'sklearn'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="scikit-learn is required"):
        predict_total_for_item_count(orders, item_count=2)
