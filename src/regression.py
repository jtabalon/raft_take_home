from dataclasses import dataclass
from typing import Iterable, List

from src.models import (
    OrderRecord,
    RegressionModelSummary,
    RegressionPrediction,
    RegressionResponse,
    validate_model,
)


REGRESSION_NOTE = "Demonstration baseline trained on the current parsed API response."


class InsufficientRegressionData(ValueError):
    """Raised when the parsed orders cannot support a useful regression demo."""


@dataclass(frozen=True)
class RegressionTrainingRow:
    item_count: int
    total: float


def build_training_rows(orders: Iterable[OrderRecord]) -> List[RegressionTrainingRow]:
    rows: List[RegressionTrainingRow] = []
    for order in orders:
        item_count = len(order.items)
        if item_count < 1:
            continue
        rows.append(RegressionTrainingRow(item_count=item_count, total=float(order.total)))
    return rows


def predict_total_for_item_count(
    orders: Iterable[OrderRecord],
    item_count: int,
) -> RegressionResponse:
    if item_count < 1:
        raise ValueError("item_count must be at least 1")

    rows = build_training_rows(orders)
    if len(rows) < 2:
        raise InsufficientRegressionData(
            "At least two parsed orders with item counts and totals are required."
        )

    distinct_item_counts = {row.item_count for row in rows}
    if len(distinct_item_counts) < 2:
        raise InsufficientRegressionData(
            "At least two distinct item counts are required for a useful regression."
        )

    x_values = [[row.item_count] for row in rows]
    y_values = [row.total for row in rows]

    try:
        from sklearn.linear_model import LinearRegression
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for regression predictions. "
            "Run `python3 -m pip install -r requirements.txt` in your virtual environment."
        ) from exc

    model = LinearRegression()
    model.fit(x_values, y_values)
    predicted_total = float(model.predict([[item_count]])[0])

    return validate_model(
        RegressionResponse,
        {
            "model": validate_model(
                RegressionModelSummary,
                {
                    "type": "sklearn.linear_model.LinearRegression",
                    "feature": "item_count",
                    "sample_count": len(rows),
                    "coefficient": _round_currency(float(model.coef_[0])),
                    "intercept": _round_currency(float(model.intercept_)),
                },
            ),
            "prediction": validate_model(
                RegressionPrediction,
                {
                    "item_count": item_count,
                    "predicted_total": _round_currency(predicted_total),
                },
            ),
            "note": REGRESSION_NOTE,
        },
    )


def regression_error_response(error: Exception) -> RegressionResponse:
    return validate_model(
        RegressionResponse,
        {
            "error": str(error),
            "note": REGRESSION_NOTE,
        },
    )


def _round_currency(value: float) -> float:
    return round(value, 2)
