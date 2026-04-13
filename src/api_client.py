import logging
from typing import Any, Dict, List, Optional

import requests


class APIResponseError(RuntimeError):
    """Raised when the customer API returns an unexpected payload."""


class CustomerAPIClient:
    def __init__(
        self,
        base_url: str,
        session: Optional[requests.Session] = None,
        timeout: int = 10,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.logger = logging.getLogger("order_agent.api_client")

    def fetch_orders(self, limit: Optional[int] = None) -> List[str]:
        params = {}
        if limit is not None:
            params["limit"] = limit

        url = "{base_url}/api/orders".format(base_url=self.base_url)
        self.logger.info("Fetching orders from customer API")
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        return self._extract_orders(payload)

    def _extract_orders(self, payload: Dict[str, Any]) -> List[str]:
        orders = self._find_orders_list(payload)
        if orders is None:
            raise APIResponseError(
                "Customer API payload did not include a usable orders list."
            )
        return [str(order) for order in orders]

    def _find_orders_list(self, payload: Any) -> Optional[List[Any]]:
        if isinstance(payload, list):
            return payload

        if not isinstance(payload, dict):
            return None

        for key in ("raw_orders", "orders"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

        for nested_key in ("data", "result", "results"):
            nested = payload.get(nested_key)
            if isinstance(nested, dict):
                nested_orders = self._find_orders_list(nested)
                if nested_orders is not None:
                    return nested_orders

        return None
