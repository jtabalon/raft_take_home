import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypedDict

from pydantic import ValidationError

from src.api_client import CustomerAPIClient
from src.models import (
    InvalidOrder,
    OrderQuerySpec,
    OrderRecord,
    OrderRepairResult,
    OrdersResponse,
    QueryExtractionResult,
    model_to_dict,
    validate_model,
)
from src.parsers import normalize_state, parse_order_text, parse_query_deterministic


class AgentState(TypedDict):
    user_query: str
    limit: Optional[int]
    filter_spec: OrderQuerySpec
    raw_orders: List[str]
    chunks: List[List[str]]
    parsed_orders: List[OrderRecord]
    validated_orders: List[OrderRecord]
    invalid_orders: List[InvalidOrder]
    final_orders: List[Dict[str, Any]]
    errors: List[str]


class LLMClient(Protocol):
    def extract_query_spec(self, query: str) -> Optional[OrderQuerySpec]:
        ...

    def repair_order(self, raw_order: str) -> Optional[OrderRecord]:
        ...


class OpenRouterLLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openai/gpt-oss-120b:exacto",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.logger = logging.getLogger("order_agent.llm")

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on environment.
            raise RuntimeError(
                "langchain-openai is required for OpenRouter LLM access. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        client = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=0,
        )
        response = client.invoke(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(text_parts)
        return str(content)

    def extract_query_spec(self, query: str) -> Optional[OrderQuerySpec]:
        system_prompt = (
            "Extract filter criteria from the user query. Return JSON only. "
            "Allowed keys: state, min_total, max_total, order_ids, min_order_id, "
            "max_order_id, buyer_name, reason. "
            "Use a two-letter US state code when possible. Use null for unknown fields. "
            "Do not invent constraints."
        )
        raw = self._chat(system_prompt, query)
        payload = _parse_json_response(raw)
        _coerce_list_fields(payload, ("order_ids",))
        result = validate_model(QueryExtractionResult, payload)
        normalized_state = normalize_state(result.state) if result.state else None
        return validate_model(
            OrderQuerySpec,
            {
                "state": normalized_state,
                "min_total": result.min_total,
                "max_total": result.max_total,
                "order_ids": result.order_ids,
                "min_order_id": result.min_order_id,
                "max_order_id": result.max_order_id,
                "buyer_name": result.buyer_name,
            },
        )

    def repair_order(self, raw_order: str) -> Optional[OrderRecord]:
        system_prompt = (
            "You normalize messy customer order text into JSON. Return JSON only. "
            "Allowed keys: orderId, buyer, city, state, total, items, reason. "
            "If a field is missing or uncertain, use null and explain in reason. "
            "Never infer values that are not present in the text."
        )
        raw = self._chat(system_prompt, raw_order)
        payload = _parse_json_response(raw)
        _coerce_list_fields(payload, ("items",))
        result = validate_model(OrderRepairResult, payload)
        missing = [
            field
            for field in ("orderId", "buyer", "city", "state", "total")
            if getattr(result, field) in (None, "")
        ]
        if missing:
            self.logger.warning(
                "LLM repair could not fully parse order; missing=%s reason=%s",
                missing,
                result.reason,
            )
            return None

        return validate_model(
            OrderRecord,
            {
                "orderId": str(result.orderId),
                "buyer": result.buyer,
                "city": result.city,
                "state": normalize_state(result.state),
                "total": float(result.total),
                "items": result.items,
                "source": "llm",
            },
        )


def _parse_json_response(raw: str) -> Dict[str, Any]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
        candidate = re.sub(r"```$", "", candidate).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _coerce_list_fields(payload: Dict[str, Any], field_names: tuple[str, ...]) -> None:
    for field_name in field_names:
        if payload.get(field_name) is None:
            payload[field_name] = []


def build_default_llm_client() -> Optional[LLMClient]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenRouterLLMClient(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


class OrderAgent:
    def __init__(
        self,
        api_client: CustomerAPIClient,
        llm_client: Optional[LLMClient] = None,
        chunk_size: int = 50,
    ) -> None:
        self.api_client = api_client
        self.llm_client = llm_client if llm_client is not None else build_default_llm_client()
        self.chunk_size = max(1, chunk_size)
        self.logger = logging.getLogger("order_agent.agent")
        self.graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "langgraph is required to run the agent. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        graph = StateGraph(AgentState)
        graph.add_node("parse_user_request", self.parse_user_request)
        graph.add_node("fetch_orders", self.fetch_orders)
        graph.add_node("chunk_orders_if_needed", self.chunk_orders_if_needed)
        graph.add_node("parse_chunk", self.parse_chunk)
        graph.add_node("validate_records", self.validate_records)
        graph.add_node("filter_records", self.filter_records)
        graph.add_node("format_response", self.format_response)

        graph.set_entry_point("parse_user_request")
        graph.add_edge("parse_user_request", "fetch_orders")
        graph.add_edge("fetch_orders", "chunk_orders_if_needed")
        graph.add_edge("chunk_orders_if_needed", "parse_chunk")
        graph.add_edge("parse_chunk", "validate_records")
        graph.add_edge("validate_records", "filter_records")
        graph.add_edge("filter_records", "format_response")
        graph.add_edge("format_response", END)

        return graph.compile()

    def run(self, query: str, limit: Optional[int] = None) -> OrdersResponse:
        result = self._run_graph(query=query, limit=limit)
        return validate_model(OrdersResponse, {"orders": result["final_orders"]})

    def run_with_records(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> Tuple[OrdersResponse, List[OrderRecord]]:
        result = self._run_graph(query=query, limit=limit)
        response = validate_model(OrdersResponse, {"orders": result["final_orders"]})
        return response, list(result["validated_orders"])

    def _run_graph(self, query: str, limit: Optional[int] = None) -> AgentState:
        state: AgentState = {
            "user_query": query,
            "limit": limit,
            "filter_spec": validate_model(OrderQuerySpec, {}),
            "raw_orders": [],
            "chunks": [],
            "parsed_orders": [],
            "validated_orders": [],
            "invalid_orders": [],
            "final_orders": [],
            "errors": [],
        }
        started_at = time.perf_counter()
        self.logger.info(
            "Starting agent run; query_length=%s limit=%s",
            len(query),
            limit,
        )
        result = self.graph.invoke(state)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        self.logger.info(
            "Finished agent run; elapsed_ms=%.2f final_order_count=%s",
            elapsed_ms,
            len(result["final_orders"]),
        )
        return result

    def parse_user_request(self, state: AgentState) -> Dict[str, Any]:
        deterministic_spec = parse_query_deterministic(state["user_query"])
        llm_spec = None

        if self.llm_client is not None:
            try:
                llm_spec = self.llm_client.extract_query_spec(state["user_query"])
            except Exception as exc:
                self.logger.warning("LLM query parsing failed: %s", exc)

        filter_spec = _merge_query_specs(deterministic_spec, llm_spec, self.logger)
        self.logger.info("Parsed query spec: %s", model_to_dict(filter_spec))
        return {"filter_spec": filter_spec}

    def fetch_orders(self, state: AgentState) -> Dict[str, Any]:
        raw_orders = self.api_client.fetch_orders(limit=state["limit"])
        self.logger.info("Fetched %s raw orders", len(raw_orders))
        return {"raw_orders": raw_orders}

    def chunk_orders_if_needed(self, state: AgentState) -> Dict[str, Any]:
        raw_orders = state["raw_orders"]
        chunks = [
            raw_orders[index : index + self.chunk_size]
            for index in range(0, len(raw_orders), self.chunk_size)
        ]
        self.logger.info("Prepared %s chunks", len(chunks))
        return {"chunks": chunks}

    def parse_chunk(self, state: AgentState) -> Dict[str, Any]:
        parsed_orders: List[OrderRecord] = []
        invalid_orders: List[InvalidOrder] = []
        errors: List[str] = []

        for chunk in state["chunks"]:
            for raw_order in chunk:
                order, error = parse_order_text(raw_order)
                if order is not None:
                    parsed_orders.append(order)
                    continue

                repaired_order = None
                if self.llm_client is not None:
                    try:
                        repaired_order = self.llm_client.repair_order(raw_order)
                    except Exception as exc:
                        self.logger.warning("LLM order repair failed: %s", exc)
                        errors.append(str(exc))

                if repaired_order is not None:
                    parsed_orders.append(repaired_order)
                    continue

                invalid_orders.append(
                    validate_model(
                        InvalidOrder,
                        {
                            "raw_order": raw_order,
                            "error": error or "Unable to parse order",
                            "source": "regex+llm",
                        },
                    )
                )

        self.logger.info(
            "Parsed %s orders successfully, %s invalid",
            len(parsed_orders),
            len(invalid_orders),
        )
        return {
            "parsed_orders": parsed_orders,
            "invalid_orders": invalid_orders,
            "errors": state["errors"] + errors,
        }

    def validate_records(self, state: AgentState) -> Dict[str, Any]:
        valid_orders: List[OrderRecord] = []
        invalid_orders = list(state["invalid_orders"])

        for order in state["parsed_orders"]:
            try:
                valid_orders.append(validate_model(OrderRecord, model_to_dict(order)))
            except ValidationError as exc:
                invalid_orders.append(
                    validate_model(
                        InvalidOrder,
                        {
                            "raw_order": order.orderId,
                            "error": str(exc),
                            "source": order.source,
                        },
                    )
                )

        return {
            "parsed_orders": valid_orders,
            "validated_orders": valid_orders,
            "invalid_orders": invalid_orders,
        }

    def filter_records(self, state: AgentState) -> Dict[str, Any]:
        filter_spec = state["filter_spec"]
        filtered = [
            order for order in state["parsed_orders"] if _matches_filter(order, filter_spec)
        ]
        filtered.sort(key=lambda order: int(order.orderId))
        return {"parsed_orders": filtered}

    def format_response(self, state: AgentState) -> Dict[str, Any]:
        final_orders = [order.to_public_dict() for order in state["parsed_orders"]]
        return {"final_orders": final_orders}


def _merge_query_specs(
    deterministic_spec: OrderQuerySpec,
    llm_spec: Optional[OrderQuerySpec],
    logger: logging.Logger,
) -> OrderQuerySpec:
    if llm_spec is None:
        return deterministic_spec

    merged = model_to_dict(deterministic_spec)
    llm_payload = model_to_dict(llm_spec)
    _drop_llm_total_ranges_that_duplicate_order_id_ranges(merged, llm_payload)
    for key, llm_value in llm_payload.items():
        deterministic_value = merged.get(key)
        if deterministic_value not in (None, [], ""):
            if llm_value not in (None, [], "") and deterministic_value != llm_value:
                logger.warning(
                    "Ignoring conflicting LLM query field %s=%s; keeping deterministic=%s",
                    key,
                    llm_value,
                    deterministic_value,
                )
            continue
        merged[key] = llm_value
    return validate_model(OrderQuerySpec, merged)


def _drop_llm_total_ranges_that_duplicate_order_id_ranges(
    deterministic_payload: Dict[str, Any],
    llm_payload: Dict[str, Any],
) -> None:
    pairs = (
        ("min_order_id", "min_total"),
        ("max_order_id", "max_total"),
    )
    for order_id_key, total_key in pairs:
        order_id_value = deterministic_payload.get(order_id_key)
        total_value = llm_payload.get(total_key)
        if order_id_value is not None and _same_numeric_value(order_id_value, total_value):
            llm_payload[total_key] = None


def _same_numeric_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def _matches_filter(order: OrderRecord, filter_spec: OrderQuerySpec) -> bool:
    if filter_spec.state and order.state != filter_spec.state:
        return False
    if filter_spec.min_total is not None and order.total <= filter_spec.min_total:
        return False
    if filter_spec.max_total is not None and order.total >= filter_spec.max_total:
        return False
    if filter_spec.order_ids and order.orderId not in filter_spec.order_ids:
        return False
    if filter_spec.min_order_id is not None or filter_spec.max_order_id is not None:
        try:
            order_id = int(order.orderId)
        except ValueError:
            return False
        if filter_spec.min_order_id is not None and order_id <= filter_spec.min_order_id:
            return False
        if filter_spec.max_order_id is not None and order_id >= filter_spec.max_order_id:
            return False
    if filter_spec.buyer_name and order.buyer.lower() != filter_spec.buyer_name.lower():
        return False
    return True
