from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def validate_model(model_cls, payload):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class OrderQuerySpec(BaseModel):
    state: Optional[str] = None
    min_total: Optional[float] = None
    max_total: Optional[float] = None
    order_ids: List[str] = Field(default_factory=list)
    min_order_id: Optional[int] = None
    max_order_id: Optional[int] = None
    buyer_name: Optional[str] = None

    def is_empty(self) -> bool:
        return (
            self.state is None
            and self.min_total is None
            and self.max_total is None
            and not self.order_ids
            and self.min_order_id is None
            and self.max_order_id is None
            and self.buyer_name is None
        )


class OrderRecord(BaseModel):
    orderId: str
    buyer: str
    city: str
    state: str
    total: float
    items: List[str] = Field(default_factory=list)
    source: str = "regex"

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "orderId": self.orderId,
            "buyer": self.buyer,
            "state": self.state,
            "total": self.total,
        }


class OrdersResponse(BaseModel):
    orders: List[Dict[str, Any]] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


class InvalidOrder(BaseModel):
    raw_order: str
    error: str
    source: str = "unknown"


class OrderRepairResult(BaseModel):
    orderId: Optional[str] = None
    buyer: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    total: Optional[float] = None
    items: List[str] = Field(default_factory=list)
    reason: Optional[str] = None


class QueryExtractionResult(BaseModel):
    state: Optional[str] = None
    min_total: Optional[float] = None
    max_total: Optional[float] = None
    order_ids: List[str] = Field(default_factory=list)
    min_order_id: Optional[int] = None
    max_order_id: Optional[int] = None
    buyer_name: Optional[str] = None
    reason: Optional[str] = None
