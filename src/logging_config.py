import contextlib
import contextvars
import logging
import secrets
from typing import Iterator


_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id",
    default="-",
)
_base_log_record_factory = logging.getLogRecordFactory()


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = _request_id.get()
        return True


def get_request_id() -> str:
    return _request_id.get()


def new_request_id() -> str:
    return secrets.token_hex(4)


@contextlib.contextmanager
def request_id_context(request_id: str) -> Iterator[None]:
    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)


def _record_factory(*args, **kwargs) -> logging.LogRecord:
    record = _base_log_record_factory(*args, **kwargs)
    record.request_id = _request_id.get()
    return record


def setup_logging(level: str = "INFO") -> None:
    logging.setLogRecordFactory(_record_factory)
    request_id_filter = RequestIdFilter()
    root_logger = logging.getLogger()
    root_logger.addFilter(request_id_filter)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=(
            "%(asctime)s | %(levelname)s | request_id=%(request_id)s | "
            "%(name)s | %(message)s"
        ),
    )
    for handler in root_logger.handlers:
        handler.addFilter(request_id_filter)
