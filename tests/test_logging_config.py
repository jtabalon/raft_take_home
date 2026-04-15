import io
import logging

from src.logging_config import (
    RequestIdFilter,
    get_request_id,
    request_id_context,
    setup_logging,
)


def test_log_records_get_default_request_id(caplog):
    setup_logging("INFO")
    caplog.set_level(logging.INFO, logger="order_agent.test")

    logging.getLogger("order_agent.test").info("default request")

    assert caplog.records[-1].request_id == "-"


def test_scoped_request_id_reaches_formatted_logs():
    setup_logging("INFO")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter("request_id=%(request_id)s %(message)s"))
    logger = logging.getLogger("order_agent.test.format")
    original_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    try:
        with request_id_context("req-test"):
            logger.info("scoped request")

        assert "request_id=req-test scoped request" in stream.getvalue()
        assert get_request_id() == "-"
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = True
