import re
from typing import List, Optional, Tuple

from src.models import OrderQuerySpec, validate_model


STATE_NAME_TO_CODE = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}


ORDER_ID_RE = re.compile(r"\bOrder\s*#?\s*(\d+)\b", re.IGNORECASE)
ORDER_ID_FIELD_PATTERN = r"orders?[_\s-]?ids?"
BUYER_RE = re.compile(
    r"Buyer\s*[:=]\s*(.+?)(?=,\s*(?:Location|Total|Items?)\b|\|\s*(?:Location|Total|Items?)\b|$)",
    re.IGNORECASE,
)
LOCATION_RE = re.compile(
    r"Location\s*[:=]\s*(.+?)(?=,\s*Total\b|\|\s*Total\b|$)",
    re.IGNORECASE,
)
TOTAL_RE = re.compile(r"Total\s*[:=]\s*\$?\s*([0-9]+(?:\.[0-9]{1,2})?)", re.IGNORECASE)
ITEMS_RE = re.compile(r"Items?\s*[:=]\s*(.+)$", re.IGNORECASE)


def normalize_state(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    cleaned = value.strip().lower()
    if len(cleaned) == 2 and cleaned.isalpha():
        return cleaned.upper()
    return STATE_NAME_TO_CODE.get(cleaned)


def parse_location(location: str) -> Tuple[Optional[str], Optional[str]]:
    parts = [part.strip() for part in location.split(",") if part.strip()]
    if not parts:
        return None, None

    if len(parts) == 1:
        return parts[0], None

    city = ", ".join(parts[:-1])
    state = normalize_state(parts[-1])
    return city, state


def parse_items(items_text: Optional[str]) -> List[str]:
    if not items_text:
        return []
    return [item.strip() for item in items_text.split(",") if item.strip()]


def parse_order_text(raw_order: str):
    order_id_match = ORDER_ID_RE.search(raw_order)
    buyer_match = BUYER_RE.search(raw_order)
    location_match = LOCATION_RE.search(raw_order)
    total_match = TOTAL_RE.search(raw_order)
    items_match = ITEMS_RE.search(raw_order)

    missing = []
    if not order_id_match:
        missing.append("orderId")
    if not buyer_match:
        missing.append("buyer")
    if not location_match:
        missing.append("location")
    if not total_match:
        missing.append("total")

    if missing:
        return None, "Missing fields: {fields}".format(fields=", ".join(missing))

    city, state = parse_location(location_match.group(1))
    if not city or not state:
        return None, "Could not normalize location"

    try:
        total = float(total_match.group(1))
    except ValueError:
        return None, "Could not parse total"

    payload = {
        "orderId": order_id_match.group(1).strip(),
        "buyer": buyer_match.group(1).strip(),
        "city": city,
        "state": state,
        "total": total,
        "items": parse_items(items_match.group(1) if items_match else None),
        "source": "regex",
    }
    from src.models import OrderRecord

    return validate_model(OrderRecord, payload), None


def parse_query_deterministic(query: str) -> OrderQuerySpec:
    lowered = query.lower()
    payload = {
        "state": None,
        "min_total": None,
        "max_total": None,
        "order_ids": [],
        "min_order_id": None,
        "max_order_id": None,
        "buyer_name": None,
    }

    for state_name, state_code in STATE_NAME_TO_CODE.items():
        if re.search(r"\b{state}\b".format(state=re.escape(state_name)), lowered):
            payload["state"] = state_code
            break

    if payload["state"] is None:
        state_match = re.search(r"\b([A-Z]{2})\b", query)
        if state_match:
            payload["state"] = normalize_state(state_match.group(1))

    min_patterns = [
        r"\b(?:over|above|greater than|more than)\s+\$?(\d+(?:\.\d+)?)",
        r"\b(?:at least|minimum(?: total)?(?: of)?)\s+\$?(\d+(?:\.\d+)?)",
    ]
    max_patterns = [
        r"\b(?:under|below|less than)\s+\$?(\d+(?:\.\d+)?)",
        r"\b(?:at most|maximum(?: total)?(?: of)?)\s+\$?(\d+(?:\.\d+)?)",
    ]
    order_id_min_patterns = [
        rf"\b{ORDER_ID_FIELD_PATTERN}\b\s+(?:is|are)?\s*(?:over|above|greater than|more than)\s+(\d+)\b",
    ]
    order_id_max_patterns = [
        rf"\b{ORDER_ID_FIELD_PATTERN}\b\s+(?:is|are)?\s*(?:under|below|less than)\s+(\d+)\b",
    ]

    for pattern in min_patterns:
        match = re.search(pattern, lowered)
        if match:
            payload["min_total"] = float(match.group(1))
            break

    for pattern in max_patterns:
        match = re.search(pattern, lowered)
        if match:
            payload["max_total"] = float(match.group(1))
            break

    for pattern in order_id_min_patterns:
        match = re.search(pattern, lowered)
        if match:
            payload["min_order_id"] = int(match.group(1))
            payload["min_total"] = None
            break

    for pattern in order_id_max_patterns:
        match = re.search(pattern, lowered)
        if match:
            payload["max_order_id"] = int(match.group(1))
            payload["max_total"] = None
            break

    order_id_matches = []
    order_id_patterns = [
        r"\border\s*#?\s*(\d{3,})\b",
        r"\border\s+id\s*#?\s*(\d{3,})\b",
        r"\border\s+ids?\s*[:#]?\s*([\d,\s]+)\b",
        rf"\b{ORDER_ID_FIELD_PATTERN}\b\s*[:#]?\s*([\d,\s]+)\b",
    ]
    for pattern in order_id_patterns:
        for match in re.finditer(pattern, query, re.IGNORECASE):
            if match.lastindex != 1:
                continue
            value = match.group(1)
            if "," in value or " " in value:
                order_id_matches.extend(re.findall(r"\d{3,}", value))
            else:
                order_id_matches.append(value)
    if order_id_matches:
        payload["order_ids"] = sorted(set(order_id_matches))

    buyer_match = re.search(
        r"\bbuyer\s+(?:was|is|named)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        query,
    )
    if buyer_match:
        payload["buyer_name"] = buyer_match.group(1).strip()

    return validate_model(OrderQuerySpec, payload)
