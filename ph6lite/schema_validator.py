RECEIPT_REQUIRED = {
    "schema_version", "object_id", "session_id", "captured_at",
    "source_path", "stored_path", "filename", "hash_blake2b256",
    "measurement_ref", "state", "authority"
}

MEASUREMENT_REQUIRED = {
    "schema_version", "object_id", "captured_at",
    "size_bytes", "entropy_shannon", "line_count", "sha256_preview_only"
}

SOSO_TOKEN_REQUIRED = {"schema_version", "token_id", "token_type", "anchors", "confidence"}
SOSO_EDGE_REQUIRED = {"schema_version", "edge_id", "from_object_id", "to_object_id",
                      "relation", "confidence"}

VALID_RELATIONS = {"weak_related", "session_related", "strong_related"}
VALID_STATES = {"CAPTURED"}
VALID_AUTHORITIES = {"PH6-Lite"}


def validate_receipt(data: dict) -> list[str]:
    errors = []
    missing = RECEIPT_REQUIRED - data.keys()
    if missing:
        errors.append(f"missing fields: {missing}")
    if data.get("state") not in VALID_STATES:
        errors.append(f"invalid state: {data.get('state')}")
    if data.get("authority") not in VALID_AUTHORITIES:
        errors.append(f"invalid authority: {data.get('authority')}")
    return errors


def validate_measurement(data: dict) -> list[str]:
    errors = []
    missing = MEASUREMENT_REQUIRED - data.keys()
    if missing:
        errors.append(f"missing fields: {missing}")
    for field in ("size_bytes", "entropy_shannon", "line_count"):
        if field in data and not isinstance(data[field], (int, float)):
            errors.append(f"{field} must be numeric")
    return errors


def validate_soso_token(data: dict) -> list[str]:
    missing = SOSO_TOKEN_REQUIRED - data.keys()
    return [f"missing fields: {missing}"] if missing else []


def validate_soso_edge(data: dict) -> list[str]:
    errors = []
    missing = SOSO_EDGE_REQUIRED - data.keys()
    if missing:
        errors.append(f"missing fields: {missing}")
    if data.get("relation") not in VALID_RELATIONS:
        errors.append(f"invalid relation: {data.get('relation')}")
    return errors
