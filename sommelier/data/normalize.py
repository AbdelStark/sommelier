from __future__ import annotations

import hashlib


def normalize_query(query: str) -> str:
    return " ".join(query.casefold().strip().split())


def query_digest(query: str) -> str:
    return hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()
