"""Client de l'API publique (non officielle) utilisée par le site PMU."""
from __future__ import annotations

import time
from datetime import date

import requests

BASE = "https://online.turfinfo.api.pmu.fr/rest/client/1/programme"
HEADERS = {"User-Agent": "pmu-predictor/1.0", "Accept": "application/json"}

_session = requests.Session()
_session.headers.update(HEADERS)


def fmt_date(d: date) -> str:
    return d.strftime("%d%m%Y")


def _get(url: str, retries: int = 4) -> dict | list | None:
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (204, 404):
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 ** attempt)
    return None


def programme(d: date) -> dict | None:
    data = _get(f"{BASE}/{fmt_date(d)}")
    return data.get("programme") if isinstance(data, dict) else None


def participants(d: date, reunion: int, course: int) -> list[dict]:
    data = _get(f"{BASE}/{fmt_date(d)}/R{reunion}/C{course}/participants")
    return data.get("participants", []) if isinstance(data, dict) else []


def rapports(d: date, reunion: int, course: int) -> list[dict]:
    data = _get(f"{BASE}/{fmt_date(d)}/R{reunion}/C{course}/rapports-definitifs")
    return data if isinstance(data, list) else []
