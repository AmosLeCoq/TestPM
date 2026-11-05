"""Envoi de messages vers un webhook Discord."""
from __future__ import annotations

import os
import time

import requests

COLOR_OK, COLOR_KO, COLOR_INFO = 0x2ECC71, 0xE74C3C, 0x3498DB


def webhook_url(url: str | None = None) -> str | None:
    return url or os.environ.get("DISCORD_WEBHOOK_URL")


def send(embeds: list[dict], url: str | None = None, content: str | None = None) -> None:
    """Envoie les embeds un par un (respecte les limites de taille de Discord)."""
    url = webhook_url(url)
    if not url:
        print("[discord] DISCORD_WEBHOOK_URL non défini : message affiché ici seulement.")
        if content:
            print(content)
        for e in embeds:
            print(f"--- {e.get('title', '')}\n{e.get('description', '')}")
        return
    payloads = []
    if content:
        payloads.append({"content": content[:2000]})
    for e in embeds:
        e = dict(e)
        e["description"] = e.get("description", "")[:4000]
        payloads.append({"embeds": [e]})
    for body in payloads:
        body["username"] = "PMU Predictor"
        for _ in range(5):
            r = requests.post(url, json=body, timeout=30)
            if r.status_code == 429:
                time.sleep(float(r.json().get("retry_after", 1)))
                continue
            if r.status_code >= 400:
                print(f"[discord] erreur {r.status_code}: {r.text[:200]}")
            break
        time.sleep(0.5)
