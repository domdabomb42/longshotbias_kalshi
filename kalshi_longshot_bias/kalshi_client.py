"""HTTP client for Kalshi API v2 with retries and pagination."""
from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import logging
import time
from typing import Any, Dict, Iterator, List
from urllib.parse import urlsplit

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

import httpx

logger = logging.getLogger(__name__)


@dataclass
class KalshiClientConfig:
    base_url: str
    access_key_id: str | None = None
    private_key_path: str | None = None
    private_key_pem: str | None = None
    verify: bool = True
    timeout: float = 20.0
    max_retries: int = 5
    backoff_base: float = 0.5
    backoff_max: float = 8.0
    request_delay: float = 0.0


class KalshiSigner:
    def __init__(self, access_key_id: str, private_key: rsa.RSAPrivateKey) -> None:
        self.access_key_id = access_key_id
        self.private_key = private_key

    @classmethod
    def from_config(cls, config: KalshiClientConfig) -> "KalshiSigner | None":
        if not config.access_key_id:
            return None
        pem_data = None
        if config.private_key_pem:
            pem_data = config.private_key_pem.encode("utf-8")
        elif config.private_key_path:
            pem_data = open(config.private_key_path, "rb").read()
        if not pem_data:
            return None
        private_key = serialization.load_pem_private_key(pem_data, password=None)
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise RuntimeError("Invalid private key type for Kalshi auth.")
        return cls(config.access_key_id, private_key)

    def headers_for(self, method: str, full_path: str) -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        message = (timestamp + method.upper() + full_path).encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        signature_b64 = base64.b64encode(signature).decode("ascii")
        return {
            "KALSHI-ACCESS-KEY": self.access_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }


class KalshiClient:
    def __init__(self, config: KalshiClientConfig) -> None:
        self.config = config
        headers: Dict[str, str] = {"User-Agent": "kalshi-longshot-bias/0.1", "Accept": "application/json"}

        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout,
            headers=headers,
            verify=config.verify,
        )
        self._signer = KalshiSigner.from_config(config)
        self._base_path = urlsplit(str(self._client.base_url)).path.rstrip("/")

    def close(self) -> None:
        self._client.close()

    def request(self, method: str, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        url = path if path.startswith("/") else f"/{path}"
        params = params or {}
        headers = None
        if self._signer:
            full_path = self._full_path_for_sign(url)
            headers = self._signer.headers_for(method, full_path)
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                if self.config.request_delay > 0:
                    time.sleep(self.config.request_delay)
                resp = self._client.request(method, url, params=params, headers=headers)
            except httpx.RequestError as exc:
                last_exc = exc
                delay = min(self.config.backoff_base * (2**attempt), self.config.backoff_max)
                logger.warning("Request error %s %s (%s). Retrying in %.1fs", method, url, exc, delay)
                time.sleep(delay)
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    delay = float(retry_after)
                else:
                    delay = min(self.config.backoff_base * (2**attempt), self.config.backoff_max)
                logger.warning(
                    "HTTP %s for %s %s. Retrying in %.1fs",
                    resp.status_code,
                    method,
                    url,
                    delay,
                )
                time.sleep(delay)
                continue

            if resp.status_code >= 400:
                detail = resp.text[:400]
                raise RuntimeError(f"Kalshi API error {resp.status_code} for {url}: {detail}")

            if not resp.content:
                return {}
            try:
                return resp.json()
            except json.JSONDecodeError as exc:
                snippet = resp.text[:400]
                raise RuntimeError(f"Invalid JSON from {url}: {exc}. Body starts: {snippet}") from exc

        if last_exc:
            raise RuntimeError(f"Failed to call {url}: {last_exc}") from last_exc
        raise RuntimeError(f"Failed to call {url} after {self.config.max_retries} attempts")

    def get(self, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self.request("GET", path, params=params)

    def _full_path_for_sign(self, url: str) -> str:
        if url.startswith("http://") or url.startswith("https://"):
            parsed = urlsplit(url)
            path = parsed.path
        else:
            path = url
        path = path.split("?")[0]
        return f"{self._base_path}{path}"

    def paginate(
        self,
        path: str,
        params: Dict[str, Any] | None = None,
        data_key: str | None = None,
        limit: int = 200,
    ) -> Iterator[List[Dict[str, Any]]]:
        params = dict(params or {})
        params.setdefault("limit", limit)

        while True:
            payload = self.get(path, params=params)
            items = self._extract_items(payload, data_key)
            yield items

            cursor = (
                payload.get("next_cursor")
                or payload.get("cursor")
                or payload.get("next")
                or payload.get("pagination", {}).get("next_cursor")
            )
            if not cursor:
                break
            params["cursor"] = cursor

    @staticmethod
    def _extract_items(payload: Dict[str, Any], data_key: str | None) -> List[Dict[str, Any]]:
        if data_key and data_key in payload:
            return payload.get(data_key, []) or []

        for key in ("markets", "data", "results", "items"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]

        # Fallback: first list in payload
        for value in payload.values():
            if isinstance(value, list):
                return value
        raise RuntimeError("Unable to find list payload in response")
