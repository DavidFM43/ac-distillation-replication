"""Check that a running self-hosted vLLM endpoint only answers with its API key (reads .vllm_endpoint.json).

    uv run python infra/check_endpoint.py

/v1/models must reject requests without the key (401) and accept them with it (200).
Standard library only.
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT_FILE = Path(".vllm_endpoint.json")


def status_of(url: str, api_key: str | None) -> int:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    if not ENDPOINT_FILE.exists():
        sys.exit(f"No {ENDPOINT_FILE}: start an endpoint with `infra/vllm_host.py up` first")
    endpoint = json.loads(ENDPOINT_FILE.read_text())
    url = f"{endpoint['base_url']}/models"
    without_key, with_key = status_of(url, None), status_of(url, endpoint["api_key"])
    ok = (without_key, with_key) == (401, 200)
    print(f"auth: without key -> {without_key}, with key -> {with_key} -> {'ok' if ok else 'UNEXPECTED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
