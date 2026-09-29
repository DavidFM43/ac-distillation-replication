#!/usr/bin/env bash
# Start (or refresh) local Prometheus + Grafana: http://127.0.0.1:3000
# Re-run after each `infra/vllm_host.py up`, since every deployment gets a new API key.
set -euo pipefail
cd "$(dirname "$0")"

# vLLM's official dashboard, pinned to the engine version; its one hard-coded datasource uid
# is pointed at ours (the rest use a datasource variable that resolves to the default).
VLLM_DASHBOARD=https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/examples/observability/prometheus_grafana/grafana.json
curl -fsSL "$VLLM_DASHBOARD" \
  | sed 's/"b281712d-8bff-41ef-9f3f-71ad43c05e9b"/"prometheus"/g' > grafana/dashboards/vllm.json

# Current vLLM API key for Prometheus (empty if no deployment is up).
python3 -c "import json; print(json.load(open('../.vllm_endpoint.json'))['api_key'], end='')" \
  > .vllm_api_key 2>/dev/null || : > .vllm_api_key
chmod 600 .vllm_api_key

docker compose up -d --force-recreate prometheus
docker compose up -d grafana
echo "Grafana: http://127.0.0.1:3000   Prometheus: http://127.0.0.1:9090"
