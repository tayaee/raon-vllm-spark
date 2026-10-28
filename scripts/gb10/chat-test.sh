#!/usr/bin/env bash
# Smoke-test the local Raon server (default 127.0.0.1:8000).
#   ./scripts/gb10/serve-raon-nvfp4.sh &   # terminal 1
#   ./scripts/gb10/chat-test.sh             # terminal 2
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-Raon-1-mini-preview}"
BASE="http://${HOST}:${PORT}"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "==> GET /health"
curl -sf --max-time 30 "${BASE}/health" >/dev/null || fail "/health unreachable"
echo ok

echo "==> GET /v1/models"
MODELS="$(curl -sf --max-time 30 "${BASE}/v1/models")" || fail "/v1/models"
echo "$MODELS" | python3 -c "import json,sys; print('models:', [m['id'] for m in json.load(sys.stdin)['data']])"

echo "==> POST /v1/chat/completions (non-stream)"
curl -sf --max-time 600 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"서울의 수도는 어디야? 한 문장으로 답해."}],"max_tokens":512,"temperature":1.0}' \
  | python3 -c "
import json,sys
r = json.load(sys.stdin)
msg = r['choices'][0]['message']
print('keys:', sorted(msg.keys()))
print('reasoning:', (msg.get('reasoning_content') or '')[:200])
print('content:', (msg.get('content') or '')[:500])
print('usage:', r.get('usage'))
assert ((msg.get('content') or '') + (msg.get('reasoning_content') or '')).strip(), 'empty content+reasoning'
print('non-stream OK')
"

echo "==> POST /v1/chat/completions (stream)"
curl -sfN --max-time 600 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"1+1은? 짧게 답해."}],"max_tokens":32,"temperature":1.0,"stream":true}' \
  | python3 -c "
import sys
chunks = 0
for line in sys.stdin:
    line = line.strip()
    if line.startswith('data: ') and line != 'data: [DONE]':
        chunks += 1
print(f'stream chunks: {chunks}')
assert chunks > 0, 'no stream chunks'
print('stream OK')
"

echo "ALL CHAT TESTS PASSED"
