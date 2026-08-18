# Agent Harness

A small Python harness from the Agent Learning Notes:

- Turn states: `running` / `pending` / `completed` / `cancelled` / `failed`
- Tools: `read_file`, `write_file`, `bash`, `search_web`, `remember`, `ask_user`, `spawn`
- JSONL history locally, or Supabase when `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` are set
- Redis via `REDIS_URL`: cancel flag, turn event stream, write idempotency

## Setup

```bash
cd /Users/leowayne/Documents/agent-harness
uv sync
cp .env.example .env
```

Defaults match Kun: main model `xai/grok-4.5` via sub2api, small model `deepseek-v4-flash`.
Set `HARNESS_API_KEY` (and `HARNESS_SMALL_API_KEY` if DeepSeek uses a different key).

## Use

```bash
uv run harness chat "What files are in this directory?"
uv run harness chat --session ses_xxx "continue"
uv run harness resume --session ses_xxx "France"
uv run harness cancel --session ses_xxx
```

Without Supabase credentials, sessions stay under `data/sessions/`.
With Supabase, history, memory, and artifacts go to Postgres.

HTTP handshake after Vercel deploy: `GET /` or `GET /handshake`.
