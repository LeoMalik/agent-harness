# Agent Harness

A small Python harness from the Agent Learning Notes:

- Turn states: `running` / `pending` / `completed` / `cancelled` / `failed`
- Tools: `read_file`, `write_file`, `bash`, `search_web`, `remember`, `ask_user`, `spawn`
- JSONL history, resume, memory reminders, skill catalog, saved graph templates
- Redis via `REDIS_URL`: cancel flag, turn event stream, write idempotency. Local first; change the URL for cloud.

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

Sessions are stored under `data/sessions/`. Memory is keyed by `user_id` + `workspace_id`.
Redis defaults to `redis://127.0.0.1:6379/0`. If Redis is down, the runtime falls back to an in-process store.
