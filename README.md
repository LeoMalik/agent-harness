# Agent Harness

A small Python harness from the Agent Learning Notes:

- Turn states: `running` / `pending` / `completed` / `cancelled` / `failed`
- Lifecycle hooks: `session_start`, `user_prompt_submit`, `before_llm_call`, `after_llm_call`, `before_tool`, `after_tool_call`, `pre_compact`, `after_turn`
- Before-hooks are sync and time out after `HARNESS_HOOK_TIMEOUT` (default 30s, fail-closed)
- After-hooks persist memory / metrics in the background and never reject
- Each tool has its own `before_hooks` / `after_hooks` (empty unless needed). `ask_user` has an interrupt hook; `write_file` / `bash` have approval + idempotency
- Per-user credits in `user_balances` (default 1_000_000, 1 credit per turn)
- Extensible defaults in `user_settings`: default model, reasoning effort, `soul.md`, and future JSON settings
- Workspace-aware session navigation with inbox, unread, starred, archive, and trash filters
- Chat statistics persist model, reasoning effort, token usage, tool calls, and credits
- History, memory, artifacts, balances, user settings, and session metadata go to Supabase (`SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`)
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

Supabase is required for sessions, events, memory, and artifacts.

HTTP handshake after Vercel deploy: `GET /` or `GET /handshake`.
