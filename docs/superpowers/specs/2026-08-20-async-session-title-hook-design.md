# Async Session Title Hook Design

## Goal

Generate a concise title for a new session from its first user message without delaying the main Agent response. Title generation must be implemented as a lifecycle Hook, not inline Runtime logic.

## Title Contract

- Follow the language of the first user message.
- Chinese titles: at most 12 Han characters.
- English titles: at most 8 whitespace-delimited words.
- One line only.
- No Markdown, quotation marks, labels such as `Title:`, or trailing sentence punctuation.
- Describe the user's task; do not answer it.
- Generate once. Never overwrite a non-empty title.

## Lifecycle

`Runtime.start_turn()` keeps synchronous validation and persistence in this order:

1. Create and save the Turn.
2. Run synchronous `USER_PROMPT_SUBMIT` Hooks such as balance validation.
3. Reject early if a synchronous Hook fails.
4. Persist the `user_message` event.
5. Dispatch non-blocking `USER_PROMPT_SUBMIT` After-hooks.
6. Enter the main Agent Loop immediately.

The title Hook is registered as a non-blocking `USER_PROMPT_SUBMIT` Hook. It runs after the first user message is durable, in parallel with the main model request.

## SessionTitleHook

The Hook receives `HookContext` containing Runtime, Session, Turn, and the first submitted text in `extra["user_text"]`.

It exits without calling a model when:

- `session.title` is already non-empty;
- this is not the first user message in the session;
- the user text is empty.

Otherwise it:

1. Creates a dedicated small `LLM(config, small=True)` instance.
2. Calls it with `tools=None` and a title-only System Prompt.
3. Sanitizes the model response according to the Title Contract.
4. Re-reads the Session before writing so a concurrent manual title change is not overwritten.
5. Writes the title only if the stored title is still empty.
6. Persists the Session and publishes `session.title_updated` with `session_id` and `title`.

The Hook does not reuse the main Runtime LLM instance and does not mutate Turn state.

## Failure and Concurrency

- Title generation is non-critical and never changes Turn success or failure.
- Timeout, missing small-model credentials, invalid output, database failure, or model failure leave the title empty.
- Failures are logged by the existing async Hook runner.
- No fallback to synchronous substring titles is allowed.
- Compare-before-write prevents an async result from replacing a title that was set while generation was running.
- Multiple dispatches are harmless because the Hook checks both first-message status and current title.

## Frontend Behavior

- Empty titles render as `未命名聊天`.
- The existing session-list refresh after a turn retrieves the generated title when it is already available.
- Polling consumes `session.title_updated`; on receipt, the frontend reloads sessions and updates the active header without interrupting chat streaming.
- No loading spinner is required for the title.

## Tests

Required automated coverage:

1. The first submitted message dispatches the title Hook after `user_message` persistence.
2. Title generation runs without delaying the main model call.
3. Existing titles are never overwritten.
4. Later turns do not generate another title.
5. Chinese output is sanitized to at most 12 Han characters.
6. English output is sanitized to at most 8 words.
7. Quotes, Markdown, labels, newlines, and trailing punctuation are removed.
8. Small-model or persistence failure does not fail the Turn.
9. `session.title_updated` is published after a successful write.
10. The frontend handles `session.title_updated` by refreshing the session list.

## Acceptance

A production end-to-end run must show that:

- the first user message receives a normal main-model answer;
- a concise generated title appears in the left session list;
- the title was generated with the configured small model;
- the event stream contains `session.title_updated`;
- the session title persists after page reload;
- subsequent turns do not replace it.
