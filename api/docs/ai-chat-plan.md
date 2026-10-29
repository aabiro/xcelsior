# DONE

**Approx. completed:** 2026-06-12

---

## Plan: Xcelsior AI Assistant — In-Website Chat

Build "Xcel AI" — a full-featured AI assistant at `/dashboard/ai` that can answer questions (RAG over docs), perform platform actions (launch jobs, check billing, manage hosts), onboard both renters and providers, and pin to a right-side context panel. Uses Claude 3.5 Haiku with native tool-calling, streams via SSE, and builds on the existing chat infrastructure. Feature-flagged behind `FEATURE_AI_ASSISTANT`, developed on `feature/ai-assistant` branch.

---

### Phase 1: Backend — AI Chat Engine with Tool-Calling

**Step 1.1: New module `ai_assistant.py` — Tool registry & orchestration** *(no dependencies)*

- **Tool registry**: Dict mapping tool names → async handler functions + JSON schema descriptions
- **15 tools**:
  - *Read-only*: `get_account_info`, `get_billing_summary`, `list_jobs`, `get_job_details(job_id)`, `search_marketplace`, `get_pricing`, `get_host_status`, `get_reputation`, `search_docs(query)`, `recommend_gpu(workload)`, `estimate_cost(gpu, count, hours)`, `get_sla_terms(tier)`
  - *Write (require confirmation)*: `launch_job`, `stop_job(job_id)`, `create_api_key(name, scope)`
- **System prompt builder**: Extends `build_system_prompt()` from chat.py with user context injection (role, reputation tier, active jobs, balance), tool descriptions in Anthropic format, renter/provider/both persona branching
- **Confirmation state machine**: Tracks pending write-action approvals per conversation
- **Token metering**: Count input/output tokens for cost monitoring
- Reuses patterns from: scheduler.py → `submit_job()`, marketplace.py → `search_offers()`, billing.py → `BillingEngine`, reputation.py → `compute_score()`, sla.py → tier terms, db.py → `UserStore`

**Step 1.2: Anthropic SDK integration** *(depends on 1.1)*

- Add `anthropic>=0.39.0` to requirements.txt
- New env vars: `ANTHROPIC_API_KEY`, `AI_ASSISTANT_MODEL` (default: `claude-3-5-haiku-20241022`), `AI_ASSISTANT_MAX_TOKENS` (default: 4096)
- Implement `stream_assistant_response()` async generator using `anthropic.AsyncAnthropic`:
  - Sends messages with `tools=` parameter
  - When model returns `tool_use` block → execute read-only tool automatically, or yield `confirmation_required` event for write tools → feed result back → continue streaming
  - Yields typed SSE events: `token`, `tool_call`, `tool_result`, `confirmation_required`, `meta`, `done`
- Fallback: If Anthropic fails → degrade to existing Groq/OpenAI chat from chat.py (Q&A only, no tool-calling)

**Step 1.3: API endpoints** *(depends on 1.1, 1.2)*

5 new routes in api.py:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/ai/chat` | Main streaming endpoint (SSE, auth required, 20 req/min per user) |
| `POST` | `/api/ai/confirm/{confirmation_id}` | Approve/reject a pending write action |
| `GET` | `/api/ai/conversations` | List user's AI conversations |
| `GET` | `/api/ai/conversations/{id}` | Fetch conversation history (ownership check) |
| `DELETE` | `/api/ai/conversations/{id}` | Delete conversation |
| `GET` | `/api/ai/suggestions` | Context-aware starter chips (different for renter/provider/new user) |

All routes return 404 when `FEATURE_AI_ASSISTANT=false`.

**Step 1.4: Database migration `006_ai_assistant.py`** *(parallel with 1.1)*

New Alembic migration with 4 tables:

- `ai_conversations` — `conversation_id PK, user_id FK→users, title, created_at, updated_at, message_count, total_input_tokens, total_output_tokens, metadata JSONB`
- `ai_messages` — `message_id PK, conversation_id FK, role (user/assistant/tool_call/tool_result), content, tool_name, tool_input JSONB, tool_output JSONB, tokens_in, tokens_out, created_at` — cascading delete
- `ai_confirmations` — `confirmation_id PK, conversation_id FK, user_id, tool_name, tool_args JSONB, status (pending/approved/rejected/expired), created_at, resolved_at`
- `ai_docs` — `doc_id SERIAL, source TEXT, chunk TEXT, tsv tsvector GENERATED ALWAYS`  with GIN index for BM25 search

Separate tables from existing `chat_*` tables — different schema needs (tool calls, confirmations, token tracking).

**Step 1.5: RAG — Documentation search** *(parallel with 1.1)*

- BM25 full-text search via PostgreSQL `tsvector` on `ai_docs` table (no vector DB needed yet)
- Ingest script: Chunks llms.txt + `docs/*.md` at ~500 tokens per chunk with 50-token overlap
- `search_docs()` tool queries with `ts_rank` and returns top-5 chunks
- Future upgrade path: Add pgvector embeddings if BM25 retrieval quality is insufficient

**Step 1.6: Feature flag & env config** *(parallel with anything)*

- Add to .env.example: `FEATURE_AI_ASSISTANT`, `ANTHROPIC_API_KEY`, `AI_ASSISTANT_MODEL`, `AI_ASSISTANT_MAX_TOKENS`, `AI_ASSISTANT_RATE_LIMIT`
- .env.test: `FEATURE_AI_ASSISTANT=true`
- .env (prod): `FEATURE_AI_ASSISTANT=false` until ready
- Pass through in docker-compose.yml

---

### Phase 2: Frontend — AI Chat Page & Context Panel

**Step 2.1: AI Chat page — `/dashboard/ai`** *(depends on Phase 1)*

Create frontend/src/app/(dashboard)/dashboard/ai/page.tsx/dashboard/ai/page.tsx):

**Landing state** (no messages):
- Centered vertically, Xcel logo mark + "Xcel AI" heading
- Tagline: "What can I help you build?"
- Large rounded input (max-w-2xl, rounded-full, 48px tall) with send button
- Disclaimer: "Xcel AI can make mistakes. Please verify important information."
- Suggestion chips from `GET /api/ai/suggestions`:
  - Renter: "What GPU for fine-tuning?", "Launch a training job", "Check my usage"
  - Provider: "How much can I earn?", "Set up my hosting", "Check my uptime"
  - New user: "🖥️ Rent Compute", "🏗️ Provide GPUs", "🔄 Both"
- Chips: `bg-surface-hover` + border, hover → `bg-accent-cyan/8`

**Chat state** (after first message):
- Input moves to bottom (sticky), messages scroll above
- User messages: right-aligned `bg-accent-cyan/8`
- Assistant messages: left-aligned `bg-surface`, typewriter streaming
- Tool calls: Inline cards — pending shows "🔄 Searching marketplace..." with pulse, completed shows collapsible result card (GPU table, job status, billing summary)
- **Confirmation cards**: Action description + params table + "Approve" (green) / "Cancel" (gray) buttons → `POST /api/ai/confirm/{id}`
- Markdown + syntax highlighting (reuse from ChatWidget.tsx)
- "Stop generating" button during streaming, auto-scroll

**Header**: "Xcel AI" + beta badge, "New conversation", "Open in panel" (pin icon), conversation history (clock icon)

**Step 2.2: Context panel mode** *(depends on 2.1)*

- "Open in panel" pins chat to right side of dashboard layout (w-96, border-l)
- Layout splits: main content (flex-1) + AI panel
- Panel persists across page navigation (rendered in layout.tsx/layout.tsx))
- "Back to full page" returns to `/dashboard/ai`
- State in `localStorage: xcelsior-ai-panel-open`
- Existing ChatWidget bubble gets "Open Xcel AI" option

**Step 2.3: Nav item & integration** *(depends on 2.1)*

- Add to sidebar `navItems` in layout.tsx/layout.tsx): `{ href: "/dashboard/ai", key: "dash.ai", icon: Sparkles }`
- Position: After main nav items, before Help at bottom

**Step 2.4: Streaming hook `useAiChat.ts`** *(parallel with 2.1)*

New hook extending useChatStream.ts pattern:
- Handles `tool_call`, `tool_result`, `confirmation_required` SSE event types
- Renders inline loading → result cards on tool calls
- Confirmation card management (approve/reject → API call)
- Conversation CRUD (new, load history, delete)
- AbortController for mid-stream cancel
- Retry on network failure

**Step 2.5: i18n** *(parallel)*

~40 new keys in both en.ts and fr.ts: `ai.tagline`, `ai.subtitle`, `ai.placeholder`, `ai.disclaimer`, `ai.send`, `ai.stop`, `ai.new_conversation`, `ai.open_panel`, `ai.confirm_action`, `ai.approve`, `ai.cancel`, `dash.ai`, chip labels, error messages.

---

### Phase 3: Onboarding — Renter vs Provider Detection

**Step 3.1: Smart onboarding detection** *(depends on Phase 1 tools)*

- AI checks user profile on first conversation (role, has hosts, has jobs)
- Brand new user → "Welcome! Are you looking to rent GPU compute, provide your GPUs, or both?" + three chips
- Existing provider → loads host data automatically
- Existing renter → loads job history

**Step 3.2: Provider wizard (in-chat)** *(depends on 3.1)*

- AI asks for hardware specs (or user pastes `nvidia-smi` output)
- Parses GPU model, VRAM, count, driver version from natural language or CLI output
- Calls `search_marketplace` to recommend competitive pricing
- Estimates monthly revenue from current demand data
- Generates bootstrap command: `curl -sSL https://xcelsior.ca/setup | XCELSIOR_KEY=xcel_abc123 bash`
- Walks through: Docker, NVIDIA toolkit, firewall ports
- Monitors node coming online via SSE events → "✅ 2× RTX 4090 detected. You're live!"

**Step 3.3: Renter wizard (in-chat)** *(depends on 3.1)*

- AI asks about workload (training, inference, batch)
- Calls `recommend_gpu` with workload description → suggests optimal config
- Calls `estimate_cost` → shows pricing comparison
- Can directly launch a test job via `launch_job` tool (with confirmation)
- Generates example Dockerfile/config inline

---

### Phase 4: Testing & Quality

**Step 4.1: Backend tests — `tests/test_ai_assistant.py`** *(parallel with Phase 1)*

~25 test cases:
- Tool registry completeness (all tools registered, schemas valid JSON)
- Each tool handler with mock DB data (correct result format)
- System prompt generation (includes user context, tool list)
- Conversation CRUD (create, list, get, delete, ownership)
- Confirmation flow (pending → approved → executed; pending → rejected → cancelled)
- Rate limiting (per-user, 21st request in 60s → 429)
- Feature flag (404 when `FEATURE_AI_ASSISTANT=false`)
- Streaming response format (valid SSE events, parseable JSON)
- Token metering (counts match Anthropic response usage)
- RAG search (BM25 returns relevant chunks, empty query handled)
- PII redaction in prompts
- Mock Anthropic API for deterministic responses

**Step 4.2: Frontend tests** *(parallel with Phase 2)*

- Landing state render (heading, input, chips appear)
- Chat state transition (send message → input moves to bottom)
- Tool call card rendering (loading → result)
- Confirmation flow (card appears → approve → result renders)
- `useAiChat` hook (SSE parsing, event types, abort)

**Step 4.3: Integration tests** *(after Phase 1 + 2)*

- E2E: "What's my balance?" → streams with billing tool → correct amount
- E2E: "Launch a 2×A100 job" → confirmation card → approve → job in DB
- Fallback: Mock Anthropic down → Groq fallback works (Q&A only)
- All existing 1118+ tests still pass

---

### Phase 5: Deployment

**Step 5.1**: `git checkout -b feature/ai-assistant` — all dev on this branch

**Step 5.2**: Deploy to test env with `FEATURE_AI_ASSISTANT=true`, verify end-to-end

**Step 5.3**: Production rollout — flip flag to `true`, initial admin-only, then all users

---

### Relevant Files

**Backend — Create:**
- `ai_assistant.py` — Tool registry, system prompt, orchestration, streaming
- `migrations/versions/006_ai_assistant.py` — 4 new tables

**Backend — Modify:**
- api.py — Add 6 new `/api/ai/*` endpoints
- requirements.txt — Add `anthropic>=0.39.0`

**Frontend — Create:**
- `frontend/src/app/(dashboard)/dashboard/ai/page.tsx` — AI chat page
- `frontend/src/hooks/useAiChat.ts` — Streaming hook with tool-call support
- `frontend/src/components/ui/ai-chat-panel.tsx` — Reusable chat panel (page & side panel)
- `frontend/src/components/ui/ai-confirmation-card.tsx` — Action confirmation card
- `frontend/src/components/ui/ai-tool-result.tsx` — Tool result display cards

**Frontend — Modify:**
- frontend/src/app/(dashboard)/layout.tsx/layout.tsx) — Nav item + side panel integration
- en.ts — ~40 AI keys
- fr.ts — ~40 AI keys
- api.ts — AI chat API functions

**Config — Modify:**
- .env.example — AI assistant env vars
- docker-compose.yml — Pass through new env vars

**Tests — Create:**
- `tests/test_ai_assistant.py` — ~25 backend test cases

---

### Verification

1. `FEATURE_AI_ASSISTANT=false` → all `/api/ai/*` return 404
2. `FEATURE_AI_ASSISTANT=true` → `/dashboard/ai` loads with suggestions
3. "What's my balance?" → streams with billing tool call → correct balance
4. "Launch a 2×A100 job" → confirmation card → approve → job appears in instances
5. "What GPU for Llama fine-tuning?" → `search_docs` + `recommend_gpu` → actionable recommendation
6. Provider "How much can I earn?" → `search_marketplace` → revenue estimate
7. 21st message in 60s → rate limit error
8. Kill Anthropic → Groq fallback (degraded, no tools)
9. All existing tests pass (1118+)
10. `docker compose build` succeeds
11. French locale → all AI strings render in French
12. "Open in panel" → chat pins right → persists across page navigation

### Decisions

- **Model**: Claude 3.5 Haiku — cheapest with native `tool_use` (~$0.25/$1.25 per 1M tokens in/out)
- **RAG**: PostgreSQL BM25 `tsvector` first (no vector DB) — upgrade to pgvector later if needed
- **Feature flag**: `FEATURE_AI_ASSISTANT` env var — simple, works with existing deploy
- **Storage**: Separate `ai_*` tables — NOT reusing `chat_*` tables (different schema for tool calls, confirmations, tokens)
- **Existing ChatWidget**: Kept as-is for lightweight support; Xcel AI is the full-powered version
- **Branch**: `feature/ai-assistant`
- **Write actions**: ALWAYS require explicit user confirmation — never auto-execute
- **Token billing**: Track internally, don't charge users initially — monitor costs before adding billing
- **CLI wizard**: Separate future phase (not in this plan) 
