# OpenClaw Archaeologist Reference

Pre-loaded knowledge about the OpenClaw architecture to accelerate deep analysis.

## Core Architecture Concepts

### The Gateway
- Single long-lived daemon that owns ALL messaging surfaces
- Runs on `localhost:18789` by default (configurable)
- Exposes: WebSocket API (control plane) + HTTP (canvas, WebChat, Control UI)
- Invariant: exactly one Gateway per host for any given Baileys (WhatsApp) session

### The Plugin SDK (`openclaw/plugin-sdk`)
The `package.json` exports **100+ named entry points** under `./plugin-sdk/*`. This is the
**public, versioned API** for third-party plugin authors. Key SDK modules include:
- `channel-runtime` — base class/contract for channel adapters
- `channel-contract` / `channel-core` — channel interface definitions
- `agent-runtime` — agent lifecycle hooks from plugin perspective
- `provider-stream` / `provider-auth` — LLM provider abstractions
- `reply-runtime` / `reply-chunking` / `reply-dispatch-runtime` — message delivery pipeline
- `hook-runtime` — plugin hook registration
- `approval-runtime` — exec approval gating
- `memory-core` — memory/storage subsystem
- `config-runtime` / `config-schema` — config access + schema extension
- `ssrf-policy` / `ssrf-runtime` — SSRF protection for network tools

**Important for analysis:** When a user asks "how do channels work" or "how are plugins built",
the plugin-sdk export map in `package.json` is the single best index of what surface area exists.
Each `./plugin-sdk/<module>` entry corresponds to a conceptual unit of the system.


- The actual LLM runtime is called "Pi agent core" (`runEmbeddedPiAgent`)
- OpenClaw wraps Pi agent core: session management, channel delivery, tool wiring, and event subscription are OpenClaw layers
- Session transcripts stored as JSONL at `~/.openclaw/agents/<agentId>/sessions/<SessionId>.jsonl`

### Channel Adapters
Each channel (WhatsApp/Telegram/Discord/etc) is a separate adapter. Key channels:
- **WhatsApp**: uses Baileys library (unofficial Web API)
- **Telegram**: uses grammY library
- **Discord, Slack**: official bot APIs
- **Signal, iMessage, Matrix**: third-party bridges

DM policy pattern (shared by all channels):
```
dmPolicy: "pairing" | "allowlist" | "open" | "disabled"
allowFrom: ["+15555550123"]  // or ["tg:123"] or ["*"]
```

### WebSocket Wire Protocol
- Transport: WebSocket, text frames, JSON payloads
- First frame MUST be `connect` (identity + auth + caps)
- After handshake: `{type:"req", id, method, params}` → `{type:"res", id, ok, payload|error}`
- Server-push: `{type:"event", event, payload, seq?, stateVersion?}`
- Clients: role = (unset / default) — control plane operators
- Nodes: role = "node" — iOS/Android/macOS devices with camera/screen/voice

### Session Model
- Session key format: `agent:<agentId>:<channel>:<peer>`
- Scope controlled by `session.dmScope`: `main` | `per-peer` | `per-channel-peer` | `per-account-channel-peer`
- Runs are serialized per session lane (prevents tool/session races)
- `agent` RPC returns `{ runId, acceptedAt }` immediately; `agent.wait` blocks for lifecycle end

### Agent Loop Entry Points
1. Gateway RPC: `agent` and `agent.wait`
2. CLI: `openclaw agent --message "..."` 

Full loop:
```
agent RPC → validate → resolve session → agentCommand → runEmbeddedPiAgent
  → serialize via queue → build system prompt → call Pi core
  → subscribe to events (tool/assistant/lifecycle) → enforce timeout
  → return payloads + usage
```

### Bootstrap Files (injected into system prompt)
In `agents.defaults.workspace` (~/.openclaw/workspace):
- `AGENTS.md` — operating instructions + memory
- `SOUL.md` — persona, boundaries, tone
- `TOOLS.md` — user-maintained tool notes
- `BOOTSTRAP.md` — one-time first-run ritual (deleted after completion)
- `IDENTITY.md` — agent name/vibe/emoji
- `USER.md` — user profile

### Skills System
- Format: AgentSkills-compatible (`SKILL.md` with YAML frontmatter + instructions)
- Load precedence (highest → lowest):
  1. `<workspace>/skills`
  2. `<workspace>/.agents/skills`
  3. `~/.agents/skills`
  4. `~/.openclaw/skills`
  5. Bundled skills (npm package)
  6. `skills.load.extraDirs`
- Skills are snapshotted at session start; changes take effect on next new session
- Gating: `metadata.openclaw.requires.bins/env/config` at load time

### Plugin Hook System
Hooks run inside the agent loop at defined lifecycle points:
- `before_model_resolve` — override provider/model before session
- `before_prompt_build` — inject context before prompt submission
- `before_agent_reply` — claim a turn or return synthetic reply
- `before_tool_call` / `after_tool_call` — intercept tools
- `agent_end` — post-completion inspection
- `message_received` / `message_sending` / `message_sent`
- `session_start` / `session_end`
- `gateway_start` / `gateway_stop`

### Compaction
When context gets long, auto-compaction summarizes the conversation history.
Emits `compaction` stream events. On retry, buffers/summaries are reset.

### Queue Modes (for channel message handling)
- `collect` — hold messages until current turn ends, then process together
- `steer` — inject into current run (delivered after current tool calls finish)
- `followup` — hold messages, start new turn after current completes

### Block Streaming
- Default: off (`agents.defaults.blockStreamingDefault: "off"`)
- When on: emits partial replies as `text_end` or `message_end`
- Block size: 800-1200 chars, prefers paragraph/newline breaks
- `blockStreamingCoalesce`: merges chunks before send to reduce spam

### Config System
- File: `~/.openclaw/openclaw.json` (JSON5)
- Hot reload: `hybrid` mode by default (most changes apply without restart)
- Restart required only for: `gateway.*` (port/auth/TLS) and infrastructure
- Config validated strictly against TypeBox schema on startup

### Multi-Agent Routing
```json
{
  "agents": { "list": [{ "id": "home", "workspace": "..." }, { "id": "work", "workspace": "..." }] },
  "bindings": [
    { "agentId": "home", "match": { "channel": "whatsapp", "accountId": "personal" } },
    { "agentId": "work", "match": { "channel": "whatsapp", "accountId": "biz" } }
  ]
}
```

### Sandboxing
- Docker-based sandbox: `agents.defaults.sandbox.mode: "non-main" | "all"`
- Default sandbox allows: bash, process, read, write, edit, sessions tools
- Default sandbox denies: browser, canvas, nodes, cron, discord, gateway

### Pairing & Device Trust
- All WS clients include device identity on `connect`
- New device IDs require pairing approval
- Gateway issues device token for subsequent connects
- Signature payload `v3` binds platform + deviceFamily; metadata pinned on reconnect

## TypeScript Build Setup
- Builder: `tsdown` (rollup-based, produces ESM)
- Multiple tsconfig files for different build targets:
  - `tsconfig.core.json` — core source
  - `tsconfig.extensions.json` — extension system
  - `tsconfig.test.json` — test files
- Type generation: TypeBox schemas → JSON Schema → Swift models (for iOS app)
- Test runner: Vitest
- Linter: oxlint + oxfmt
- Dead code: knip

## Key External Dependencies

| Package | Role |
|---------|------|
| `baileys` | WhatsApp Web protocol implementation |
| `grammy` | Telegram bot framework |
| `@sinclair/typebox` | Runtime type checking + schema generation |
| TypeScript + tsx | Language + dev-mode runner |
| `tsdown` | Production bundler |
| `vitest` | Test runner |
| `zizmor` | GitHub Actions security scanner |
