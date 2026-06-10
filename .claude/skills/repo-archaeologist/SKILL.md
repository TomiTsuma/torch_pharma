---
name: repo-archaeologist
description: >
  Deep-dives into how a repository's code actually works — tracing data flow, control flow,
  key abstractions, design patterns, and architectural decisions. Produces a thorough technical
  analysis that explains WHY things are structured the way they are, not just WHAT exists.
  Use this skill whenever the user wants to understand the internals of a codebase, trace how
  a feature works end-to-end, understand a specific subsystem, or needs a developer-level
  explanation of architecture. Trigger for: "how does X work", "trace the flow of", "explain
  the architecture", "what does this module do", "how is X implemented", "walk me through",
  "dig into", "deep dive", "explain how the agent loop works", "how does the gateway handle",
  "what's the session model", "understand the internals", or any follow-up after repo-cartographer.
  Should run AFTER repo-cartographer (or use its manifest if already available).
---

# Repo Archaeologist

The second skill in the analysis chain. While `repo-cartographer` maps *what exists*,
this skill explains *how it works* — the flows, abstractions, and design rationale.

---

## Prerequisites

Check if a Repo Inventory Manifest from `repo-cartographer` exists. If yes, load it for context.
If no, offer to run the cartographer first OR proceed with a focused investigation on the
specific area the user asked about.

---

## Investigation Modes

The archaeologist operates in one of three modes depending on what the user asked:

### Mode A: Full System Architecture Analysis
*Triggered by: "explain the architecture", "how does the whole thing work", "deep dive the codebase"*

Run all phases below in order.

### Mode B: Subsystem Deep Dive
*Triggered by: "how does the gateway work", "explain the channel system", "how are sessions handled"*

Run only Phase 2 (targeted reading) and Phase 3 (flow tracing) for the named subsystem.
Skip broad scanning.

### Mode C: Feature Trace
*Triggered by: "trace what happens when a WhatsApp message arrives", "how does a tool call work"*

Start from the named entrypoint/trigger and follow the execution path end-to-end.

---

## Phase 1 — Structural Deep Scan (Mode A only)

Read each major directory's contents one level deep, then sample key files:

```bash
# Get the layout of core source directories
ls -la src/
ls -la packages/
ls -la extensions/

# For each subdir in src/, list contents
for d in src/*/; do echo "=== $d ==="; ls "$d" 2>/dev/null; done
```

Then sample key files for each major subsystem found (read top 60-100 lines of each):
```bash
# Example for a gateway-style codebase
head -80 src/gateway/index.ts 2>/dev/null
head -80 src/agent/index.ts 2>/dev/null
```

---

## Phase 2 — Targeted File Reading

### Identifying the key files to read

For each subsystem under investigation, find the core files:

```bash
# Find the main entry/export for a module
grep -r "export " src/<subsystem>/ --include="*.ts" -l | head -10
# Find where major types/interfaces are defined
grep -r "^export (type|interface|class)" src/ --include="*.ts" -l | head -20
# Find the main class or function for a subsystem
grep -rn "class Gateway\|class Agent\|class Session\|class Channel" src/ --include="*.ts" | head -10
```

### Reading strategy

Read files in this order:
1. **Type/interface definitions** — these define the contract
2. **Main class / function implementations** — these define the behavior
3. **Factory functions / constructors** — these reveal how things are wired together
4. **Event emitters / listeners** — these reveal the async communication pattern
5. **Test files** — these often reveal intent and edge cases more clearly than the impl

```bash
# Read a file fully (for files < 200 lines)
cat src/<path>/<file>.ts

# Read a large file in sections
head -100 src/<path>/<file>.ts
sed -n '100,200p' src/<path>/<file>.ts
grep -n "^export\|^class\|^function\|^const\|^type\|^interface" src/<path>/<file>.ts
```

### For OpenClaw specifically — key files to always read

**The plugin-sdk export map as a code index:**
The `package.json` exports 100+ named `./plugin-sdk/*` paths. Each is a conceptual unit.
Before reading source files, use this as an index of what exists:
```bash
cat package.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
exports = d.get('exports', {})
sdk_paths = sorted([k for k in exports if 'plugin-sdk' in k])
from collections import Counter
prefixes = Counter()
for k in sdk_paths:
    seg = k.replace('./plugin-sdk/', '').split('-')[0]
    prefixes[seg] += 1
print('SDK subsystems by prefix count:')
for k, v in prefixes.most_common(20):
    print(f'  {k}: {v} modules')
" 2>/dev/null
```

**Gateway core:**
```bash
head -100 src/gateway/index.ts 2>/dev/null || find src -name "gateway*" -type f | head -5
```

**Agent/session management:**
```bash
find src -name "*agent*" -o -name "*session*" | grep -v node_modules | grep "\.ts$" | head -10
```

**Channel system:**
```bash
find src -name "*channel*" -type f | grep "\.ts$" | head -10
ls src/channels/ 2>/dev/null
```

**WebSocket protocol:**
```bash
find src -name "*ws*" -o -name "*websocket*" -o -name "*protocol*" | grep "\.ts$" | head -10
```

**Skills loading:**
```bash
find src -name "*skill*" | grep "\.ts$" | head -10
```

---

## Phase 3 — Flow Tracing

This is the core analytical work. Trace the execution flow for the most important paths.

### How to trace a flow

1. Identify the **trigger** (an incoming message, a CLI command, an RPC call)
2. Find the **handler** (the function that receives it)
3. Follow each function call, reading the next function's signature + body
4. Note **where state is written** and **where it is read later**
5. Identify **async boundaries** (Promises, event emitters, queues)
6. Find the **terminal action** (what actually happens at the end)

```bash
# Find where a trigger is handled
grep -rn '"agent"\|"send"\|"connect"' src/ --include="*.ts" | grep "case\|method\|route\|handler" | head -20

# Trace call chains
grep -rn "runEmbeddedPiAgent\|agentCommand\|handleMessage" src/ --include="*.ts" | head -20

# Find event listeners
grep -rn "\.on(\|\.emit(\|addEventListener" src/ --include="*.ts" | head -30
```

### Standard flows to trace for a gateway/agent system like OpenClaw

**Flow 1: Inbound message → agent response**
```
[Channel adapter receives message]
  → message normalization
  → session resolution (which agent? which session?)
  → queue/concurrency check
  → agent loop start
    → context assembly (system prompt, skills, history)
    → LLM call
    → tool execution (if any)
    → reply streaming
  → channel delivery
```

**Flow 2: WebSocket client connection**
```
[Client connects to WS server]
  → first frame must be "connect"
  → device identity validation
  → pairing check (known device?)
  → auth token verification
  → capabilities negotiation (hello-ok)
  → event subscription setup
```

**Flow 3: Tool call execution**
```
[LLM returns tool_use block]
  → tool name resolution
  → before_tool_call hooks
  → tool execution (bash/read/write/etc)
  → result sanitization
  → after_tool_call hooks
  → result injected back into context
  → LLM called again with tool result
```

---

## Phase 4 — Abstraction & Pattern Analysis

After reading and tracing, identify the key design decisions:

### Dependency injection / wiring
```bash
# How are components assembled?
grep -rn "new Gateway\|createGateway\|setup\|init\|bootstrap" src/ --include="*.ts" | head -20
```

### Error handling patterns
```bash
grep -rn "try {" src/ --include="*.ts" | wc -l
grep -rn "catch (e\|catch (err\|catch (error" src/ --include="*.ts" | head -10
grep -rn "Result<\|Either<\|\.ok\|\.err" src/ --include="*.ts" | head -10
```

### State management
```bash
# Where is mutable state kept?
grep -rn "let \|var \|this\." src/ --include="*.ts" | grep -v "test\|spec" | head -20
# Is there a central store?
grep -rn "Store\|state\|Map<\|Record<" src/ --include="*.ts" -l | head -10
```

### Concurrency model
```bash
grep -rn "queue\|mutex\|lock\|semaphore\|serialize\|await\|Promise\.all" src/ --include="*.ts" | head -20
```

### Extension points
```bash
grep -rn "hook\|plugin\|middleware\|interceptor\|register\|extend" src/ --include="*.ts" -l | head -10
```

---

## Phase 5 — Produce the Technical Analysis Document

Write a detailed analysis document. Structure it based on what was found.

### Document format

```markdown
# Technical Analysis: <repo-name> — <subsystem or "Full Architecture">

**Analysis date:** <date>
**Scope:** <what was analyzed>

---

## 1. Mental Model

<The single best analogy or mental model for understanding this system.
E.g.: "Think of OpenClaw as a router/hub. The Gateway is the hub — it
owns all messaging surfaces. Clients (macOS app, CLI) and Nodes (iOS, Android)
connect to it over WebSocket. The agent runtime is the brain that processes
messages and drives responses back out through the correct channel.">

## 2. Core Abstractions

For each major concept, explain: what it is, how it's represented in code, and what it does.

### <Abstraction Name> (e.g. "Session")
- **What it is:** <plain English>
- **Code representation:** `<TypeName>` in `<file.ts>`
- **Lifecycle:** <created when / destroyed when>
- **Key operations:** <list>
- **Important notes:** <gotchas, invariants, constraints>

### <Next Abstraction>
...

## 3. Key Flows

### Flow: <Name> (e.g. "Inbound WhatsApp message → Agent response")

Describe the flow as numbered steps with code references:

1. **Message received** — `channels/whatsapp/index.ts:handleMessage()` receives the raw Baileys event
2. **Normalization** — converted to OpenClaw's internal `InboundMessage` type
3. **Session resolution** — `SessionManager.resolveSession()` finds or creates the session
   - Session key format: `agent:<agentId>:<channel>:<peer>`
   - Scope determined by `session.dmScope` config
4. **Queue check** — serialized via per-session lane to prevent races
5. **Agent loop** — `runEmbeddedPiAgent()` called with session context
   - Loads skills snapshot
   - Assembles system prompt (AGENTS.md + SOUL.md + skills + bootstrap files)
   - Calls Pi agent core runtime
6. **LLM call** — model selected from `agents.defaults.model`, auth profile resolved
7. **Tool execution** — tools run one at a time, results fed back to model
8. **Reply streaming** — assistant deltas emitted as `agent` stream events
9. **Channel delivery** — final reply sent back via the originating channel adapter

### Flow: <Next Flow>
...

## 4. Data Flow Diagram

(Described in text since we can't draw here — the documentarian skill will render this)

```
[External channel] 
  → ChannelAdapter (normalize + emit)
  → Gateway (route to agent)
    → AgentRunner (session + queue)
      → EmbeddedPiAgent (LLM + tools)
        → ToolExecutor
      ← tool results
    ← assistant reply
  → ChannelAdapter (send reply)
[External channel]
```

## 5. Design Patterns Identified

| Pattern | Where used | Why |
|---------|-----------|-----|
| Event emitter | Gateway ↔ channels | Decouples channel lifecycle from core |
| Command queue | Per-session serialization | Prevents tool/session races |
| Plugin hooks | agent loop lifecycle | Extension without modifying core |
| Strategy | Channel adapters | Uniform interface across messaging platforms |
| Observer | WS event subscriptions | Clients subscribe to only relevant events |

## 6. Key Invariants & Constraints

Things that MUST always be true for the system to work correctly:

1. **One Gateway per host** — exactly one Baileys session per host; two gateways = WhatsApp conflict
2. **First WS frame must be `connect`** — any non-connect frame = hard close
3. **Session write lock** — `SessionManager` acquires write lock before streaming begins
4. **Idempotency keys** — side-effecting RPCs (`send`, `agent`) require idempotency keys for safe retry
5. <others>

## 7. Complexity Hotspots

Areas that are dense, subtle, or high-risk to change:

- **`<filename>:<linerange>`** — <why it's complex>
- **`<filename>:<linerange>`** — <why it's complex>

## 8. Extension Points

How to extend or hook into the system:

| Extension type | How | Where |
|---------------|-----|-------|
| New channel adapter | Implement ChannelAdapter interface | `src/channels/<name>/` |
| New tool | Register in tool registry | `src/tools/` |
| Skill | Add `SKILL.md` folder | `~/.openclaw/workspace/skills/` |
| Plugin hook | Register `before_tool_call` etc | Plugin architecture |
| Cron job | Add to `cron` config | `openclaw.json` |

## 9. What's Not Obvious (Gotchas)

Things a developer would only learn after reading the code carefully or hitting a bug:

1. **<gotcha>** — <explanation>
2. **<gotcha>** — <explanation>
```

---

## Output & Handoff

After producing the analysis, present it to the user. Then offer:

> "Deep analysis complete. I can now:
> - **Trace a specific flow** in more detail (just ask about it)
> - **Generate documentation** from this analysis (uses `repo-documentarian`)
> - **Explain a specific file or function** you're curious about
> - **Draw a diagram** of any part of the architecture"

---

## Quality Checklist

Before presenting the analysis:
- [ ] Every major abstraction is explained in plain English before code references
- [ ] At least 2-3 complete flows are traced end-to-end with code references
- [ ] Invariants/constraints are explicitly listed (not buried in prose)
- [ ] Gotchas section exists and has at least 2-3 non-obvious items
- [ ] Mental model analogy is concrete and accurate
- [ ] Analysis depth matches what the user asked for (don't over-explain for a focused question)
