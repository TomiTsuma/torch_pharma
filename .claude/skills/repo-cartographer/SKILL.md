---
name: repo-cartographer
description: >
  Maps and inventories a software repository before any deep analysis — classifying files by
  role, identifying tech stack, detecting architectural patterns, and producing a structured
  inventory manifest that downstream skills (repo-archaeologist, repo-documentarian) can consume.
  Use this skill FIRST whenever the user wants to understand, explore, or document any codebase.
  Trigger for: "understand this repo", "map this codebase", "what does this project do",
  "explain this repository", "inventory this project", "give me an overview", "what's in this repo",
  "explore the codebase", "help me understand the structure", or any request to analyze a GitHub URL
  or local directory. Always run this before repo-archaeologist or repo-documentarian.
---

# Repo Cartographer

The first step in understanding any codebase. This skill produces a **Repo Inventory Manifest** —
a structured, human-readable document that answers: "What IS this thing?"

Downstream skills (repo-archaeologist, repo-documentarian) consume this manifest.
Do not skip this skill when a full repo analysis is requested.

---

## Phase 0 — Source Acquisition

### GitHub URL provided
```bash
# Clone to a temp directory
git clone <url> /tmp/repo-analysis/<repo-name> --depth=1
cd /tmp/repo-analysis/<repo-name>
```

### Local path provided
```bash
cd <provided-path>
```

If the user provides neither, ask: "Where is the repository? (GitHub URL or local path)"

---

## Phase 1 — Directory Tree Scan

Run the tree scan to get a full picture of the layout:

```bash
# Full tree, ignoring noise
find . -type f \
  -not -path './.git/*' \
  -not -path '*/node_modules/*' \
  -not -path '*/.venv/*' \
  -not -path '*/dist/*' \
  -not -path '*/build/*' \
  -not -path '*/__pycache__/*' \
  -not -path '*/.next/*' \
  -not -path '*/target/*' \
  | sort | head -300

# Top-level directory structure
ls -la
```

Also run:
```bash
# Count files by extension to understand language mix
find . -type f -not -path './.git/*' -not -path '*/node_modules/*' \
  | grep -oE '\.[a-zA-Z0-9]+$' | sort | uniq -c | sort -rn | head -30
```

---

## Phase 2 — Tech Stack Identification

Read every manifest file present. For each one found, extract the key information listed below.

### JavaScript / TypeScript ecosystem
```bash
# IMPORTANT: package.json can be very large (1000+ lines in plugin-heavy projects).
# DO NOT cat the whole file — use targeted extraction instead:
cat package.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('name:', d.get('name'))
print('version:', d.get('version'))
print('description:', d.get('description'))
print('bin:', d.get('bin', {}))
print('main:', d.get('main', ''))
print('module:', d.get('module', ''))
print('type:', d.get('type', ''))
export_keys = list(d.get('exports', {}).keys())
print('export count:', len(export_keys))
print('top exports:', export_keys[:10])
sdk_exports = [k for k in export_keys if 'sdk' in k or 'plugin' in k]
print('sdk/plugin exports:', len(sdk_exports))
print('scripts:', list(d.get('scripts', {}).keys())[:20])
deps = list(d.get('dependencies', {}).keys())
print('top deps:', deps[:15])
" 2>/dev/null

# Workspace config
cat pnpm-workspace.yaml 2>/dev/null
cat pnpm-lock.yaml 2>/dev/null | head -10
ls tsconfig*.json 2>/dev/null
cat tsconfig.json 2>/dev/null
```
Extract: `name`, `version`, `bin` (entrypoints), `scripts`, top `dependencies`, export map structure (note if there's a plugin-sdk or large export surface), workspace `packages`.

**Red flag:** If `exports` has 50+ entries, the package has a published SDK with a large API surface — note this explicitly in the manifest under "Key Subsystems".



### Python ecosystem
```bash
cat pyproject.toml 2>/dev/null
cat setup.py 2>/dev/null
cat requirements.txt 2>/dev/null | head -30
cat Pipfile 2>/dev/null
```
Extract: build backend (`hatchling`/`setuptools`/`poetry`), entry points, key dependencies.

### Rust
```bash
cat Cargo.toml 2>/dev/null
ls Cargo.lock 2>/dev/null
```
Extract: crate name, edition, binary targets, workspace members.

### Go
```bash
cat go.mod 2>/dev/null
cat go.sum 2>/dev/null | head -5
```
Extract: module path, Go version, key requires.

### Java / JVM
```bash
cat pom.xml 2>/dev/null | head -60
cat build.gradle 2>/dev/null | head -40
```

### Swift / iOS / macOS
```bash
ls *.xcodeproj/ *.xcworkspace/ Package.swift 2>/dev/null
cat Package.swift 2>/dev/null
cat .swiftlint.yml 2>/dev/null
cat .swiftformat 2>/dev/null
```

### Containers & Infrastructure
```bash
cat Dockerfile 2>/dev/null
cat docker-compose.yml 2>/dev/null
cat fly.toml 2>/dev/null
cat render.yaml 2>/dev/null
ls k8s/ kubernetes/ helm/ 2>/dev/null
```

### CI/CD
```bash
ls .github/workflows/ 2>/dev/null
ls .circleci/ .travis.yml .gitlab-ci.yml 2>/dev/null
```

### Code quality tooling
```bash
ls .eslintrc* .prettierrc* .oxlintrc* .biome* .oxfmtrc* 2>/dev/null
cat .pre-commit-config.yaml 2>/dev/null
```

---

## Phase 3 — Entrypoint & Binary Detection

Find what actually runs:

```bash
# Node: look at bin field in package.json and main script
cat package.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('bin:', d.get('bin','')); print('main:', d.get('main','')); print('module:', d.get('module',''))" 2>/dev/null

# Look for CLI entry files
find . -name "index.ts" -o -name "index.js" -o -name "main.ts" -o -name "cli.ts" \
  -not -path '*/node_modules/*' | head -10

# Look for explicit entry in root
ls *.mjs *.js *.ts 2>/dev/null | head -10
```

For OpenClaw-style repos, also check:
```bash
cat openclaw.mjs 2>/dev/null | head -30
```

---

## Phase 4 — Architecture Pattern Detection

### Monorepo detection
```bash
# pnpm workspaces
cat pnpm-workspace.yaml 2>/dev/null
# npm workspaces
cat package.json | grep -A10 '"workspaces"' 2>/dev/null
# Check packages/ apps/ dir structure
ls packages/ apps/ 2>/dev/null
```

### Key source directories — read each present one
```bash
ls src/ 2>/dev/null
ls packages/ 2>/dev/null
ls apps/ 2>/dev/null
ls extensions/ 2>/dev/null
ls plugins/ 2>/dev/null
ls ui/ 2>/dev/null
ls lib/ 2>/dev/null
```

For each directory found, record what it contains at one level deep:
```bash
ls -la <dir>/
```

### Platform / companion app detection
Look for: `Swabble/`, `ios/`, `android/`, `macos/`, `electron/`, `apps/`

### Plugin SDK detection
If the project has a large `exports` map with named SDK subpaths, it's publishing a **plugin architecture**:
```bash
# Check for plugin/SDK exports in package.json
cat package.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
exports = d.get('exports', {})
sdk_paths = [k for k in exports if any(x in k for x in ['sdk', 'plugin', 'runtime', 'core'])]
print(f'SDK surface: {len(sdk_paths)} named exports')
# Cluster by prefix to find subsystems
from collections import Counter
prefixes = Counter()
for k in sdk_paths:
    parts = k.strip('./').split('/')
    if len(parts) > 1:
        prefixes[parts[1].split('-')[0]] += 1
print('Subsystem clusters:', dict(prefixes.most_common(15)))
" 2>/dev/null
```


```bash
ls test/ tests/ __tests__/ spec/ qa/ 2>/dev/null
cat vitest.config.ts 2>/dev/null
cat jest.config.* 2>/dev/null
```

---

## Phase 5 — Docs & Convention Files

Read these files to understand intent and conventions:

```bash
# Always read these if present
cat README.md 2>/dev/null
cat VISION.md 2>/dev/null
cat CONTRIBUTING.md 2>/dev/null
cat AGENTS.md 2>/dev/null        # AI agent instructions
cat CLAUDE.md 2>/dev/null        # Claude-specific instructions (often symlink to AGENTS.md)
cat CHANGELOG.md 2>/dev/null | head -60
cat docs.acp.md 2>/dev/null      # ACP protocol docs (OpenClaw-specific)
```

For OpenClaw specifically, also check:
```bash
ls docs/ 2>/dev/null
ls .pi/ 2>/dev/null
ls .agents/ 2>/dev/null
ls .agents/skills/ 2>/dev/null
ls skills/ 2>/dev/null
```

---

## Phase 6 — Produce the Repo Inventory Manifest

After gathering all information, write the manifest. This is the structured output of this skill.

### Manifest format

```markdown
# Repo Inventory Manifest: <repo-name>

**Scanned:** <date>
**Source:** <url or path>
**Primary language(s):** <list>
**Runtime target:** <Node/Python/Swift/Browser/etc>

---

## 1. What Is This?

<2-4 sentence plain-English description of what the project does, who uses it,
and what problem it solves. Use README + VISION.md if available.>

## 2. Tech Stack

| Layer | Technology |
|-------|-----------|
| Runtime | e.g. Node 24 |
| Package manager | pnpm / npm / uv / cargo |
| Language | TypeScript 5.x / Python 3.12 |
| Build | tsdown / tsc / Webpack / Vite |
| Test framework | Vitest / Jest / pytest |
| Linter/formatter | oxlint + prettier / eslint |
| Container | Docker (multi-stage) |
| Deployment | fly.io / render / k8s |
| CI | GitHub Actions |
| Other notable | list any |

## 3. Directory Map

<repo-root>/
├── src/                  # [ROLE: core source] <brief description>
│   ├── gateway/          # <brief description>
│   ├── channels/         # <brief description>
│   └── ...
├── packages/             # [ROLE: shared packages] <brief>
├── apps/                 # [ROLE: companion apps] <brief>
├── ui/                   # [ROLE: frontend UI] <brief>
├── extensions/           # [ROLE: extension system] <brief>
├── skills/               # [ROLE: bundled skills] <brief>
├── test/                 # [ROLE: tests] <brief>
├── scripts/              # [ROLE: build/dev scripts] <brief>
├── docs/                 # [ROLE: documentation] <brief>
└── ...

## 4. Entrypoints

| Entrypoint | File | Purpose |
|------------|------|---------|
| CLI binary | `openclaw.mjs` / `src/cli.ts` | <description> |
| Gateway daemon | `src/gateway/index.ts` | <description> |
| npm package | `dist/index.js` | <description> |
| Web UI | `ui/` | <description> |

## 5. Architectural Pattern

**Pattern detected:** <monorepo / single-package / microservices / plugin-based / etc>

<2-3 sentences describing the architecture. E.g.: "This is a TypeScript monorepo 
managed with pnpm workspaces. The core runtime lives in `src/` and is published 
as the `openclaw` npm package. Platform-specific companion apps (macOS, iOS) live 
in `apps/` and communicate with the gateway via WebSocket.">

## 6. Key Subsystems

List each major subsystem and its root directory:

- **Gateway** (`src/gateway/`) — <one-line description>
- **Channel adapters** (`src/channels/`) — <one-line description>
- **Agent runtime** (`src/agent/`) — <one-line description>
- **Plugin SDK** (`dist/plugin-sdk/`, exported as `openclaw/plugin-sdk/*`) — <number> named entry points for third-party plugin authors; covers channel, provider, approval, memory, reply, hook, config subsystems
- **UI** (`ui/`) — <one-line description>
- **Skills** (`skills/`) — <one-line description>
- **Extensions** (`extensions/`) — <one-line description>

## 7. External Dependencies (notable)

List the most architecturally significant packages only (not exhaustive):

| Package | Purpose |
|---------|---------|
| `@ai-sdk/...` | LLM provider abstraction |
| `baileys` | WhatsApp Web protocol |
| `grammy` | Telegram bot framework |
| etc | etc |

## 8. Deployment & Infrastructure

<Brief paragraph on how this is deployed, what containers are involved,
what services it connects to.>

## 9. Test Infrastructure

<Brief: test runner, location of tests, any notable test fixtures or QA tooling.>

## 10. Open Questions for Deeper Analysis

List 3-5 things that the cartographer scan couldn't answer but are worth
investigating in the archaeologist phase:

1. How does the channel adapter abstraction work? What's the interface contract?
2. How does the agent loop handle session isolation between concurrent users?
3. What is the role of `packages/` vs `src/` — are these published separately?
4. ...
```

---

## Output & Handoff

Save the manifest to the working directory:
```bash
mkdir -p /tmp/repo-analysis/<repo-name>/
# Write the manifest — used by downstream skills
```

Then present it to the user inline. Follow with:

> "The inventory is complete. You can now:
> - Ask me to **dig deeper** into how a specific subsystem works (uses `repo-archaeologist`)  
> - Ask me to **generate documentation** for the repo (uses `repo-documentarian`)
> - Ask me about any specific part of the codebase"

---

## Quality Checklist

Before presenting the manifest, verify:
- [ ] All top-level directories are accounted for and classified
- [ ] At least one entrypoint is identified
- [ ] Architecture pattern is named and explained
- [ ] Tech stack table is complete (runtime, package manager, language, build, test)
- [ ] Open questions are specific and actionable (not generic)
- [ ] Plain-English summary (section 1) can be understood without reading any code
