# OpenClaw Repo Cartographer Reference

This reference covers what the cartographer should know specifically about the
`openclaw/openclaw` repository structure so it can produce a high-fidelity manifest
without having to re-discover known facts.

## Known Directory Roles

| Directory | Role |
|-----------|------|
| `src/` | Core TypeScript source — gateway, channels, agent runtime |
| `packages/` | Published/shared npm packages (lower-level libs) |
| `apps/` | Platform companion apps — macOS menu bar, iOS node, Android node |
| `ui/` | Control UI (web admin + WebChat) served by the gateway |
| `extensions/` | Extension/plugin system |
| `skills/` | Bundled skills (AgentSkills-compatible `SKILL.md` folders) |
| `.agents/skills/` | Project-level agent skills |
| `test/` | Integration and unit tests |
| `test-fixtures/` | Test fixture files |
| `qa/` | QA tooling and test scripts |
| `scripts/` | Build, release, and dev helper scripts |
| `docs/` | Documentation source (Mintlify at docs.openclaw.ai) |
| `assets/` | Static assets (logos, images) |
| `Swabble/` | Likely a sub-project or companion tool |
| `vendor/a2ui/` | Vendored A2UI library (agent-to-UI canvas system) |
| `.pi/` | Pi agent core integration files |
| `.github/` | GitHub Actions CI/CD workflows |
| `patches/` | Dependency patches (pnpm patches) |
| `git-hooks/` | Git hook scripts |

## Critical: package.json Is Massive (1,476 lines)

The root `package.json` is not a normal manifest — it exports **100+ named plugin-sdk entry
points** under `"exports"`. This is the public API surface for plugin authors:

```json
"exports": {
  ".": "./dist/index.js",
  "./plugin-sdk": { ... },
  "./plugin-sdk/channel-runtime": { ... },
  "./plugin-sdk/agent-runtime": { ... },
  // 100+ more...
}
```

**What this means:** OpenClaw has a fully published **Plugin SDK** that third-party plugin authors
use to build channel adapters, providers, and extensions. This is NOT just an internal bundling
detail — it's a versioned, public API. The plugin-sdk exports map to compiled `dist/plugin-sdk/*.js`
files, each backed by TypeScript source in `src/`.

When reading `package.json` for the cartographer, **do not try to read the full file** (it will
exceed context). Instead extract only:
```bash
# Get just the key top-level fields
cat package.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('name:', d.get('name'))
print('version:', d.get('version'))
print('description:', d.get('description'))
print('bin:', list(d.get('bin', {}).keys()))
print('main:', d.get('main'))
print('type:', d.get('type'))
print('plugin-sdk exports count:', sum(1 for k in d.get('exports', {}) if 'plugin-sdk' in k))
print('scripts:', list(d.get('scripts', {}).keys())[:20])
"
```

## Key Root Files

| File | Purpose |
|------|---------|
| `openclaw.mjs` | Main CLI entrypoint (ESM) |
| `package.json` | Root package manifest — 1,476 lines, 100+ plugin-sdk exports, pnpm workspace root |
| `pnpm-workspace.yaml` | Workspace member definitions |
| `tsconfig.json` | Root TypeScript config |
| `tsconfig.core.json` | Core source TS config |
| `tsconfig.extensions.json` | Extensions TS config |
| `tsdown.config.ts` | Build config (tsdown = rollup-based TS bundler) |
| `vitest.config.ts` | Test runner config |
| `knip.config.ts` | Dead code detection |
| `Dockerfile` | Main container image |
| `Dockerfile.sandbox` | Sandbox container (for isolated agent sessions) |
| `docker-compose.yml` | Local dev compose |
| `fly.toml` | Fly.io deployment config |
| `render.yaml` | Render.com deployment config |
| `AGENTS.md` | AI agent operating instructions (injected into agent context) |
| `CLAUDE.md` | Symlink to AGENTS.md |
| `VISION.md` | Project vision document |
| `CONTRIBUTING.md` | Contribution guidelines |
| `CHANGELOG.md` | Release changelog |
| `INCIDENT_RESPONSE.md` | Security incident response guide |
| `SECURITY.md` | Security policy |
| `docs.acp.md` | ACP (Agent Communication Protocol) docs |
| `.codex` | Codex integration config |
| `.env.example` | Environment variable template |
| `.oxlintrc.json` | oxlint linter config |
| `.oxfmtrc.jsonc` | oxfmt formatter config |
| `.swiftlint.yml` | Swift linter config (macOS/iOS app) |
| `.swiftformat` | Swift formatter config |

## Architecture Summary (for manifest section 5)

OpenClaw is a **TypeScript monorepo** (pnpm workspaces) with:
- A **long-lived Gateway daemon** as the control plane
- **Channel adapters** that connect messaging platforms (WhatsApp/Telegram/Slack/Discord/etc)
- An **embedded agent runtime** (Pi agent core) that runs LLM sessions
- **WebSocket-based** client/node protocol for all control-plane communication
- **Platform companion apps** (macOS, iOS, Android) that connect as "nodes"
- A **skill system** (AgentSkills-compatible) for extending agent capabilities
- **Docker sandboxing** for multi-user isolation

## Multi-Platform Scope

This is not just a server project. It spans:
- **Server** (Gateway daemon, Node.js)
- **Web** (Control UI + WebChat, served by Gateway)
- **macOS** (Native menu bar app, OpenClaw.app)
- **iOS** (Node app for voice/canvas)
- **Android** (Node app for voice/canvas/camera)
- **CLI** (openclaw binary)
