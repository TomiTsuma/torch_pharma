# OpenClaw Documentation Reference

Pre-compiled knowledge for generating accurate OpenClaw documentation.

## Official Docs Links (always include these)

- Getting started: https://docs.openclaw.ai/start/getting-started
- Architecture: https://docs.openclaw.ai/concepts/architecture
- Agent Runtime: https://docs.openclaw.ai/concepts/agent
- Agent Loop: https://docs.openclaw.ai/concepts/agent-loop
- Configuration: https://docs.openclaw.ai/gateway/configuration
- Configuration Reference: https://docs.openclaw.ai/gateway/configuration-reference
- Security: https://docs.openclaw.ai/gateway/security
- Skills: https://docs.openclaw.ai/tools/skills
- Creating Skills: https://docs.openclaw.ai/tools/creating-skills
- Session model: https://docs.openclaw.ai/concepts/session
- Channels: https://docs.openclaw.ai/channels
- Discord: https://discord.gg/clawd
- ClawHub (skills registry): https://clawhub.ai
- DeepWiki: https://deepwiki.com/openclaw/openclaw

## Quick Start Snippet (verbatim from README)

```bash
npm install -g openclaw@latest
# or: pnpm add -g openclaw@latest
openclaw onboard --install-daemon
```

Runtime requirement: Node 24 (recommended) or Node 22.16+.

## Development Quick Start (verbatim from README)

```bash
git clone https://github.com/openclaw/openclaw.git
cd openclaw
pnpm install
pnpm openclaw setup          # first run only
pnpm ui:build                # optional: prebuild Control UI
pnpm gateway:watch           # dev loop with auto-reload
```

## Supported Channels (complete list)

WhatsApp, Telegram, Slack, Discord, Google Chat, Signal, iMessage, BlueBubbles,
IRC, Microsoft Teams, Matrix, Feishu, LINE, Mattermost, Nextcloud Talk, Nostr,
Synology Chat, Tlon, Twitch, Zalo, Zalo Personal, WeChat, QQ, WebChat.

## Chat Commands (operator reference)

`/status`, `/new`, `/reset`, `/compact`, `/think <level>`, `/verbose on|off`,
`/trace on|off`, `/usage off|tokens|full`, `/restart`, `/activation mention|always`

## Session Tools

`sessions_list`, `sessions_history`, `sessions_send`

## Security Defaults

- Default DM policy: `pairing` (unknown senders get a pairing code)
- Approve with: `openclaw pairing approve <channel> <code>`
- Run `openclaw doctor` to surface risky DM policies
- Group/channel safety: `agents.defaults.sandbox.mode: "non-main"`

## Development Channels

- `stable`: tagged releases, npm dist-tag `latest`
- `beta`: prerelease tags, npm dist-tag `beta`
- `dev`: main branch head, npm dist-tag `dev`
- Switch: `openclaw update --channel stable|beta|dev`

## Key Config Paths

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/.openclaw/workspace",
      "model": "anthropic/claude-sonnet-4-6",
      "sandbox": { "mode": "non-main" },
      "heartbeat": { "every": "30m", "target": "last" }
    }
  },
  "channels": {
    "whatsapp": { "allowFrom": ["+15555550123"] },
    "telegram": { "enabled": true, "botToken": "...", "dmPolicy": "pairing" }
  },
  "session": {
    "dmScope": "per-channel-peer",
    "reset": { "mode": "daily", "atHour": 4 }
  }
}
```

## Architecture Description (for docs)

OpenClaw is a personal AI assistant platform built on Node.js/TypeScript. The core concept
is a single long-lived **Gateway daemon** that owns all messaging surfaces. Channels
(WhatsApp via Baileys, Telegram via grammY, Slack, Discord, Signal, and 20+ others) connect
to the Gateway, which routes messages to an embedded **agent runtime** (Pi agent core). The
agent uses configurable LLM providers, runs tools, and delivers replies back through the
originating channel. Control-plane clients (macOS menu bar app, CLI, web admin) and device
nodes (iOS, Android) connect to the Gateway over a typed **WebSocket protocol**.

## Sponsors (for README)

OpenAI, GitHub, NVIDIA, Vercel, Blacksmith, Convex

## License

MIT

## Creator

Built for Molty (a space lobster AI assistant 🦞) by Peter Steinberger and the community.
- openclaw.ai | soul.md | steipete.me | @openclaw
