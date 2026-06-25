# Secret & environment-PII scanning (local pre-commit gate)

This repo runs [gitleaks](https://github.com/gitleaks/gitleaks) as a **local
pre-commit hook** to stop secrets *and* environment-specific identifiers from
being committed. It was added after private LAN IPs (`192.168.x.x`), a Windows
username (in `C:\Users\<name>\…` paths) and a host name leaked into committed
`docs/superpowers/plans` and `specs`.

## What it catches

- **Credentials** — all built-in gitleaks rules (`useDefault = true`): cloud
  keys, generic API tokens, private keys, JWTs, …
- **Environment PII** (custom rules in [`.gitleaks.toml`](../.gitleaks.toml)) —
  the class that stock secret scanners miss:
  - `private-ipv4` — RFC1918 addresses (`10.x`, `172.16–31.x`, `192.168.x`).
  - `windows-user-home-path` — `C:\Users\<name>`, `c:/Users/<name>`,
    `/c/Users/<name>` (leaks the local account name).
  - `amivoice-appkey-inline` — an AmiVoice `appkey = "…"` assigned in-tree.

Redaction placeholders (`<USER>`, `<NAS_HOST>`) and standard Windows accounts
(`Public`, `Default`, …) are allowlisted, so the sanitized docs/tests pass.

## Setup (once per clone)

```sh
# 1. pre-commit (a Python tool)
uv tool install pre-commit          # or: pipx install pre-commit

# 2. gitleaks binary on PATH (>= 8.18). No Go/Docker build needed.
winget install gitleaks             # or: scoop install gitleaks

# 3. wire the git hook
pre-commit install
```

After this, every `git commit` scans the **staged** changes and aborts the
commit on a finding.

## Caveats

- **Local only.** `git commit --no-verify` bypasses it and there is no
  server-side enforcement. If you want a hard gate, add a `gitleaks` GitHub
  Actions job on `pull_request` and require it in branch protection.
- **History is not rewritten.** The sanitized values still exist in older
  commits; this gate only prevents *new* leaks.

## Handling false positives

- One line: append `# gitleaks:allow` to it.
- A recurring pattern: add a regex/path to `[allowlist]` in
  [`.gitleaks.toml`](../.gitleaks.toml).

## Manual scans

```sh
# Staged changes only (what the hook runs):
gitleaks git --pre-commit --staged --config .gitleaks.toml --redact --verbose

# Whole working tree (audit everything):
gitleaks dir . --config .gitleaks.toml --redact --verbose

# Full commit history:
gitleaks git . --config .gitleaks.toml --redact --verbose
```
