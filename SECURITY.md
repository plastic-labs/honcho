# Security Policy

## Reporting a vulnerability

**Do not file a public GitHub issue for security vulnerabilities.**

Please report security issues privately using one of:

1. **[GitHub Private Vulnerability Reporting](https://github.com/plastic-labs/honcho/security/advisories/new)** (preferred)
2. Email **<support@honcho.dev>** with subject line `[SECURITY] …`

Include as much of the following as you can:

- Description of the issue and its impact
- Steps to reproduce, or a proof of concept
- Affected component (API, deriver, auth/JWT, SDK, managed offering, etc.)
- Honcho version or image tag, and whether you are on managed or self-hosted

Honcho stores conversational data and peer representations. **Do not** attach production user content, API keys, JWTs, or other secrets to a report unless we explicitly ask for a redacted sample.

## What to expect

We will acknowledge valid reports as soon as we can and will keep you updated on remediation status. Please give us a reasonable window to investigate and fix before any public disclosure.

## Supported versions

Security fixes are applied to the latest release on `main` and, when practical, to the most recent tagged release line. Older versions may not receive backports.

## Non-security bugs

For ordinary bugs, memory/recall quality issues, and feature requests, use the [issue templates](https://github.com/plastic-labs/honcho/issues/new/choose).
