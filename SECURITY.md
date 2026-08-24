# Security Policy

## Supported Versions

The `main` branch of this repo maps to the latest canary version of Honcho. To see which versions are supported please refer to the git tags in the repo or the [compatibility guide](https://honcho.dev/docs/changelog/compatibility-guide).

## Reporting a Vulnerability

Do not open a public issue for a suspected vulnerability. Report it privately through one of:

1. **[GitHub Private Vulnerability Reporting](https://github.com/plastic-labs/honcho/security/advisories/new)** — preferred; it keeps the report, our replies, and any fix coordinated in one place.
2. Email [support@honcho.dev](mailto:support@honcho.dev) with `[SECURITY]` in the subject.

Include as much of the following as you have:

- **Version** — a git commit SHA, or the release tag you are running
- **Deployment** — self-hosted or the managed service at `api.honcho.dev`
- **Affected component** — API, deriver, dialectic, auth/JWT, an SDK, or the managed offering
- **Reproduction** — the exact steps, requests, or script that trigger it
- **Proof of concept** — the smallest thing that demonstrates the issue actually works
- **Impact** — what an attacker gains, and what they need to already have to get it
- **How you found it** — manual review, fuzzing, a scanner, or model-assisted analysis

Reports with a working proof of concept get looked at first. A report that only describes a
theoretical problem is much slower for us to act on, because we have to build the repro
ourselves before we can confirm anything.

Honcho stores conversational data and peer representations. **Do not attach production user
content, API keys, or JWTs** to a report — if we need a sample, we will ask for a redacted
one.

## Testing

Test against an instance you operate. Do not run security testing against `api.honcho.dev`
or against any Honcho deployment that is not yours — self-hosting is a first-class path and
takes a few minutes to set up, see [Self-hosting](./README.md#self-hosting).

## What to Expect

We will acknowledge your report and tell you whether we consider it in scope. If it is, we
will let you know when a fix ships.

We do not commit to a response SLA, we do not coordinate CVE assignment on request, and we
do not operate a disclosure timeline you can hold us to. This is a small team.

## Out of Scope

The following are not treated as vulnerabilities. Reports consisting only of these will be
closed without a detailed response:

- Automated scanner output with no working proof of concept
- Model-generated findings that have not been verified by a human against a running instance
- Missing security headers or TLS configuration with no demonstrated exploit
- Rate limiting, or resource exhaustion with no demonstrated impact beyond your own instance
- Vulnerabilities in dependencies with no demonstrated exploit path through Honcho
- Configuration weaknesses that require an already-compromised host, or that come from
  deliberately insecure settings (for example running with `AUTH_USE_AUTH=false`, which is
  the documented local-development default and is not intended for a public deployment)
- Social engineering, phishing, and physical access

For ordinary bugs, memory or recall quality problems, and feature requests, use the
[issue templates](https://github.com/plastic-labs/honcho/issues/new/choose) instead.

## No Bug Bounty

The Honcho project does not offer any rewards for reported bugs or
vulnerabilities. We do not aid security researchers to get such rewards for
Honcho problems from other sources.

A bug bounty gives people too strong incentives to find and make up "problems"
in bad faith that cause overload and abuse.

We still appreciate and value valid vulnerability reports.
