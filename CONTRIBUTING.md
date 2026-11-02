# Contributing to Honcho

Honcho is developed by a small team of maintainers, who make all changes to the codebase.
**Pull requests are limited to maintainers.** See
[this issue](https://github.com/plastic-labs/honcho/issues/1448) for why.

You can still help a lot. Clear bug reports, memory-quality reports with a reproduction,
integration requests, and design discussion all move the project forward.

To understand how Honcho works or to run it locally, see the
[architecture](https://honcho.dev/docs/v3/documentation/core-concepts/architecture) and
[local setup](https://honcho.dev/docs/v3/contributing/self-hosting) docs.

## Issues instead of pull requests

If you find a bug or want a change, please
[create an issue](https://github.com/plastic-labs/honcho/issues/new/choose) instead. A clear
issue with a minimal reproduction helps us more than a pull request. See
[Filing an issue](#filing-an-issue).

## Think your work is an exception?

If you have a real need that an issue doesn't cover, come to
[Discord](https://discord.gg/honcho) and make the case. Maintainers are most active there.

## AI usage policy

You may use AI to contribute, but it must never waste a maintainer's time or make their work
unpleasant.

To enforce this, and regardless of whether AI was actually used, a maintainer may close your
issue without notice and block your account.

## Filing an issue

- **Use the [issue templates](https://github.com/plastic-labs/honcho/issues/new/choose)** and
  fill in every required field. `gh issue create` and the API skip the form, so copy the
  fields from [`.github/ISSUE_TEMPLATE/`](./.github/ISSUE_TEMPLATE/).
- **File only what you hit in real use**, directly or through an integration. Not findings
  from reading the code, a synthetic test, or another repo's tracker.
- **One root cause per issue.** If the README or docs describe the behavior, it is a feature
  request, not a bug.
- **Keep it short:** in English, under 600 words of prose, with code linked by permalink.
  Redact API keys, JWTs, and production user content.
- **Questions go to [Discord](https://discord.gg/honcho).** Other integrations, such as
  OpenClaw and Claude Code, have their own repos, linked from the
  [integrations catalog](https://honcho.dev/docs/v3/guides/overview).

Only a maintainer applies `maintainer-approved`, which means we have agreed the change should
be made. If something is missing or wrong, one of these labels is applied with a comment
saying what to fix.

| Label         | Applied when                                                                                                                                                            | Closed after                                   |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `needs-repro` | No steps or version to reproduce, or the problem was found by reading the code, in a synthetic test, or carried over from another repo without being run against Honcho | 7 days without a reply                         |
| `needs-info`  | A required field is missing, the issue is not in English, or the prose runs over 600 words                                                                              | 7 days without a reply                         |
| `invalid`     | Illegible, spam, a security finding filed publicly, or belongs in another repo                                                                                          | 3 days                                         |
| `question`    | A support question rather than a defect. Ask in [Discord](https://discord.gg/honcho), or email <support@honcho.dev> about the managed service                           | 3 days                                         |
| `duplicate`   | Same root cause as an existing issue, linked in the comment                                                                                                             | 3 days, unless you explain why it is different |
| `wontfix`     | Documented behavior, or an unsupported build or platform: a forked or patched build, an unsupported operating system, third-party packaging                             | Closed by a maintainer                         |

If we got it wrong, say so on the issue and it will be reopened. An issue that breaks the
[AI usage policy](#ai-usage-policy) can be closed without any of these steps.

## If you're an agent

- **Check for write access before opening a pull request.** Run
  `gh api repos/plastic-labs/honcho --jq .permissions.push`. If it prints `false`, file an
  issue instead.
- **File only what the person you are working with hit** while using Honcho, following
  [Filing an issue](#filing-an-issue).
- **Do not report checks you did not run.**

## Security

Report vulnerabilities privately through
[GitHub Private Vulnerability Reporting](https://github.com/plastic-labs/honcho/security/advisories/new),
never in a public issue. See [SECURITY.md](./SECURITY.md). Honcho does not operate a bug
bounty.

## License

Contributions are licensed under the project's [AGPL-3.0 License](./LICENSE).
