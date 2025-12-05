• Added a Fly-powered self-hosted runner flow and hooked it to the unified test suite.

- .github/workflows/start-fly-runner.yml spins up or reuses a Fly machine (FLY_RUNNER_APP/FLY_RUNNER_IMAGE overridable via repo vars), passes GH_TOKEN, RUN_ID, RUNNER_LABELS, waits until a runner with the current
    run_id label is online, and surfaces the machine ID/labels/name as outputs.
- .github/workflows/unified-tests.yml triggers on push to main (and manual dispatch), calls the runner workflow, then runs the unified tests on that self-hosted runner with uv sync/uv run python -m
    tests.unified.run after a Docker availability check. It sets Anthropic-centric LLM env defaults and cleans up the Fly machine and GitHub runner afterward.

  Secrets/vars to configure:

- FLY_API_TOKEN for Fly access.
- GH_TOKEN (PAT with repo-level runner admin rights) for runner registration/list/delete; it’s also sent into the Fly machine.
- LLM_ANTHROPIC_API_KEY for LLM judge assertions. Adjust the provider/model envs in unified-tests.yml if you prefer different models or keys.
- Optional repo vars to override defaults: FLY_RUNNER_APP, FLY_RUNNER_REGION, FLY_RUNNER_IMAGE, FLY_RUNNER_VM_SIZE, FLY_RUNNER_VM_MEMORY, FLY_RUNNER_NAME_PREFIX.

  Assumptions: the Fly runner image honors RUNNER_LABELS/RUN_ID when registering, and Docker is available on that machine (the workflow fails fast with docker info otherwise). I couldn’t run the workflow locally
  here. Next steps: (1) set the secrets/vars, (2) kick off the workflow_dispatch to confirm the runner comes online and the unified tests complete, (3) tweak model/env selections if you want a different provider.
