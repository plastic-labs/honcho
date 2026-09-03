#!/usr/bin/env bash
#
# Honcho sandbox — an ephemeral, seeded stack that resets to a known state in
# seconds, so harness testing stops depending on machine state.
#
#   sandbox.sh up [--provider mock|real] [--build]
#   sandbox.sh seed [--provider ...]
#   sandbox.sh reset [--provider ...]
#   sandbox.sh status [--provider ...]
#   sandbox.sh down [--provider ...]
#
# See README.md.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

DB=honcho_sandbox
BUILT_IMAGE=honcho-sandbox/honcho:built

PROVIDER="${HONCHO_SANDBOX_PROVIDER:-mock}"
BUILD=0

die() { echo "error: $*" >&2; exit 1; }
say() { echo "==> $*"; }

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

COMMAND="${1:-}"
[ -n "$COMMAND" ] || die "usage: sandbox.sh {up|seed|reset|status|down} [--provider mock|real] [--build]"
shift

while [ $# -gt 0 ]; do
  case "$1" in
    --provider) PROVIDER="${2:-}"; shift 2 ;;
    --provider=*) PROVIDER="${1#*=}"; shift ;;
    --build) BUILD=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$PROVIDER" in
  mock|real) ;;
  *) die "unknown provider '$PROVIDER' (expected 'mock' or 'real')" ;;
esac

# Each mode keeps its own template, so both can coexist and switching modes is
# free rather than forcing a reseed.
TEMPLATE="${DB}_seeded_${PROVIDER}"

# --------------------------------------------------------------------------
# Compose invocation
#
# The base file is provider-agnostic and is not a working stack on its own;
# exactly one provider overlay is always composed on top.
# --------------------------------------------------------------------------

# The pinned default...
# shellcheck disable=SC1091
set -a; . "$HERE/image.env"; set +a

# ...then whatever the running stack was actually started with. Without this,
# `up --build` followed by a plain `reset` would flip the image back to the pinned
# digest, and Compose would helpfully recreate every container from it mid-reset.
STATE="$HERE/.state.env"
if [ "$BUILD" = 1 ]; then
  HONCHO_SANDBOX_IMAGE="$BUILT_IMAGE"
elif [ -f "$STATE" ]; then
  # shellcheck disable=SC1090
  set -a; . "$STATE"; set +a
fi
export HONCHO_SANDBOX_IMAGE

# The provider the running stack was actually created with, read on its own because
# the source above is skipped under --build. Empty means nothing has been started
# through `up` since the last `down`.
RUNNING_PROVIDER=""
if [ -f "$STATE" ]; then
  RUNNING_PROVIDER="$(sed -n 's/^SANDBOX_RUNNING_PROVIDER=//p' "$STATE")"
fi

write_state() {
  cat > "$STATE" <<EOF
HONCHO_SANDBOX_IMAGE=$HONCHO_SANDBOX_IMAGE
SANDBOX_RUNNING_PROVIDER=$PROVIDER
EOF
  RUNNING_PROVIDER="$PROVIDER"
}

compose() {
  docker compose \
    -f "$HERE/compose.yml" \
    -f "$HERE/compose.$PROVIDER.yml" \
    --project-directory "$HERE" \
    "$@"
}

psql_db() {
  # Maintenance connections go to `postgres`, never to $DB — you cannot drop a
  # database you are connected to.
  compose exec -T database psql -v ON_ERROR_STOP=1 -U postgres -d postgres "$@"
}

db_exists() {
  [ "$(psql_db -tAc "SELECT 1 FROM pg_database WHERE datname='$1'" 2>/dev/null)" = "1" ]
}

# --------------------------------------------------------------------------
# Preconditions
# --------------------------------------------------------------------------

preflight() {
  docker info >/dev/null 2>&1 || die "the Docker daemon is not running"

  if [ "$PROVIDER" = real ]; then
    # Fail before starting anything. A real-mode stack with no key boots fine and
    # then derives nothing, which reads as "the deriver found nothing".
    [ -f "$HERE/real.env" ] || die \
      "real mode needs $HERE/real.env. Copy real.env.example and add a provider key."
    grep -Eq '^\s*LLM_[A-Z]+_API_KEY=.+' "$HERE/real.env" || die \
      "no LLM_*_API_KEY with a value in real.env."
    if grep -Eq '^\s*LLM_[A-Z]+_API_KEY=(sk-replace-me|your-api-key-here)\s*$' "$HERE/real.env"; then
      die "real.env still has the placeholder key from real.env.example."
    fi
  fi
  # Must not end on a failed test. Under `set -e`, a non-zero return from the last
  # command in this function aborts the script with no output at all — which is
  # exactly how the first real-mode `up` "failed": silently, before doing anything.
  return 0
}

# Mock mode runs src/mock_provider out of the same image. A digest that predates the
# module resolves and pulls fine, then crash-loops one service on a missing module —
# so check for it up front and name the two ways out.
check_mock_provider_present() {
  [ "$PROVIDER" = mock ] || return 0
  if ! docker run --rm --entrypoint test "$HONCHO_SANDBOX_IMAGE" -d /app/src/mock_provider; then
    die "$(cat <<MSG
the image $HONCHO_SANDBOX_IMAGE does not contain src/mock_provider.

It predates the mock provider. Either bump the digest in sandbox/image.env to an
image that has it, or build from the working tree:

    sandbox/sandbox.sh up --build
MSG
)"
  fi
}

# Containers carry the provider wiring they were created with, and neither seed nor
# reset recreates them: seed uses `start` plus `--no-recreate` to avoid paying a
# container rebuild, and reset touches no container at all. So a --provider that
# disagrees with the running stack does not change what the stack talks to. It only
# changes what gets recorded — seed would derive through the running provider and
# then stamp the requested one into the template, defeating the fingerprint guard,
# which compares the flag rather than reality.
#
# The real-stack-seeding-a-mock-template case is the damaging one: it spends money,
# produces non-deterministic conclusions, and labels them `mock`, so every later
# reset restores that as the deterministic baseline. Refuse rather than mislead.
require_running_provider() {
  [ -n "$RUNNING_PROVIDER" ] || return 0
  [ "$RUNNING_PROVIDER" != "$PROVIDER" ] || return 0
  die "$(cat <<MSG
the stack is running the $RUNNING_PROVIDER provider, but this is a $PROVIDER-mode command.

seed and reset do not recreate containers, so this would act through the
$RUNNING_PROVIDER provider while recording $PROVIDER. Switch the stack first:

    sandbox/sandbox.sh up --provider $PROVIDER
MSG
)"
}

seed_py() {
  SANDBOX_BASE_URL="http://127.0.0.1:${SANDBOX_API_PORT:-18000}" \
    uv run --no-project --with-editable "$REPO/sdks/python" \
      python "$HERE/seed.py" "$1"
}

# Conclusion levels and premise links are not reachable from the public API, so this
# part of the fixture is written by a script running inside the api container, which
# already is Honcho's venv with the api's settings and embedding client. The script
# arrives on stdin and the fixture in the environment, so nothing has to be mounted
# and nothing is left behind in the container.
#
# It decides for itself whether the fixture declares any conclusions, rather than
# being gated by a grep here that a comment mentioning a level name would fool.
inject_conclusions() {
  compose exec -T \
    -e SANDBOX_FIXTURE_JSON="$(cat "$HERE/fixture.json")" \
    api /app/.venv/bin/python - < "$HERE/inject_conclusions.py"
}

# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

cmd_up() {
  preflight

  compose pull --quiet database redis
  if [ "$BUILD" = 1 ]; then
    say "building $BUILT_IMAGE from the working tree"
    docker build -t "$BUILT_IMAGE" "$REPO"
  elif docker image inspect "$HONCHO_SANDBOX_IMAGE" >/dev/null 2>&1; then
    # Already here — either pulled on an earlier run, or built locally by --build
    # and recorded in .state.env, in which case pulling it would fail outright
    # because the tag exists in no registry.
    say "using local image $HONCHO_SANDBOX_IMAGE"
  else
    say "pulling $HONCHO_SANDBOX_IMAGE"
    docker pull --quiet "$HONCHO_SANDBOX_IMAGE" >/dev/null
  fi

  check_mock_provider_present

  say "starting the $PROVIDER stack"
  # --remove-orphans because both modes share one Compose project: switching from
  # mock to real leaves mock-provider running and otherwise unreferenced, where it
  # would sit idle and make `status` misleading about what the stack is using.
  compose up -d --wait --remove-orphans
  # Recorded before the seed/reset below, so their provider guard sees the stack
  # that was just created rather than the one it replaced.
  write_state

  if db_exists "$TEMPLATE"; then
    say "template $TEMPLATE already present — resetting to it"
    cmd_reset
  else
    cmd_seed
  fi

  echo
  say "sandbox ready at http://127.0.0.1:${SANDBOX_API_PORT:-18000} (provider: $PROVIDER)"
}

cmd_seed() {
  preflight
  require_running_provider
  # Seeding an already-seeded database appends to it — the fixture would land a
  # second time and the "expected 6 messages, found 12" check in seed.py would
  # (correctly) fail. Start from an empty, freshly migrated database every time so
  # `seed` means the same thing regardless of what was there before.
  say "clearing the database before seeding"
  compose stop api deriver >/dev/null
  psql_db -c "DROP DATABASE IF EXISTS $DB WITH (FORCE)" >/dev/null
  psql_db -c "CREATE DATABASE $DB" >/dev/null
  compose exec -T database psql -q -v ON_ERROR_STOP=1 -U postgres -d "$DB" \
    -c "CREATE EXTENSION IF NOT EXISTS vector" >/dev/null
  # Redis still holds entries keyed to the database that was just dropped; without
  # this the api serves cached peers that no longer exist and seeding conflicts.
  compose exec -T redis redis-cli FLUSHALL >/dev/null
  # The api entrypoint runs scripts/provision_db.py, so starting it is what applies
  # the Alembic migrations to the new database.
  compose start api deriver >/dev/null
  compose up -d --wait --no-recreate >/dev/null

  say "seeding ($PROVIDER)"
  seed_py seed
  inject_conclusions
  seed_py verify

  say "snapshotting to $TEMPLATE"
  stamp_fingerprint
  # api and deriver hold pooled connections; CREATE DATABASE ... TEMPLATE needs
  # the source to have none.
  compose stop api deriver >/dev/null
  disconnect_db "$DB"
  psql_db -c "DROP DATABASE IF EXISTS $TEMPLATE" >/dev/null
  psql_db -c "CREATE DATABASE $TEMPLATE TEMPLATE $DB" >/dev/null
  compose start api deriver >/dev/null
  compose up -d --wait --no-recreate >/dev/null
  say "snapshot ready — reset restores this state without re-deriving"
}

cmd_reset() {
  preflight
  require_running_provider
  db_exists "$TEMPLATE" || die \
    "no template for $PROVIDER mode. Run: sandbox/sandbox.sh seed --provider $PROVIDER"

  check_fingerprint

  # Nothing is stopped or restarted. DROP ... WITH (FORCE) evicts the api and
  # deriver connection pools itself, and both reconnect on their next use
  # (db.POOL_PRE_PING is on) — the api on its next request, the deriver on its
  # next poll a quarter-second later.
  #
  # Both log one OperationalError as their in-flight connection dies. That noise
  # is the cost of not paying ~9s of container stop/start on every reset, which
  # would defeat the point. Restarting them instead is the fallback if this ever
  # stops being true.
  #
  # CREATE DATABASE ... TEMPLATE additionally needs the *source* to be
  # connectionless, and check_fingerprint just read from it.
  disconnect_db "$TEMPLATE"
  psql_db -c "DROP DATABASE $DB WITH (FORCE)" >/dev/null
  psql_db -c "CREATE DATABASE $DB TEMPLATE $TEMPLATE" >/dev/null
  compose exec -T redis redis-cli FLUSHALL >/dev/null
  say "reset to $TEMPLATE"
}

cmd_status() {
  compose ps
  echo
  if [ -n "$RUNNING_PROVIDER" ] && [ "$RUNNING_PROVIDER" != "$PROVIDER" ]; then
    echo "provider:  $RUNNING_PROVIDER (running) — this command asked for $PROVIDER"
  else
    echo "provider:  $PROVIDER"
  fi
  echo "image:     $HONCHO_SANDBOX_IMAGE"
  echo "api:       http://127.0.0.1:${SANDBOX_API_PORT:-18000}"
  for mode in mock real; do
    if db_exists "${DB}_seeded_${mode}"; then
      echo "template:  ${DB}_seeded_${mode}  present"
    else
      echo "template:  ${DB}_seeded_${mode}  absent"
    fi
  done
}

cmd_down() {
  say "tearing down the $PROVIDER stack and its volumes"
  compose down -v
  rm -f "$STATE"
}

# --------------------------------------------------------------------------
# Snapshot staleness
#
# A template that predates a migration, or was built against a different
# provider, restores silently and wrongly. Both are recorded inside the template
# itself so reset can refuse rather than mislead.
# --------------------------------------------------------------------------

fingerprint() {
  local alembic
  alembic="$(compose exec -T database psql -tAX -U postgres -d "$DB" \
    -c "SELECT version_num FROM alembic_version" 2>/dev/null | tr -d '[:space:]')"
  local files
  files="$(cat "$HERE/fixture.json" "$HERE/seed.py" "$HERE/inject_conclusions.py" \
    | shasum -a 256 | cut -c1-16)"
  echo "alembic=$alembic fixture=$files provider=$PROVIDER"
}

stamp_fingerprint() {
  compose exec -T database psql -v ON_ERROR_STOP=1 -U postgres -d "$DB" >/dev/null <<SQL
CREATE TABLE IF NOT EXISTS sandbox_fingerprint (value text PRIMARY KEY);
TRUNCATE sandbox_fingerprint;
INSERT INTO sandbox_fingerprint VALUES ('$(fingerprint)');
SQL
}

check_fingerprint() {
  local want have
  want="$(fingerprint)"
  have="$(compose exec -T database psql -tAX -U postgres -d "$TEMPLATE" \
    -c "SELECT value FROM sandbox_fingerprint" 2>/dev/null | tr -d '\r')"

  # The live database carries whatever revision the api container migrated it to
  # on boot; the template carries the revision it was seeded at. A difference is
  # exactly the migration drift this guard exists to catch, so it is compared,
  # not normalised away.
  [ "$want" = "$have" ] || die "$(cat <<MSG
$TEMPLATE is stale and would restore the wrong state.

  template: ${have:-<no fingerprint>}
  current:  $want

Reseed:  sandbox/sandbox.sh seed --provider $PROVIDER
MSG
)"
}

disconnect_db() {
  # Defaults to the live database. CREATE DATABASE ... TEMPLATE also requires the
  # *source* to have no connections, and reading the fingerprint just opened one.
  local target="${1:-$DB}"
  psql_db -c \
    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='$target' AND pid <> pg_backend_pid()" \
    >/dev/null
}

case "$COMMAND" in
  up) cmd_up ;;
  seed) cmd_seed ;;
  reset) cmd_reset ;;
  status) cmd_status ;;
  down) cmd_down ;;
  *) die "unknown command '$COMMAND' (expected up, seed, reset, status, or down)" ;;
esac
