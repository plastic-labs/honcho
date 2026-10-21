#!/usr/bin/env bash
#
# Erasure-slice acceptance runner: two api replicas and two derivers on the mock
# provider, one message delete under a paused multi-observer batch, Redis down,
# then a swap to the previous image. Results land in results/<label>.json.
#
#   sandbox/slice.sh run --label X --image-under-test IMG [--previous-image IMG]
#                        [--delete-action api|sql] [--release-on same|other]
#                        [--results-dir DIR]
#   sandbox/slice.sh build IMG [GIT_REF]   build IMG from the working tree or a ref
#   sandbox/slice.sh down                  remove this project's containers and volumes

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

compose() {
  COMPOSE_PROJECT_NAME=slice_harness HONCHO_SANDBOX_IMAGE=unused SLICE_APP_IMAGE=unused \
    SLICE_PAUSE_DIR="$HERE/.slice/pause" \
    docker compose -p slice_harness \
    -f "$HERE/compose.yml" -f "$HERE/compose.mock.yml" -f "$HERE/compose.slice.yml" "$@"
}

case "${1:-}" in
  run)
    shift
    exec uv run --project "$REPO" python "$HERE/slice_runner.py" "$@"
    ;;
  build)
    image="${2:?usage: slice.sh build IMG [GIT_REF]}"
    ref="${3:-}"
    if [ -z "$ref" ]; then
      docker build -t "$image" "$REPO"
    else
      context="$(mktemp -d)"
      trap 'rm -rf "$context"' EXIT
      git -C "$REPO" archive "$ref" | tar -x -C "$context"
      docker build -t "$image" "$context"
    fi
    ;;
  down)
    compose down -v --remove-orphans
    ;;
  *)
    sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'
    [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || exit 1
    ;;
esac
