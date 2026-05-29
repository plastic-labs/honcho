#!/bin/bash
set -euo pipefail

# Build the honcho image via in-cluster BuildKit and push to the local registry.
#
# Runs identically from a laptop or from the honcho runner scale set in the
# `arc-runners` namespace. Source is always the checkout containing this
# script — no separate git clone step.
#
# Usage: ./rebuild.sh [--tag TAG] [--force] [--restart]
#
#   --tag TAG     Docker tag to publish (default: latest)
#   --force       Rebuild even if the image's SHA label already matches HEAD
#   --restart     Rollout-restart the honcho-api and honcho-deriver Deployments
#                 after a successful build. Convenience for laptop runs; CI uses
#                 a separate step.

cd "$(dirname "$0")"

REGISTRY="192.168.1.216:5000"
IMAGE="honcho"
TAG="latest"
FORCE="false"
RESTART="false"
ARCHITECTURES=("amd64" "arm64")
NAMESPACE="buildkit"
TIMESTAMP=$(date +%s)
BUILDKIT_IMAGE="moby/buildkit:v0.29.0"

WALL_START=$(date +%s)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) TAG="$2"; shift 2 ;;
        --force) FORCE="true"; shift ;;
        --restart) RESTART="true"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

missing=()
for tool in kubectl crane git jq tar gzip; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if (( ${#missing[@]} > 0 )); then
    echo "ERROR: missing required tool(s): ${missing[*]}" >&2
    echo "  On macOS: brew install ${missing[*]}" >&2
    exit 1
fi

CRANE_DOCKER_CONFIG="$(mktemp -d)"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${CRANE_DOCKER_CONFIG}" "${BUILD_DIR}"' EXIT
export DOCKER_CONFIG="${CRANE_DOCKER_CONFIG}"

if ! kubectl get ns "${NAMESPACE}" >/dev/null 2>&1; then
    echo "ERROR: cannot reach namespace '${NAMESPACE}' with current kubectl context." >&2
    echo "  Current context: $(kubectl config current-context 2>/dev/null || echo '<none>')" >&2
    exit 1
fi

SOURCE_SHA="$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "==> Building ${IMAGE} (${SOURCE_SHA}) on ${ARCHITECTURES[*]}"

# SHA short-circuit: if the existing image was built from this exact HEAD,
# skip unless --force.
if [[ "${FORCE}" != "true" && "${SOURCE_SHA}" != "unknown" ]]; then
    existing_sha=$(crane config --insecure "${REGISTRY}/${IMAGE}:${TAG}-${ARCHITECTURES[0]}" 2>/dev/null \
        | jq -r '.config.Labels["org.opencontainers.image.revision"] // ""' 2>/dev/null \
        || true)
    if [[ "${existing_sha}" == "${SOURCE_SHA}" ]]; then
        echo "==> ${REGISTRY}/${IMAGE}:${TAG} already built at ${SOURCE_SHA}; nothing to do (--force overrides)"
        exit 0
    fi
    if [[ -n "${existing_sha}" ]]; then
        echo "    Existing image built from ${existing_sha}; rebuilding to ${SOURCE_SHA}"
    fi
fi

# Honcho's repo is much bigger than the Dockerfile context needs. The
# Dockerfile only references: src/, migrations/, scripts/, docker/,
# alembic.ini, config.toml*, plus uv.lock and pyproject.toml. Ship only
# those to keep the ConfigMap under its 1MiB limit.
echo "==> Packaging build context (filtered to Dockerfile-relevant paths)"
CONTEXT_INCLUDES=(
    Dockerfile
    uv.lock
    pyproject.toml
    alembic.ini
    src
    migrations
    scripts
    docker
)
for glob in config.toml config.toml.example; do
    [[ -e "${SCRIPT_DIR}/${glob}" ]] && CONTEXT_INCLUDES+=("${glob}")
done

# COPYFILE_DISABLE=1 keeps macOS `tar` from silently shipping AppleDouble
# `._<filename>` metadata alongside real files. Inside the image, Alembic
# scans `migrations/versions/*.py` and tries to load those companions as
# Python — which errors with "source code string cannot contain null
# bytes" and crashloops honcho-api. Harmless on Linux; essential on
# macOS builds.
COPYFILE_DISABLE=1 tar -C "${SCRIPT_DIR}" -czf "${BUILD_DIR}/context.tar.gz" "${CONTEXT_INCLUDES[@]}"
CONTEXT_SIZE=$(du -k "${BUILD_DIR}/context.tar.gz" | cut -f1)
echo "    Context size: ${CONTEXT_SIZE}KB"

if (( CONTEXT_SIZE > 1000 )); then
    echo "ERROR: context tarball (${CONTEXT_SIZE}KB) exceeds ConfigMap 1MiB limit." >&2
    echo "  Trim the source tree or use a PVC-based context." >&2
    exit 1
fi

CONTEXT_CM="${IMAGE}-build-context-${TIMESTAMP}"
kubectl -n "${NAMESPACE}" create configmap "${CONTEXT_CM}" \
    --from-file=context.tar.gz="${BUILD_DIR}/context.tar.gz"

cleanup() {
    kubectl -n "${NAMESPACE}" delete configmap "${CONTEXT_CM}" --ignore-not-found 2>/dev/null || true
    for arch in "${ARCHITECTURES[@]}"; do
        kubectl -n "${NAMESPACE}" delete job "buildkit-${IMAGE}-${arch}-${TIMESTAMP}" --ignore-not-found 2>/dev/null || true
    done
    rm -rf "${CRANE_DOCKER_CONFIG}" "${BUILD_DIR}"
}
trap cleanup EXIT

# Submit one Job per arch, targeting the per-arch buildkitd Service.
declare -a JOB_NAMES=()
for arch in "${ARCHITECTURES[@]}"; do
    job_name="buildkit-${IMAGE}-${arch}-${TIMESTAMP}"
    JOB_NAMES+=("${job_name}")
    arch_tag="${IMAGE}:${TAG}-${arch}"
    cache_ref="${REGISTRY}/${IMAGE}/buildkit-cache:${arch}"

    echo "==> Submitting ${arch} build job ${job_name}"
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job_name}
  namespace: ${NAMESPACE}
spec:
  ttlSecondsAfterFinished: 600
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      initContainers:
        - name: extract-context
          image: busybox:latest
          command: ["/bin/sh","-c"]
          args: ["set -ex; mkdir -p /workspace; tar -C /workspace -xzf /context/context.tar.gz; ls /workspace | head"]
          volumeMounts:
            - { name: workspace, mountPath: /workspace }
            - { name: context, mountPath: /context }
      containers:
        - name: buildctl
          image: ${BUILDKIT_IMAGE}
          command: ["buildctl"]
          args:
            - "--addr"
            - "tcp://buildkitd-${arch}.buildkit.svc:1234"
            - "build"
            - "--frontend"
            - "dockerfile.v0"
            - "--local"
            - "context=/workspace"
            - "--local"
            - "dockerfile=/workspace"
            - "--opt"
            - "platform=linux/${arch}"
            - "--opt"
            - "label:org.opencontainers.image.revision=${SOURCE_SHA}"
            - "--output"
            - "type=image,name=${REGISTRY}/${arch_tag},push=true,registry.insecure=true"
            - "--export-cache"
            - "type=registry,ref=${cache_ref},mode=max,registry.insecure=true"
            - "--import-cache"
            - "type=registry,ref=${cache_ref},registry.insecure=true,ignoreerror=true"
          resources:
            requests:
              cpu: 100m
              memory: 256Mi
            limits:
              memory: 1Gi
          volumeMounts:
            - { name: workspace, mountPath: /workspace }
      volumes:
        - name: workspace
          emptyDir: {}
        - name: context
          configMap:
            name: ${CONTEXT_CM}
EOF
done

echo ""
echo "==> Streaming concurrent builds"
declare -a PIDS=()
for i in "${!ARCHITECTURES[@]}"; do
    arch="${ARCHITECTURES[$i]}"
    job_name="${JOB_NAMES[$i]}"
    (
        # kubectl wait returns immediately if no resources match yet, so
        # wait for the pod to exist before waiting for it to be ready, and
        # give the completion wait enough headroom for a cold-cache build
        # with a rewritten uv.lock (~10min on these arches).
        kubectl wait --for=create pod -l "job-name=${job_name}" \
            -n "${NAMESPACE}" --timeout=120s >/dev/null 2>&1 || true
        kubectl wait --for=condition=ready pod -l "job-name=${job_name}" \
            -n "${NAMESPACE}" --timeout=300s >/dev/null 2>&1 || true
        kubectl logs -f "job/${job_name}" -n "${NAMESPACE}" 2>&1 \
            | sed "s/^/[${arch}] /" || true
        if kubectl wait --for=condition=complete "job/${job_name}" \
                -n "${NAMESPACE}" --timeout=1200s >/dev/null 2>&1; then
            echo "[${arch}] build complete"
        else
            echo "[${arch}] build FAILED" >&2
            kubectl logs "job/${job_name}" -n "${NAMESPACE}" --tail=50 2>/dev/null \
                | sed "s/^/[${arch}-err] /" >&2 || true
            exit 1
        fi
    ) &
    PIDS+=($!)
done

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || FAILED=1
done
(( FAILED )) && exit 1

echo ""
echo "==> Creating multi-arch manifest"
ARCH_REFS=()
for arch in "${ARCHITECTURES[@]}"; do
    ARCH_REFS+=("${REGISTRY}/${IMAGE}:${TAG}-${arch}")
done
crane index append --insecure \
    --tag "${REGISTRY}/${IMAGE}:${TAG}" \
    --manifest "${ARCH_REFS[0]}" \
    --manifest "${ARCH_REFS[1]}" \
    --flatten

WALL_END=$(date +%s)
echo ""
echo "==> Done in $(( WALL_END - WALL_START ))s"
echo "    Pushed: ${REGISTRY}/${IMAGE}:${TAG} (${SOURCE_SHA})"

if [[ "${RESTART}" == "true" ]]; then
    echo ""
    echo "==> Restarting honcho-api and honcho-deriver Deployments"
    kubectl -n apps rollout restart deployment/honcho-api deployment/honcho-deriver
    kubectl -n apps rollout status deployment/honcho-api --timeout=180s
    kubectl -n apps rollout status deployment/honcho-deriver --timeout=180s
fi
