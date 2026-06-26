"""Unit tests for k8s manifest correctness and security posture.

These tests render the Kustomize output and assert structural and security
properties without requiring a running Kubernetes cluster. They run in the
standard pytest suite alongside the application tests.

Requirements: kubectl must be on PATH (kubectl kustomize is built-in since v1.14).
"""

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixture: render kustomize output once per session
# ---------------------------------------------------------------------------

K8S_DIR = Path(__file__).parent.parent / "k8s"


@pytest.fixture(scope="session")
def manifests() -> list[dict[str, Any]]:
    """Render `kubectl kustomize k8s/` and return parsed YAML documents."""
    try:
        result = subprocess.run(
            ["kubectl", "kustomize", str(K8S_DIR)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        pytest.skip("kubectl not found on PATH — skipping k8s manifest tests")
    except subprocess.TimeoutExpired:
        pytest.fail("kubectl kustomize timed out after 30 s")
    assert result.returncode == 0, (
        f"kubectl kustomize failed:\n{result.stderr}"
    )
    docs = list(yaml.safe_load_all(result.stdout))
    return [d for d in docs if d is not None]


def _by_kind_name(
    manifests: list[dict[str, Any]], kind: str, name: str
) -> dict[str, Any]:
    """Return a single manifest by kind and metadata.name."""
    matches = [
        m
        for m in manifests
        if m.get("kind") == kind and m.get("metadata", {}).get("name") == name
    ]
    assert len(matches) == 1, f"Expected exactly 1 {kind}/{name}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Rendering smoke test
# ---------------------------------------------------------------------------


def test_kustomize_renders_without_error(manifests: list[dict[str, Any]]):
    """Kustomize must produce at least the core resource types."""
    kinds = {m["kind"] for m in manifests}
    expected = {
        "Namespace",
        "ConfigMap",
        "NetworkPolicy",
        "StatefulSet",
        "Service",
        "Deployment",
        "HorizontalPodAutoscaler",
        "PodDisruptionBudget",
    }
    assert expected.issubset(kinds), f"Missing kinds: {expected - kinds}"


def test_all_resources_in_honcho_namespace(manifests: list[dict[str, Any]]):
    """Every namespaced resource must be in the 'honcho' namespace."""
    # These kinds don't carry a namespace.
    cluster_scoped = {"Namespace"}
    for m in manifests:
        if m["kind"] in cluster_scoped:
            continue
        ns = m.get("metadata", {}).get("namespace")
        assert ns == "honcho", (
            f"{m['kind']}/{m['metadata']['name']} has namespace '{ns}', expected 'honcho'"
        )


# ---------------------------------------------------------------------------
# Network policy tests
# ---------------------------------------------------------------------------


def test_default_deny_ingress_policy_exists(manifests: list[dict[str, Any]]):
    """A default-deny NetworkPolicy must select all pods and allow no ingress."""
    policy = _by_kind_name(manifests, "NetworkPolicy", "default-deny-ingress")
    assert policy["spec"]["podSelector"] == {}, (
        "default-deny must select all pods (empty podSelector)"
    )
    assert "Ingress" in policy["spec"]["policyTypes"]
    assert policy["spec"].get("ingress", []) == [], (
        "default-deny policy must not contain any ingress allow rules"
    )


def test_postgres_network_policy_restricts_access(manifests: list[dict[str, Any]]):
    """Only honcho-api and honcho-deriver pods may reach postgres — no other sources."""
    policy = _by_kind_name(manifests, "NetworkPolicy", "allow-postgres-from-honcho")
    assert policy["spec"].get("podSelector", {}).get("matchLabels") == {"app": "postgres"}, (
        "allow-postgres-from-honcho must select postgres pods"
    )
    ingress_rules = policy["spec"].get("ingress", [])
    assert len(ingress_rules) == 1, (
        f"allow-postgres-from-honcho must have exactly 1 ingress rule, got {len(ingress_rules)}"
    )
    allowed_labels = {
        frozenset(src.get("podSelector", {}).get("matchLabels", {}).items())
        for src in ingress_rules[0]["from"]
    }
    assert allowed_labels == {
        frozenset({"app": "honcho-api"}.items()),
        frozenset({"app": "honcho-deriver"}.items()),
    }, f"postgres ingress sources must be exactly api+deriver, got: {allowed_labels}"
    ports = {p["port"] for p in ingress_rules[0]["ports"]}
    assert ports == {5432}, f"postgres ingress must allow only port 5432, got: {ports}"


def test_redis_network_policy_restricts_access(manifests: list[dict[str, Any]]):
    """Only honcho-api and honcho-deriver pods may reach redis — no other sources."""
    policy = _by_kind_name(manifests, "NetworkPolicy", "allow-redis-from-honcho")
    assert policy["spec"].get("podSelector", {}).get("matchLabels") == {"app": "redis"}, (
        "allow-redis-from-honcho must select redis pods"
    )
    ingress_rules = policy["spec"].get("ingress", [])
    assert len(ingress_rules) == 1, (
        f"allow-redis-from-honcho must have exactly 1 ingress rule, got {len(ingress_rules)}"
    )
    allowed_labels = {
        frozenset(src.get("podSelector", {}).get("matchLabels", {}).items())
        for src in ingress_rules[0]["from"]
    }
    assert allowed_labels == {
        frozenset({"app": "honcho-api"}.items()),
        frozenset({"app": "honcho-deriver"}.items()),
    }, f"redis ingress sources must be exactly api+deriver, got: {allowed_labels}"
    ports = {p["port"] for p in ingress_rules[0]["ports"]}
    assert ports == {6379}, f"redis ingress must allow only port 6379, got: {ports}"


def test_four_network_policies_present(manifests: list[dict[str, Any]]):
    """Exactly 4 NetworkPolicy resources must be present."""
    policies = [m for m in manifests if m["kind"] == "NetworkPolicy"]
    names = {m["metadata"]["name"] for m in policies}
    assert names == {
        "default-deny-ingress",
        "allow-api-ingress",
        "allow-postgres-from-honcho",
        "allow-redis-from-honcho",
    }


def test_allow_api_ingress_policy_semantics(manifests: list[dict[str, Any]]):
    """allow-api-ingress must target API pods and allow exactly port 8000/TCP."""
    policy = _by_kind_name(manifests, "NetworkPolicy", "allow-api-ingress")
    deployment = _by_kind_name(manifests, "Deployment", "honcho-api")

    # Policy podSelector must match the Deployment's pod template labels.
    pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert policy["spec"]["podSelector"].get("matchLabels") == pod_labels, (
        f"allow-api-ingress podSelector {policy['spec']['podSelector']} "
        f"must match API pod template labels {pod_labels}"
    )

    # Ingress must have exactly one rule allowing exactly port 8000/TCP.
    ingress_rules = policy["spec"]["ingress"]
    assert len(ingress_rules) == 1, (
        f"allow-api-ingress must have exactly 1 ingress rule, got {len(ingress_rules)}"
    )
    ingress_ports = ingress_rules[0]["ports"]
    assert len(ingress_ports) == 1, (
        f"allow-api-ingress ingress rule must specify exactly 1 port, got {ingress_ports}"
    )
    entry = ingress_ports[0]
    assert entry.get("port") == 8000, (
        f"allow-api-ingress must allow port 8000, got {entry.get('port')}"
    )
    assert entry.get("protocol", "TCP") == "TCP", (
        f"allow-api-ingress must use TCP protocol, got {entry.get('protocol')}"
    )


# ---------------------------------------------------------------------------
# StatefulSet headless Service tests
# ---------------------------------------------------------------------------


def test_postgres_service_is_headless(manifests: list[dict[str, Any]]):
    """The postgres Service must be headless (clusterIP: None) for the StatefulSet."""
    svc = _by_kind_name(manifests, "Service", "postgres")
    assert svc["spec"].get("clusterIP") == "None", (
        "postgres Service must be headless (clusterIP: None)"
    )


def test_redis_service_is_headless(manifests: list[dict[str, Any]]):
    """The redis Service must be headless (clusterIP: None) for the StatefulSet."""
    svc = _by_kind_name(manifests, "Service", "redis")
    assert svc["spec"].get("clusterIP") == "None", (
        "redis Service must be headless (clusterIP: None)"
    )


def test_postgres_statefulset_servicename_matches_service(
    manifests: list[dict[str, Any]],
):
    """StatefulSet.spec.serviceName must match the headless Service name."""
    sts = _by_kind_name(manifests, "StatefulSet", "postgres")
    assert sts["spec"]["serviceName"] == "postgres"


def test_redis_statefulset_servicename_matches_service(
    manifests: list[dict[str, Any]],
):
    sts = _by_kind_name(manifests, "StatefulSet", "redis")
    assert sts["spec"]["serviceName"] == "redis"


# ---------------------------------------------------------------------------
# HPA / Deployment replica interaction
# ---------------------------------------------------------------------------


def test_api_deployment_has_no_static_replicas(manifests: list[dict[str, Any]]):
    """The HPA-managed Deployment must not declare a static replica count.

    A static replicas field overrides the HPA on every kubectl apply.
    """
    deployment = _by_kind_name(manifests, "Deployment", "honcho-api")
    assert "replicas" not in deployment["spec"], (
        "honcho-api Deployment must not set replicas — the HPA manages this"
    )


def test_hpa_targets_api_deployment(manifests: list[dict[str, Any]]):
    hpa = _by_kind_name(manifests, "HorizontalPodAutoscaler", "honcho-api")
    ref = hpa["spec"]["scaleTargetRef"]
    assert ref["kind"] == "Deployment"
    assert ref["name"] == "honcho-api"
    assert hpa["spec"]["minReplicas"] == 1, (
        f"HPA minReplicas must be 1, got {hpa['spec']['minReplicas']}"
    )
    assert hpa["spec"]["maxReplicas"] == 5, (
        f"HPA maxReplicas must be 5, got {hpa['spec']['maxReplicas']}"
    )
    cpu_metrics = [
        m for m in hpa["spec"]["metrics"]
        if m.get("type") == "Resource"
        and m.get("resource", {}).get("name") == "cpu"
    ]
    assert cpu_metrics, "HPA must define a CPU Resource metric"
    utilization = cpu_metrics[0]["resource"]["target"].get("averageUtilization")
    assert utilization == 70, (
        f"HPA CPU target averageUtilization must be 70, got {utilization}"
    )


def test_pdb_selects_api_pods(manifests: list[dict[str, Any]]):
    pdb = _by_kind_name(manifests, "PodDisruptionBudget", "honcho-api")
    assert pdb["spec"]["selector"]["matchLabels"] == {"app": "honcho-api"}
    # maxUnavailable: 1 allows node drains even at minReplicas=1; minAvailable: 1
    # would deadlock when only one replica is running.
    assert "maxUnavailable" in pdb["spec"] and "minAvailable" not in pdb["spec"], (
        "PDB must not define both maxUnavailable and minAvailable; "
        "use maxUnavailable: 1 only"
    )
    assert pdb["spec"].get("maxUnavailable") == 1, (
        "PDB must use maxUnavailable: 1 to avoid deadlock with HPA minReplicas: 1"
    )


# ---------------------------------------------------------------------------
# Deriver — no HTTP probe, correct command
# ---------------------------------------------------------------------------


def test_deriver_has_no_http_probes(manifests: list[dict[str, Any]]):
    """The deriver is a queue worker, not an HTTP server.

    Neither a liveness nor a readiness HTTP probe must be present — the
    deriver has no HTTP server to probe. Health is managed by the restart
    policy (restartPolicy: Always, the Deployment default).
    """
    deployment = _by_kind_name(manifests, "Deployment", "honcho-deriver")
    container = next(
        c
        for c in deployment["spec"]["template"]["spec"]["containers"]
        if c["name"] == "deriver"
    )
    assert "livenessProbe" not in container, (
        "deriver must not have a livenessProbe — it is not an HTTP server"
    )
    readiness = container.get("readinessProbe", {})
    assert "httpGet" not in readiness, (
        "deriver must not have an HTTP readinessProbe — it is not an HTTP server"
    )


def test_deriver_runs_correct_command(manifests: list[dict[str, Any]]):
    deployment = _by_kind_name(manifests, "Deployment", "honcho-deriver")
    container = next(
        c
        for c in deployment["spec"]["template"]["spec"]["containers"]
        if c["name"] == "deriver"
    )
    assert container["command"] == ["/app/.venv/bin/python", "-m", "src.deriver"]


# ---------------------------------------------------------------------------
# API — differentiated probe types
# ---------------------------------------------------------------------------


def test_api_readiness_probe_is_http(manifests: list[dict[str, Any]]):
    """Readiness probe must use httpGet to verify the app is serving requests."""
    deployment = _by_kind_name(manifests, "Deployment", "honcho-api")
    container = next(
        c
        for c in deployment["spec"]["template"]["spec"]["containers"]
        if c["name"] == "api"
    )
    readiness = container["readinessProbe"]
    assert "httpGet" in readiness, "readinessProbe must use httpGet"
    assert readiness["httpGet"]["path"] == "/openapi.json"


def test_api_liveness_probe_is_tcp(manifests: list[dict[str, Any]]):
    """Liveness probe must use tcpSocket (different type from readiness).

    Using different probe types prevents the identical-probe anti-pattern:
    port open (liveness) != app serving responses (readiness).
    """
    deployment = _by_kind_name(manifests, "Deployment", "honcho-api")
    container = next(
        c
        for c in deployment["spec"]["template"]["spec"]["containers"]
        if c["name"] == "api"
    )
    liveness = container["livenessProbe"]
    assert "tcpSocket" in liveness, "livenessProbe must use tcpSocket"
    assert "httpGet" not in liveness, (
        "livenessProbe must not use httpGet (would be identical to readinessProbe)"
    )


# ---------------------------------------------------------------------------
# Security context tests
# ---------------------------------------------------------------------------


def _get_container(deployment: dict[str, Any], name: str) -> dict[str, Any]:
    containers = (
        deployment["spec"]["template"]["spec"].get("containers", [])
        + deployment["spec"]["template"]["spec"].get("initContainers", [])
    )
    return next(c for c in containers if c["name"] == name)


def test_api_container_runs_as_non_root(manifests: list[dict[str, Any]]):
    deployment = _by_kind_name(manifests, "Deployment", "honcho-api")
    # Pod-level
    pod_sc = deployment["spec"]["template"]["spec"].get("securityContext", {})
    assert pod_sc.get("runAsNonRoot") is True, "Pod-level runAsNonRoot must be true"
    # Container-level
    container = _get_container(deployment, "api")
    assert container["securityContext"].get("runAsNonRoot") is True


def test_api_container_no_privilege_escalation(manifests: list[dict[str, Any]]):
    deployment = _by_kind_name(manifests, "Deployment", "honcho-api")
    container = _get_container(deployment, "api")
    assert container["securityContext"].get("allowPrivilegeEscalation") is False


def test_api_container_drops_all_capabilities(manifests: list[dict[str, Any]]):
    deployment = _by_kind_name(manifests, "Deployment", "honcho-api")
    container = _get_container(deployment, "api")
    drop = container["securityContext"].get("capabilities", {}).get("drop", [])
    assert "ALL" in drop


def test_api_container_readonly_root_filesystem(manifests: list[dict[str, Any]]):
    deployment = _by_kind_name(manifests, "Deployment", "honcho-api")
    container = _get_container(deployment, "api")
    assert container["securityContext"].get("readOnlyRootFilesystem") is True


def test_api_pod_has_seccomp_profile(manifests: list[dict[str, Any]]):
    deployment = _by_kind_name(manifests, "Deployment", "honcho-api")
    pod_sc = deployment["spec"]["template"]["spec"].get("securityContext", {})
    assert pod_sc.get("seccompProfile", {}).get("type") == "RuntimeDefault"


def test_deriver_container_security_mirrors_api(manifests: list[dict[str, Any]]):
    """Deriver must have the same security posture as the API."""
    deployment = _by_kind_name(manifests, "Deployment", "honcho-deriver")
    pod_sc = deployment["spec"]["template"]["spec"].get("securityContext", {})
    assert pod_sc.get("runAsNonRoot") is True
    assert pod_sc.get("seccompProfile", {}).get("type") == "RuntimeDefault"
    container = _get_container(deployment, "deriver")
    sc = container["securityContext"]
    assert sc.get("runAsNonRoot") is True
    assert sc.get("allowPrivilegeEscalation") is False
    assert sc.get("readOnlyRootFilesystem") is True
    assert "ALL" in sc.get("capabilities", {}).get("drop", [])


def test_init_containers_run_as_nobody(manifests: list[dict[str, Any]]):
    """busybox init containers must run as UID/GID 65534 (nobody/nogroup)."""
    for deployment_name in ("honcho-api", "honcho-deriver"):
        deployment = _by_kind_name(manifests, "Deployment", deployment_name)
        init_containers = deployment["spec"]["template"]["spec"].get(
            "initContainers", []
        )
        assert len(init_containers) >= 2, (
            f"{deployment_name} must have at least 2 init containers"
        )
        for ic in init_containers:
            sc = ic.get("securityContext", {})
            assert sc.get("runAsUser") == 65534, (
                f"Init container '{ic['name']}' in {deployment_name} must run as UID 65534"
            )
            assert sc.get("runAsGroup") == 65534, (
                f"Init container '{ic['name']}' in {deployment_name} must run as GID 65534"
            )


def test_all_workloads_no_service_account_token(manifests: list[dict[str, Any]]):
    """No workload pod should mount a service account token."""
    workloads = [
        ("Deployment", "honcho-api"),
        ("Deployment", "honcho-deriver"),
        ("StatefulSet", "postgres"),
        ("StatefulSet", "redis"),
    ]
    for kind, name in workloads:
        workload = _by_kind_name(manifests, kind, name)
        assert (
            workload["spec"]["template"]["spec"].get("automountServiceAccountToken")
            is False
        ), f"{kind}/{name} must set automountServiceAccountToken: false"


# ---------------------------------------------------------------------------
# Secrets vs ConfigMap split
# ---------------------------------------------------------------------------


def test_configmap_contains_no_secret_keys(manifests: list[dict[str, Any]]):
    """Sensitive values must not appear in the ConfigMap."""
    cm = _by_kind_name(manifests, "ConfigMap", "honcho-config")
    data = cm.get("data", {})
    forbidden = {
        "DB_CONNECTION_URI",
        "AUTH_JWT_SECRET",
        "POSTGRES_PASSWORD",
        "LLM_ANTHROPIC_API_KEY",
        "LLM_GEMINI_API_KEY",
        "LLM_OPENAI_API_KEY",
    }
    leaked = forbidden & set(data.keys())
    assert not leaked, f"Secret key(s) found in ConfigMap: {leaked}"


def test_db_uri_comes_from_secret(manifests: list[dict[str, Any]]):
    """DB_CONNECTION_URI env var must be sourced from a Secret, not a ConfigMap."""
    for deployment_name in ("honcho-api", "honcho-deriver"):
        deployment = _by_kind_name(manifests, "Deployment", deployment_name)
        container = next(
            c
            for c in deployment["spec"]["template"]["spec"]["containers"]
            if c["name"] in ("api", "deriver")
        )
        db_env = next(
            e for e in container["env"] if e["name"] == "DB_CONNECTION_URI"
        )
        assert "secretKeyRef" in db_env.get("valueFrom", {}), (
            f"DB_CONNECTION_URI in {deployment_name} must come from a secretKeyRef"
        )


# ---------------------------------------------------------------------------
# Readiness/liveness probe differentiation for StatefulSets
# ---------------------------------------------------------------------------


def test_postgres_probes_use_different_types(manifests: list[dict[str, Any]]):
    """postgres readiness (exec) and liveness (tcpSocket) must differ."""
    sts = _by_kind_name(manifests, "StatefulSet", "postgres")
    container = sts["spec"]["template"]["spec"]["containers"][0]
    assert "exec" in container["readinessProbe"]
    assert "tcpSocket" in container["livenessProbe"]


def test_redis_probes_use_different_types(manifests: list[dict[str, Any]]):
    """redis readiness (exec) and liveness (tcpSocket) must differ."""
    sts = _by_kind_name(manifests, "StatefulSet", "redis")
    container = sts["spec"]["template"]["spec"]["containers"][0]
    assert "exec" in container["readinessProbe"]
    assert "tcpSocket" in container["livenessProbe"]


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------


def _assert_has_resource_limits(container: dict[str, Any], label: str):
    resources = container.get("resources", {})
    assert "limits" in resources, f"{label}: missing resource limits"
    assert "requests" in resources, f"{label}: missing resource requests"
    for kind in ("limits", "requests"):
        for field in ("memory", "cpu", "ephemeral-storage"):
            assert field in resources[kind], (
                f"{label}: missing resources.{kind}.{field}"
            )


def test_all_containers_have_resource_limits(manifests: list[dict[str, Any]]):
    """Every container (including init containers) must declare resource limits."""
    for m in manifests:
        if m["kind"] not in ("Deployment", "StatefulSet"):
            continue
        name = m["metadata"]["name"]
        spec = m["spec"]["template"]["spec"]
        for c in spec.get("containers", []) + spec.get("initContainers", []):
            _assert_has_resource_limits(c, f"{name}/{c['name']}")
