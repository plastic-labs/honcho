# Running Honcho on Kubernetes

This directory contains Kubernetes manifests for deploying all Honcho services. The setup is managed with [Kustomize](https://kustomize.io/), which is built into `kubectl` v1.14+.

## What gets deployed

| Resource | Kind | Notes |
|----------|------|-------|
| `postgres` | StatefulSet + Headless Service | pgvector/pgvector:pg15, 10 Gi PVC |
| `redis` | StatefulSet + Headless Service | redis:8.2, 2 Gi PVC |
| `honcho-api` | Deployment + ClusterIP Service | FastAPI server; runs migrations on start |
| `honcho-deriver` | Deployment (no Service) | Background queue worker |
| `honcho-api` | HorizontalPodAutoscaler | 1–5 replicas at 70% CPU |
| `honcho-api` | PodDisruptionBudget | maxUnavailable: 1 (allows drains at minReplicas=1) |
| NetworkPolicies | default-deny + allow rules | Postgres/Redis reachable only from API/Deriver |

## Testing the manifests

A pytest test suite ships with the manifests at `tests/test_k8s_manifests.py`. It renders the Kustomize output and asserts structural correctness, security posture, and architectural invariants without requiring a running cluster.

```bash
# Requires: kubectl on PATH, pytest, pyyaml
pytest tests/test_k8s_manifests.py --noconftest -p no:xdist --override-ini="addopts=" -v
```

To also validate schemas against the official Kubernetes API spec and run security scoring:

```bash
# Schema validation (install: brew install kubeconform)
kubectl kustomize k8s/ | kubeconform -strict -summary

# Security and best-practices scoring (install: brew install kube-score)
kubectl kustomize k8s/ | kube-score score -

# Misconfiguration scanning (install: brew install trivy)
trivy config k8s/
```

---

## Prerequisites

- `kubectl` v1.14+ (Kustomize is built in)
- Docker (to build the Honcho image)
- One of the following local Kubernetes distributions:
  - [k3s](https://k3s.io/)
  - [kind](https://kind.sigs.k8s.io/) (Kubernetes IN Docker)
  - [Docker Desktop](https://docs.docker.com/desktop/kubernetes/) with Kubernetes enabled

---

## Quick start

### Step 1 — Build the image

From the repository root:

```bash
docker build -t honcho:latest .
```

### Step 2 — Load the image into your cluster

The image must be available inside the cluster. How to do this depends on your distribution:

**kind**

```bash
kind load docker-image honcho:latest --name <your-cluster-name>
```

If you haven't created a cluster yet:

```bash
kind create cluster --name honcho
kind load docker-image honcho:latest --name honcho
```

**k3s**

```bash
docker save honcho:latest | sudo k3s ctr images import -
```

**Docker Desktop**

No extra step needed — Docker Desktop shares its daemon with Kubernetes, so any locally-built image is already available.

---

### Step 3 — Configure secrets

Honcho requires a small set of secrets before it can start. Copy the template and fill in your values:

```bash
cp k8s/secrets.yaml.example k8s/secrets.yaml
```

Open `k8s/secrets.yaml` and replace every placeholder:

| Key | What to put here |
|-----|-----------------|
| `POSTGRES_PASSWORD` | A strong password for PostgreSQL |
| `DB_CONNECTION_URI` | Full URI — use the same password as above |
| `AUTH_JWT_SECRET` | Random 32-byte hex string (see below) |
| `LLM_ANTHROPIC_API_KEY` | Your Anthropic API key (if using Anthropic models) |
| `LLM_GEMINI_API_KEY` | Your Google Gemini API key (if using Gemini models) |
| `LLM_OPENAI_API_KEY` | Your OpenAI API key (if using OpenAI models) |

Generate `AUTH_JWT_SECRET`:

```bash
openssl rand -hex 32
# or
uv run python scripts/generate_jwt_secret.py
```

`k8s/secrets.yaml` is listed in `.gitignore` — **never commit it**.

The `honcho` namespace must exist before the Secret can be applied. Create it first:

```bash
kubectl apply -f k8s/namespace.yaml
```

Then apply the secrets:

```bash
kubectl apply -f k8s/secrets.yaml
```

---

### Step 4 — Deploy

```bash
kubectl apply -k k8s/
```

Kustomize applies all resources in dependency order. Watch pods come up:

```bash
kubectl get pods -n honcho --watch
```

Expected output once everything is running:

```
NAME                            READY   STATUS    RESTARTS   AGE
honcho-api-<hash>               1/1     Running   0          2m
honcho-deriver-<hash>           1/1     Running   0          2m
postgres-0                      1/1     Running   0          3m
redis-0                         1/1     Running   0          3m
```

---

## Verify the deployment

**Check the API**

```bash
kubectl port-forward svc/honcho-api 8000:80 -n honcho
```

Then in another terminal:

```bash
curl http://localhost:8000/openapi.json | head -5
```

You should see the OpenAPI spec JSON.

**Check the deriver**

```bash
kubectl logs deploy/honcho-deriver -n honcho
```

You should see output like:

```
Starting deriver queue processor
Running main loop
ReconcilerScheduler started ...
```

The deriver has no HTTP server and no HTTP healthcheck — its health is managed by the Kubernetes restart policy. If the process exits, Kubernetes restarts it automatically.

**Check the API logs**

```bash
kubectl logs deploy/honcho-api -n honcho
```

Look for the alembic migration output followed by the FastAPI startup message.

---

## Useful commands

```bash
# Show all Honcho resources
kubectl get all -n honcho

# Stream API logs
kubectl logs -f deploy/honcho-api -n honcho

# Stream deriver logs
kubectl logs -f deploy/honcho-deriver -n honcho

# Open a shell in the API pod
kubectl exec -it deploy/honcho-api -n honcho -- bash

# Port-forward to postgres (for local DB inspection)
kubectl port-forward svc/postgres 5432:5432 -n honcho

# Port-forward to redis
kubectl port-forward svc/redis 6379:6379 -n honcho

# Check HPA status
kubectl get hpa -n honcho

# Check NetworkPolicies
kubectl get networkpolicy -n honcho

# Tear down everything (preserves PVCs — your data survives)
kubectl delete -k k8s/

# Tear down including persistent data
kubectl delete -k k8s/
kubectl delete pvc --all -n honcho
```

---

## Configuration

Non-secret configuration lives in `k8s/configmap.yaml`. Edit it and re-apply with `kubectl apply -k k8s/` to update the ConfigMap — but environment variables injected via `configMapKeyRef` are only read at pod start, so running pods will **not** pick up the change automatically. Trigger a rolling restart explicitly after re-applying:

```bash
kubectl rollout restart deployment/honcho-api deployment/honcho-deriver -n honcho
```

Notable settings:

| Key | Default | Notes |
|-----|---------|-------|
| `AUTH_USE_AUTH` | `"true"` | Set to `"false"` only for local development |
| `CACHE_ENABLED` | `"false"` | Set to `"true"` to activate Redis-backed caching |
| `DERIVER_WORKERS` | `"1"` | Increase if the deriver queue is a bottleneck |
| `METRICS_ENABLED` | `"false"` | Set to `"true"` to expose a Prometheus `/metrics` endpoint |

For the full list of configuration options, see `config.toml.example` in the repository root.

---

## Exposing the API externally

The default Service type is `ClusterIP`, which is only reachable inside the cluster. For external access:

**NodePort** (works on all local distributions without extra tooling):

Edit `k8s/api/service.yaml`, change `type: ClusterIP` to `type: NodePort`, and re-apply. Kubernetes will assign a random port in the 30000–32767 range. Find it with:

```bash
kubectl get svc honcho-api -n honcho
```

**LoadBalancer** (cloud providers; or bare-metal/kind with [metallb](https://metallb.universe.tf/)):

Change `type: ClusterIP` to `type: LoadBalancer`. On cloud providers (EKS, GKE, AKS) this provisions a cloud load balancer automatically.

**Ingress** (recommended for production):

Keep the Service as `ClusterIP` and create an Ingress resource pointing to `honcho-api:80`. This works with any Ingress controller (nginx, traefik, etc.) and gives you TLS termination, path routing, and more.

---

## Autoscaling

The HPA (`k8s/api/hpa.yaml`) scales the API between 1 and 5 replicas when average CPU exceeds 70%.

**Requirements**: `metrics-server` must be running in your cluster.

- **Docker Desktop**: included by default
- **k3s**: included by default
- **kind**: install manually:

  ```bash
  kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
  ```

  On kind you may also need to patch metrics-server to disable TLS verification:

  ```bash
  kubectl patch deployment metrics-server -n kube-system \
    --type='json' \
    -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
  ```

Check scaling activity:

```bash
kubectl describe hpa honcho-api -n honcho
```

---

## Networking and security

All pod-to-pod traffic within the `honcho` namespace is governed by NetworkPolicies (`k8s/network-policies.yaml`):

- **Default deny** — all ingress is blocked unless explicitly allowed
- **API** — accepts HTTP on port 8000 from any source (external traffic, port-forward)
- **Postgres** — accepts connections on port 5432 only from API and Deriver pods
- **Redis** — accepts connections on port 6379 only from API and Deriver pods
- **Egress** — unrestricted; API and Deriver need outbound access to LLM provider APIs

NetworkPolicies are enforced by the CNI plugin. Most CNI plugins (Calico, Cilium, Flannel with the Network Policy add-on) support them. On k3s, the default CNI (Flannel) does not enforce NetworkPolicies; install Calico or Cilium as a replacement if you need policy enforcement.

---

## Production notes

These manifests are a solid starting point, but production deployments should also consider:

**Managed database and cache**
- Replace the in-cluster `postgres` StatefulSet with a managed service (Amazon RDS, Google Cloud SQL, Azure Database for PostgreSQL) with automated backups, point-in-time recovery, and multi-AZ failover.
- Replace in-cluster `redis` with Amazon ElastiCache or similar.
- Update `DB_CONNECTION_URI` and `CACHE_URL` in your secrets/configmap accordingly.

**Secrets management**
- Instead of `k8s/secrets.yaml`, integrate with a secrets manager:
  - [External Secrets Operator](https://external-secrets.io/) with AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault
  - [Vault Agent Injector](https://developer.hashicorp.com/vault/docs/platform/k8s/injector)
  - [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets) for GitOps workflows

**Image tags**
- Replace `honcho:latest` in both Deployments with a specific, immutable tag (e.g., `honcho:v2.1.1` or a full digest). Using `latest` in production makes rollbacks harder and can cause inconsistent behavior.

**TLS**
- Use cert-manager to issue TLS certificates and terminate TLS at your Ingress controller.

**Ingress controller**
- Deploy nginx-ingress, traefik, or your preferred controller and create an Ingress resource pointing to `honcho-api:80` instead of using NodePort or LoadBalancer.

**Observability**
- Set `METRICS_ENABLED: "true"` in the ConfigMap.
- Add a `Service` for the deriver on port 9090 and a `ServiceMonitor` (if using the Prometheus Operator) to scrape both the API and Deriver.
- The `docker/prometheus.yml` in the repository root shows the expected scrape targets.
