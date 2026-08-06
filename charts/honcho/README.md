# Honcho Helm chart

This chart deploys the Honcho API server, deriver worker, an optional Redis
instance, and an optional database migration Job.

The chart expects PostgreSQL with the `pgvector` extension to be available. Use a
managed PostgreSQL service, an existing in-cluster database, or enable the
optional CloudNativePG template if your cluster already runs the CloudNativePG
operator.

## Install

Create a runtime Secret with the required application settings:

```bash
kubectl create namespace honcho
kubectl create secret generic honcho-runtime \
  --namespace honcho \
  --from-literal=DB_CONNECTION_URI='postgresql+psycopg://postgres:postgres@postgres.example.com:5432/postgres' \
  --from-literal=LLM_OPENAI_API_KEY='sk-...' \
  --from-literal=AUTH_JWT_SECRET='replace-with-a-random-secret'
```

Install the chart:

```bash
helm install honcho ./charts/honcho \
  --namespace honcho \
  --set runtimeSecret.enabled=true \
  --set runtimeSecret.name=honcho-runtime \
  --set database.host=postgres.example.com
```

When the included unauthenticated Redis deployment is enabled, the chart sets
`CACHE_URL` automatically. If you disable the included Redis deployment or enable
Redis authentication, provide `CACHE_URL` through the runtime Secret or
`config.env`.

## Configuration

Common values:

| Value | Description | Default |
| --- | --- | --- |
| `image.repository` | Honcho container image repository | `ghcr.io/plastic-labs/honcho` |
| `image.tag` | Honcho image tag | chart `appVersion` |
| `runtimeSecret.enabled` | Add an `envFrom` reference to a runtime Secret | `false` |
| `runtimeSecret.name` | Runtime Secret name | `<release>-honcho-runtime` |
| `database.host` | PostgreSQL host used by dependency wait init containers | `""` |
| `redis.enabled` | Deploy a Redis instance for Honcho cache/queue use | `true` |
| `migration.enabled` | Run `scripts/provision_db.py` as a Helm hook | `true` |
| `ingress.enabled` | Create an Ingress for the API service | `false` |
| `monitoring.podMonitor.enabled` | Create a Prometheus Operator PodMonitor | `false` |
| `networkPolicy.enabled` | Create NetworkPolicy resources | `false` |

The runtime Secret commonly includes:

```text
DB_CONNECTION_URI
CACHE_URL
AUTH_JWT_SECRET
LLM_OPENAI_API_KEY
LLM_ANTHROPIC_API_KEY
LLM_GEMINI_API_KEY
```

Honcho can also read any other supported environment variable through
`config.env`, `extraEnv`, or `extraEnvFrom`.

## Optional CloudNativePG database

If CloudNativePG is installed in the cluster, the chart can create a PostgreSQL
cluster:

```bash
kubectl create secret generic honcho-db-credentials \
  --namespace honcho \
  --from-literal=username=honcho \
  --from-literal=password='replace-with-a-password'

helm upgrade --install honcho ./charts/honcho \
  --namespace honcho \
  --set cnpg.enabled=true \
  --set cnpg.credentialsSecret.name=honcho-db-credentials \
  --set runtimeSecret.enabled=true \
  --set runtimeSecret.name=honcho-runtime
```

Set `DB_CONNECTION_URI` in `honcho-runtime` to the CloudNativePG read-write
service, for example:

```text
postgresql+psycopg://honcho:<password>@honcho-db-rw.honcho.svc.cluster.local:5432/honcho
```

The CNPG template is disabled by default so the chart can be used with managed
databases and clusters that do not install the CloudNativePG CRDs.

## Validate

```bash
helm lint charts/honcho
helm template honcho charts/honcho \
  --set runtimeSecret.enabled=true \
  --set runtimeSecret.name=honcho-runtime \
  --set database.host=postgres.example.com
```
