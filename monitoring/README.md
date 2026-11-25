# Prometheus monitoring

FastAPI exposes a request counter, `api_requests_total{endpoint,status}`, and a
latency histogram in seconds, `api_request_seconds{endpoint}`. The middleware
measures time until a response is produced or an exception is raised. Errors are
counted by HTTP status, including validation errors and unhandled 500s. Database
logging failures handled inside the API retain the prediction's HTTP status.
Matched routes use route templates. Unmatched paths share the `unmatched` label,
and `/metrics` scrapes are excluded from request counts and latency.

## Run locally

For a browser-only walkthrough, open [the HTML demo](prometheus-demo.html).
It simulates traffic and scrapes with synthetic data and needs no running services.

From the repository root, use Python 3.13 and native Prometheus binaries
(`prometheus` and `promtool`). Keep `MODEL_URI` and `DATABASE_URL` unset for this
check so the API uses its placeholder model without PostgreSQL.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt uvicorn
python -m pytest
ruff check .
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal, start Prometheus with the checked-in five-second scrape job.

```sh
promtool check config monitoring/prometheus.yml
prometheus --config.file=monitoring/prometheus.yml \
  --web.listen-address=127.0.0.1:9090 \
  --storage.tsdb.path="$(mktemp -d)"
```

Wait for `spark-check-api` to show **UP** at <http://127.0.0.1:9090/targets>.
In a third terminal, generate successful requests and validation errors across
several scrape intervals, then inspect the exporter.

```sh
for i in $(seq 1 20); do
  curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health
  curl -sS -o /dev/null -w '%{http_code}\n' -H 'Content-Type: application/json' \
    -d '{}' http://127.0.0.1:8000/predict
  sleep 1
done
curl -fsS http://127.0.0.1:8000/metrics | grep '^api_'
```

Expect HTTP 200 and 422. Run these expressions in the Prometheus query UI
at <http://127.0.0.1:9090/>. Rate and percentile queries need at least two samples
per series. The automated tests also exercise an injected 500.

| What to check | PromQL |
| --- | --- |
| Scrape health | `up{job="spark-check-api"}` |
| Requests since API start, by endpoint and status | `api_requests_total{job="spark-check-api"}` |
| Requests per second by endpoint | `sum by (endpoint) (rate(api_requests_total{job="spark-check-api"}[1m]))` |
| Approximate p95 latency in seconds by endpoint | `histogram_quantile(0.95, sum by (endpoint, le) (rate(api_request_seconds_bucket{job="spark-check-api"}[1m])))` |
| Client and server errors per second, separately by status | `sum by (status) (rate(api_requests_total{job="spark-check-api",status=~"[45].."}[1m]))` |

Stop both servers with Ctrl+C.

## Compose handoff

The scrape target uses loopback for two host processes. For the shared Compose
stack, mount the config and set the target to the API service's DNS name and
internal port (for example, `api:8000` if the service is named `api`).
Keep the scrape job name stable for Grafana queries. The exporter uses
process-local metrics, so use one Uvicorn worker per API container for this
setup. If the stack later uses replicas, scrape each API instance individually.
