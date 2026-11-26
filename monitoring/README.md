# Spark Check monitoring

## Prometheus API metrics

FastAPI exposes a request counter, `api_requests_total{endpoint,status}`, and a
latency histogram in seconds, `api_request_seconds{endpoint}`. The middleware
measures time until a response is produced or an exception is raised. Errors are
counted by HTTP status, including validation errors and unhandled 500s. Database
logging failures handled inside the API retain the prediction's HTTP status.
Matched routes use route templates. Unmatched paths share the `unmatched` label,
and `/metrics` scrapes are excluded from request counts and latency.

## Run Prometheus locally

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

## Run drift and model monitoring

The evaluator reads logged API predictions from PostgreSQL, compares the five input
columns against a reference CSV with Evidently, and writes one row per model,
reference, and completed window to `monitoring_results`. A rerun updates that
row, so older windows can be recomputed when observed outcomes arrive.

1. Configure the API with `DATABASE_URL` and submit one valid prediction. Check
   that it wrote a row to the API-owned `predictions` table.
2. Use a separate Python 3.13 environment for monitoring. Its pandas version is
   pinned for Evidently compatibility.

   ```sh
   monitoring_env="$(mktemp -d)"
   python3 -m venv "$monitoring_env"
   source "$monitoring_env/bin/activate"
   python -m pip install -r monitoring/requirements.txt
   python -m monitoring.evaluate --init-db
   ```

3. Set these values in the evaluator environment or pass the equivalent CLI flags.

   | Variable | Value |
   | --- | --- |
   | `DATABASE_URL` | PostgreSQL connection string for reading the prediction log and writing monitoring results |
   | `MONITORING_MODEL_VERSION` | Exact model version written by the API, such as `placeholder` for a local smoke test |
   | `MONITORING_REFERENCE_CSV` | Path to an approved reference CSV with numeric `phq9_pre`, `loneliness_pre`, `flourishing_pre`, `stress_pre`, and `neuroticism_pre` columns |
   | `MONITORING_REFERENCE_ID` | Identifier for the exact reference snapshot used |

4. Run `python -m monitoring.evaluate --watch`. The default is a five-minute
   window with at least 20 complete rows on each side. Evidently flags
   significant drift when at least half of the features drift. These are project
   defaults for the demo. Agree on them with the training and simulator owners.
   Use `--help` for options. For a one-off backfill, omit `--watch` and pass
   `--window-end` with a timezone-aware ISO timestamp. Use the original window
   length and reference ID when recomputing a stored result.

The API records the configured MLflow URI as the model version. Coordinate a
version-specific URI with the API and training owners if you need to distinguish
results across retraining runs.

PostgreSQL stores the drift share, per-feature test results, sample counts, MAE,
and RMSE. When an authorized source provides an observed end-of-term PHQ-9
score, link it to the API row in `monitoring_outcomes(prediction_id,
observed_phq9_post)`. The performance calculation uses only matched pairs and
leaves MAE and RMSE NULL until they exist. Windows with too few complete inputs
leave drift NULL.

## Provision Grafana and the retraining webhook

The Compose owner can use `monitoring/Dockerfile` with the repository as the
build context. Run the default command for the evaluator and use the same image
for a separate internal webhook service with this command.

```sh
python -m uvicorn monitoring.alerts:app --host 0.0.0.0 --port 8010
```

For host-only testing, bind the webhook to `127.0.0.1`. Set
`PREFECT_API_URL` for the webhook service and confirm the training owner has
started the existing `spark-check-training/retrain` deployment on that server.

Mount [Grafana provisioning](grafana/provisioning) under Grafana's provisioning
directory and [dashboard JSON](grafana/dashboards) at
`MONITORING_DASHBOARDS_PATH` inside the Grafana container. Set these values in the
Grafana environment.

| Variable | Value |
| --- | --- |
| `MONITORING_PROMETHEUS_URL` | Internal Prometheus URL, for example `http://prometheus:9090` |
| `MONITORING_POSTGRES_HOST` | PostgreSQL host and port, for example `postgres:5432` |
| `MONITORING_POSTGRES_DATABASE` | Database containing `monitoring_results` |
| `MONITORING_POSTGRES_USER` and `MONITORING_POSTGRES_PASSWORD` | Credentials for a role with SELECT on `monitoring_results` |
| `MONITORING_POSTGRES_SSLMODE` | TLS mode for the database connection |
| `MONITORING_DASHBOARDS_PATH` | Absolute path to the mounted dashboard directory, for example `/var/lib/grafana/dashboards` |
| `MONITORING_RETRAIN_WEBHOOK_URL` | Webhook endpoint, for example `http://monitoring-alerts:8010/alerts/drift` |
| `MONITORING_ALERT_PASSWORD` | Shared webhook password set in both Grafana and the webhook service environments |

Grafana checks the most recent window per model once a minute. It fires when a
fresh window has `drift_detected = true`. The webhook schedules the training
deployment, using the Grafana alert fingerprint and start time as the Prefect
idempotency key so retries of the same alert do not start another run. See
[Grafana provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)
and [PostgreSQL alert queries](https://grafana.com/docs/grafana/latest/datasources/postgres/alerting/).

## Verify drift and retraining

- Confirm the dashboard reads both data sources. Send API traffic and check the
  request and error panels. Run the monitoring job and check the drift and
  prediction panels for the right model and timeframe.
- Serve the training deployment in a test Prefect environment. Use the team's
  normal traffic scenario, collect enough requests in a full window, and check
  that the drift alert stays clear. Then repeat with the simulator's `--drift`
  option and check that the window is flagged, Grafana fires, and one Prefect run
  appears. If the simulator's normal distribution differs from the approved
  reference, agree on a suitable demo reference with the simulator owner.
- Keep model performance unavailable for demo traffic without observed scores.
  Capture normal and drift dashboard screenshots for the project README.
- For automated verification, install `requirements-dev.txt` and
  `monitoring/requirements.txt` in the test environment. Set
  `MONITORING_TEST_DATABASE_URL` to a disposable PostgreSQL database and run
  `python -m pytest`. The PostgreSQL test creates and rolls back its own schema.

The local verification used a synthetic reference and placeholder-model API
traffic. Normal traffic showed 0% drift. The shifted window showed 60% drift,
fired the Grafana alert, and completed one run of the existing Prefect training
flow. Replaying the notification left the run count at one. Both dashboard
captures show model performance as unavailable because these prediction rows
have no observed outcomes. [Normal traffic](screenshots/normal.png) ·
[Injected drift](screenshots/drift.png)
