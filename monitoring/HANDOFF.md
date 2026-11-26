# Spark Check monitoring handoff

Job 6 monitors the INSY 684 prediction API, detects input drift, and requests
retraining through the existing Prefect deployment. Use the
[monitoring runbook](README.md) for setup, configuration, and verification.

## Branch order

1. [PR #13](https://github.com/mariabssadh/spark_check/pull/13) adds Prometheus
   instrumentation and a local scrape config on `feat/prometheus-monitoring`.
2. `feat/monitoring-postgres` reads prediction windows and stores results.
3. `feat/evidently-monitoring` adds the window evaluator and monitoring image.
4. `feat/grafana-dashboards` provisions both data sources and the dashboard.
5. `feat/grafana-drift-alerts` provisions the drift rule and calls Prefect.

The remaining branches are stacked locally in this order. Review each feature
against the branch directly below it in the stack. Check the status of PR #13
before publishing more branches.

## Integration ownership

- **Compose owner** wires the monitoring images, database environment, Prometheus
  scrape target, and Grafana mounts into the shared stack. Use the API's internal
  service name and port in the Prometheus config and keep the job name
  `spark-check-api`.
- **Training and simulator owners** agree on the model version, reference CSV,
  and normal and drift traffic profiles. The training owner serves the existing
  `spark-check-training/retrain` deployment. Confirm what data it will use.
- **Outcome provider** defines how observed end-of-term PHQ-9 outcomes can be
  matched to predictions. MAE and RMSE remain unavailable until those pairs exist.

Run normal and drift traffic in the shared stack, confirm that exactly one
retraining run is created for the drift alert, and capture screenshots for the
project README.
