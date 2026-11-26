"""Evaluate logged predictions in repeatable windows with Evidently."""

import argparse
import logging
import os
import time
from datetime import datetime, timedelta, timezone

from monitoring.storage import (
    FEATURES,
    Result,
    Window,
    prepare_storage,
    read_predictions,
    write_result,
)

PREDICTION = "predicted_phq9_post"
OUTCOME = "observed_phq9_post"
LOG = logging.getLogger(__name__)


def evaluate(predictions, reference, min_rows: int, drift_threshold: float) -> Result:
    import pandas as pd
    from evidently import DataDefinition, Dataset, Report
    from evidently.core.datasets import Regression
    from evidently.metrics import MAE, RMSE
    from evidently.presets import DataDriftPreset

    current = pd.DataFrame(
        predictions, columns=[*FEATURES, PREDICTION, OUTCOME], dtype=float
    )
    clean_inputs = current.loc[:, list(FEATURES)].dropna()
    share = None
    feature_drift = {}
    if min(len(reference), len(clean_inputs)) >= min_rows:
        drift = Report([DataDriftPreset(drift_share=drift_threshold)]).run(
            clean_inputs, reference
        )
        for metric in drift.dict()["metrics"]:
            config = metric["config"]
            metric_type = config["type"].rsplit(":", 1)[-1]
            if metric_type == "DriftedColumnsCount":
                share = float(metric["value"]["share"])
            elif metric_type == "ValueDrift":
                feature_drift[config["column"]] = {
                    "score": float(metric["value"]),
                    "method": config["method"],
                    "threshold": config["threshold"],
                }

    labeled = current[[PREDICTION, OUTCOME]].dropna()
    mae = rmse = None
    if not labeled.empty:
        data = Dataset.from_pandas(
            labeled,
            data_definition=DataDefinition(
                regression=[Regression(target=OUTCOME, prediction=PREDICTION)]
            ),
        )
        errors = Report([MAE(error_distr=False), RMSE(error_distr=False)]).run(data)
        mae_metric, rmse_metric = errors.dict()["metrics"]
        mae = float(mae_metric["value"]["mean"])
        rmse = float(rmse_metric["value"])

    return Result(
        prediction_count=len(current),
        reference_count=len(reference),
        labeled_count=len(labeled),
        drift_share=share,
        drift_detected=None if share is None else share >= drift_threshold,
        feature_drift=feature_drift,
        mae=mae,
        rmse=rmse,
    )


def run_window(database_url, window, reference, min_rows, drift_threshold):
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        predictions = read_predictions(conn, window)
    result = evaluate(predictions, reference, min_rows, drift_threshold)
    with psycopg.connect(database_url) as conn:
        write_result(conn, window, result)
    LOG.info(
        "Window ending %s for %s: %d predictions, %d labeled, drift=%s",
        window.end,
        window.model_version,
        result.prediction_count,
        result.labeled_count,
        result.drift_detected,
    )


def main():
    import pandas as pd
    import psycopg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-db", action="store_true", help="create monitoring tables"
    )
    parser.add_argument(
        "--model-version", default=os.getenv("MONITORING_MODEL_VERSION")
    )
    parser.add_argument("--reference", default=os.getenv("MONITORING_REFERENCE_CSV"))
    parser.add_argument("--reference-id", default=os.getenv("MONITORING_REFERENCE_ID"))
    parser.add_argument("--window-minutes", type=int, default=5)
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--drift-threshold", type=float, default=0.5)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--watch", action="store_true", help="evaluate each completed window"
    )
    mode.add_argument(
        "--window-end", type=datetime.fromisoformat, help="ISO 8601 timestamp"
    )
    args = parser.parse_args()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL is required")
    if args.init_db:
        with psycopg.connect(database_url) as conn:
            prepare_storage(conn)
        return

    if not all((args.model_version, args.reference, args.reference_id)):
        parser.error("model version, reference CSV, and reference ID are required")
    if (
        args.window_minutes < 1
        or args.min_rows < 2
        or not 0 < args.drift_threshold <= 1
    ):
        parser.error(
            "use window-minutes >= 1, min-rows >= 2, and 0 < drift-threshold <= 1"
        )
    if args.window_end and args.window_end.utcoffset() is None:
        parser.error("--window-end requires a timezone offset")

    reference = pd.read_csv(
        args.reference, usecols=list(FEATURES), dtype=float
    ).dropna()
    if len(reference) < args.min_rows:
        parser.error(f"reference needs at least {args.min_rows} complete rows")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    seconds = args.window_minutes * 60
    while True:
        boundary = int(time.time()) // seconds * seconds
        end = args.window_end or datetime.fromtimestamp(boundary, tz=timezone.utc)
        window = Window(
            args.model_version, args.reference_id, end - timedelta(seconds=seconds), end
        )
        try:
            run_window(
                database_url, window, reference, args.min_rows, args.drift_threshold
            )
        except Exception:
            if not args.watch:
                raise
            LOG.exception("Monitoring window failed")
        if not args.watch:
            return
        time.sleep(max(1, boundary + seconds - time.time()))


if __name__ == "__main__":
    main()
