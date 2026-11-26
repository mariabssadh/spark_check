import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from api.db import CREATE_TABLE
from monitoring.storage import (
    Result,
    Window,
    prepare_storage,
    read_predictions,
    write_result,
)


def test_database_window_join_and_upsert():
    database_url = os.getenv("MONITORING_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip(
            "set MONITORING_TEST_DATABASE_URL to a disposable PostgreSQL database"
        )
    psycopg = pytest.importorskip("psycopg")
    from psycopg import sql
    from psycopg.rows import dict_row

    end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    window = Window("test-model", "reference-v1", end - timedelta(hours=1), end)
    with (
        psycopg.connect(database_url, row_factory=dict_row) as conn,
        conn.transaction(force_rollback=True),
    ):
        schema = sql.Identifier(f"monitoring_test_{uuid4().hex}")
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(schema))
        conn.execute(CREATE_TABLE)
        prepare_storage(conn)

        for version, when in (
            (window.model_version, window.start),
            (window.model_version, end - timedelta(minutes=1)),
            (window.model_version, end),
            ("other-model", window.start),
        ):
            conn.execute(
                "INSERT INTO predictions (model_version, created_at, predicted_phq9_post) "
                "VALUES (%s, %s, 8)",
                (version, when),
            )
        predictions = read_predictions(conn, window)
        assert len(predictions) == 2
        assert all(row["observed_phq9_post"] is None for row in predictions)
        conn.execute(
            "INSERT INTO monitoring_outcomes VALUES (%s, 10)",
            (predictions[0]["id"],),
        )
        assert read_predictions(conn, window)[0]["observed_phq9_post"] == 10

        write_result(conn, window, Result(2, 30, 0, None, None, {}, None, None))
        write_result(conn, window, Result(2, 30, 1, 0.5, True, {}, 2.0, 2.0))
        results = conn.execute("SELECT * FROM monitoring_results").fetchall()
        assert len(results) == 1
        assert results[0]["mae"] == 2.0
