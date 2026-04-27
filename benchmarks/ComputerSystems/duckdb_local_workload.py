from __future__ import annotations

import math
import time
from typing import Any

import duckdb


CUSTOMER_COUNT = 20_000
ORDER_COUNT = 120_000
LINEITEM_COUNT = 600_000

CUSTOMER_KEYS = tuple(1 + ((i * 97) % CUSTOMER_COUNT) for i in range(1, 301))
ORDER_KEYS = tuple(1 + ((i * 193) % ORDER_COUNT) for i in range(1, 301))


INDEX_CANDIDATES = {
    "idx_orders_cust": "CREATE INDEX idx_orders_cust ON orders(o_custkey)",
    "idx_orders_date": "CREATE INDEX idx_orders_date ON orders(o_orderdate)",
    "idx_lineitem_order": "CREATE INDEX idx_lineitem_order ON lineitem(l_orderkey)",
    "idx_customer_segment": "CREATE INDEX idx_customer_segment ON customer(c_mktsegment)",
    "idx_orders_priority": "CREATE INDEX idx_orders_priority ON orders(o_orderpriority)",
}

PREAGGREGATION_CANDIDATES = {
    "agg_quarter_segment_revenue": (
        "CREATE TABLE agg_quarter_segment_revenue AS "
        "SELECT date_trunc('quarter', o.o_orderdate) AS quarter_bucket, "
        "       c.c_mktsegment AS segment, "
        "       sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
        "FROM customer c "
        "JOIN orders o ON o.o_custkey = c.c_custkey "
        "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
        "GROUP BY 1, 2"
    ),
    "agg_month_shipmode_revenue": (
        "CREATE TABLE agg_month_shipmode_revenue AS "
        "SELECT date_trunc('month', l.l_shipdate) AS month_bucket, "
        "       l.l_shipmode AS shipmode, "
        "       sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
        "FROM lineitem l "
        "GROUP BY 1, 2"
    ),
    "agg_customer_year_revenue": (
        "CREATE TABLE agg_customer_year_revenue AS "
        "SELECT year(o.o_orderdate) AS revenue_year, "
        "       c.c_custkey, "
        "       sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
        "FROM customer c "
        "JOIN orders o ON o.o_custkey = c.c_custkey "
        "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
        "GROUP BY 1, 2"
    ),
    "agg_unused_priority_only": (
        "CREATE TABLE agg_unused_priority_only AS "
        "SELECT o.o_orderpriority, count(*) AS order_count "
        "FROM orders o "
        "GROUP BY 1"
    ),
}


def build_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=1")
    con.execute(
        f"""
        CREATE TABLE customer AS
        SELECT i AS c_custkey,
               'Customer #' || i AS c_name,
               CASE i % 5
                 WHEN 0 THEN 'BUILDING'
                 WHEN 1 THEN 'AUTOMOBILE'
                 WHEN 2 THEN 'HOUSEHOLD'
                 WHEN 3 THEN 'FURNITURE'
                 ELSE 'MACHINERY'
               END AS c_mktsegment,
               i % 25 AS c_nationkey
        FROM range(1, {CUSTOMER_COUNT + 1}) t(i)
        """
    )
    con.execute(
        f"""
        CREATE TABLE orders AS
        SELECT i AS o_orderkey,
               1 + ((i * 17) % {CUSTOMER_COUNT}) AS o_custkey,
               DATE '1995-01-01' + (((i * 13) % 1460) * INTERVAL 1 DAY) AS o_orderdate,
               100 + (((i * 37) % 100000) / 10.0) AS o_totalprice,
               CASE i % 5
                 WHEN 0 THEN '1-URGENT'
                 WHEN 1 THEN '2-HIGH'
                 WHEN 2 THEN '3-MEDIUM'
                 WHEN 3 THEN '4-NOT SPECIFIED'
                 ELSE '5-LOW'
               END AS o_orderpriority
        FROM range(1, {ORDER_COUNT + 1}) t(i)
        """
    )
    con.execute(
        f"""
        CREATE TABLE lineitem AS
        SELECT i AS l_lineitemkey,
               1 + ((i * 7) % {ORDER_COUNT}) AS l_orderkey,
               1 + ((i * 11) % 50000) AS l_partkey,
               1 + ((i * 13) % 10000) AS l_suppkey,
               1 + ((i * 5) % 50) AS l_quantity,
               10 + (((i * 19) % 100000) / 20.0) AS l_extendedprice,
               (((i * 3) % 10) / 100.0) AS l_discount,
               DATE '1995-01-01' + (((i * 29) % 1460) * INTERVAL 1 DAY) AS l_shipdate,
               CASE i % 5
                 WHEN 0 THEN 'AIR'
                 WHEN 1 THEN 'MAIL'
                 WHEN 2 THEN 'RAIL'
                 WHEN 3 THEN 'TRUCK'
                 ELSE 'SHIP'
               END AS l_shipmode
        FROM range(1, {LINEITEM_COUNT + 1}) t(i)
        """
    )
    return con


def normalize_name_list(value: Any, key: str) -> list[str]:
    if isinstance(value, dict):
        if key not in value:
            raise ValueError(f"missing {key}")
        value = value[key]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a list or tuple")
    out: list[str] = []
    seen = set()
    for item in value:
        name = str(item)
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def compare_results(lhs: list[tuple[Any, ...]], rhs: list[tuple[Any, ...]], tol: float = 1e-6) -> bool:
    if len(lhs) != len(rhs):
        return False
    for left_row, right_row in zip(lhs, rhs):
        if len(left_row) != len(right_row):
            return False
        for left_value, right_value in zip(left_row, right_row):
            if isinstance(left_value, float) or isinstance(right_value, float):
                if not math.isfinite(float(left_value)) or not math.isfinite(float(right_value)):
                    return False
                if abs(float(left_value) - float(right_value)) > tol:
                    return False
            elif left_value != right_value:
                return False
    return True


def _index_keys(sample_size: int, source: tuple[int, ...]) -> tuple[int, ...]:
    sample_size = max(1, min(len(source), int(sample_size)))
    return tuple(source[:sample_size])


def run_index_workload(con: duckdb.DuckDBPyConnection, manifest: dict[str, Any]) -> float:
    start_time = time.perf_counter()
    customer_keys = _index_keys(manifest.get("customer_sample", 80), CUSTOMER_KEYS)
    order_keys = _index_keys(manifest.get("order_sample", 80), ORDER_KEYS)
    urgent_customer_keys = _index_keys(manifest.get("urgent_customer_sample", 40), CUSTOMER_KEYS)
    min_order_date = str(manifest.get("min_order_date", "1997-01-01"))
    priority_value = str(manifest.get("priority_value", "1-URGENT"))

    for customer_key in customer_keys:
        con.execute(
            "SELECT sum(o_totalprice) "
            "FROM orders "
            "WHERE o_custkey = ? AND o_orderdate >= CAST(? AS DATE)",
            [customer_key, min_order_date],
        ).fetchone()
    for order_key in order_keys:
        con.execute(
            "SELECT sum(l_extendedprice * (1 - l_discount)) "
            "FROM lineitem "
            "WHERE l_orderkey = ?",
            [order_key],
        ).fetchone()
    for customer_key in urgent_customer_keys:
        con.execute(
            "SELECT count(*) "
            "FROM customer c "
            "JOIN orders o ON c.c_custkey = o.o_custkey "
            "WHERE c.c_custkey = ? AND o.o_orderpriority = ?",
            [customer_key, priority_value],
        ).fetchone()
    return time.perf_counter() - start_time


def measure_index_design(selected_indexes: list[str], manifest: dict[str, Any]) -> dict[str, float | int]:
    allowed = tuple(manifest.get("candidate_indexes", tuple(sorted(INDEX_CANDIDATES))))
    max_indexes = int(manifest.get("max_indexes", len(allowed)))
    unknown = [name for name in selected_indexes if name not in allowed]
    if unknown:
        raise ValueError(f"unknown index names: {unknown}")
    if len(selected_indexes) > max_indexes:
        raise ValueError(f"too many indexes selected: {len(selected_indexes)} > {max_indexes}")

    con = build_connection()
    start_setup = time.perf_counter()
    for name in selected_indexes:
        con.execute(INDEX_CANDIDATES[name])
    setup_runtime = time.perf_counter() - start_setup

    workload_runtime = 0.0
    repetitions = int(manifest.get("repetitions", 3))
    run_index_workload(con, manifest)
    for _ in range(repetitions):
        workload_runtime += run_index_workload(con, manifest)
    return {
        "setup_runtime_s": float(setup_runtime),
        "workload_runtime_s": float(workload_runtime),
        "total_runtime_s": float(setup_runtime + workload_runtime),
        "selected_index_count": len(selected_indexes),
    }


def _report_quarter_segment(con: duckdb.DuckDBPyConnection, use_aggregate: bool, segment_filter: tuple[str, ...]) -> list[tuple[Any, ...]]:
    values = ", ".join(f"'{value}'" for value in segment_filter)
    if use_aggregate:
        return con.execute(
            "SELECT quarter_bucket, segment, revenue "
            "FROM agg_quarter_segment_revenue "
            f"WHERE segment IN ({values}) "
            "ORDER BY quarter_bucket, segment"
        ).fetchall()
    return con.execute(
        "SELECT date_trunc('quarter', o.o_orderdate) AS quarter_bucket, "
        "       c.c_mktsegment AS segment, "
        "       sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
        "FROM customer c "
        "JOIN orders o ON o.o_custkey = c.c_custkey "
        "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
        f"WHERE c.c_mktsegment IN ({values}) "
        "GROUP BY 1, 2 "
        "ORDER BY quarter_bucket, segment"
    ).fetchall()


def _report_month_shipmode(con: duckdb.DuckDBPyConnection, use_aggregate: bool, min_shipdate: str) -> list[tuple[Any, ...]]:
    if use_aggregate:
        return con.execute(
            "SELECT month_bucket, shipmode, revenue "
            "FROM agg_month_shipmode_revenue "
            "WHERE month_bucket >= CAST(? AS DATE) "
            "ORDER BY month_bucket, shipmode",
            [min_shipdate],
        ).fetchall()
    return con.execute(
        "SELECT date_trunc('month', l.l_shipdate) AS month_bucket, "
        "       l.l_shipmode AS shipmode, "
        "       sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
        "FROM lineitem l "
        "WHERE l.l_shipdate >= CAST(? AS DATE) "
        "GROUP BY 1, 2 "
        "ORDER BY month_bucket, shipmode",
        [min_shipdate],
    ).fetchall()


def _report_customer_year(con: duckdb.DuckDBPyConnection, use_aggregate: bool, revenue_year: int, limit_rows: int) -> list[tuple[Any, ...]]:
    if use_aggregate:
        return con.execute(
            "SELECT revenue_year, c_custkey, revenue "
            "FROM agg_customer_year_revenue "
            "WHERE revenue_year = ? "
            "ORDER BY revenue DESC, c_custkey "
            "LIMIT ?",
            [revenue_year, limit_rows],
        ).fetchall()
    return con.execute(
        "SELECT year(o.o_orderdate) AS revenue_year, "
        "       c.c_custkey, "
        "       sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
        "FROM customer c "
        "JOIN orders o ON o.o_custkey = c.c_custkey "
        "JOIN lineitem l ON l.l_orderkey = o.o_orderkey "
        "GROUP BY 1, 2 "
        "HAVING year(o.o_orderdate) = ? "
        "ORDER BY revenue DESC, c.c_custkey "
        "LIMIT ?",
        [revenue_year, limit_rows],
    ).fetchall()


def _run_preaggregation_reports(
    con: duckdb.DuckDBPyConnection,
    selected: set[str],
    manifest: dict[str, Any],
) -> tuple[float, tuple[list[tuple[Any, ...]], ...]]:
    start_time = time.perf_counter()
    result_a = _report_quarter_segment(
        con,
        "agg_quarter_segment_revenue" in selected,
        tuple(manifest.get("segment_filter", ("BUILDING", "AUTOMOBILE", "HOUSEHOLD"))),
    )
    result_b = _report_month_shipmode(
        con,
        "agg_month_shipmode_revenue" in selected,
        str(manifest.get("min_shipdate", "1997-01-01")),
    )
    result_c = _report_customer_year(
        con,
        "agg_customer_year_revenue" in selected,
        int(manifest.get("revenue_year", 1998)),
        int(manifest.get("limit_rows", 100)),
    )
    runtime = time.perf_counter() - start_time
    return runtime, (result_a, result_b, result_c)


def measure_preaggregation_design(selected_preaggregations: list[str], manifest: dict[str, Any]) -> dict[str, float | int]:
    allowed = tuple(manifest.get("candidate_preaggregations", tuple(sorted(PREAGGREGATION_CANDIDATES))))
    max_preaggregations = int(manifest.get("max_preaggregations", len(allowed)))
    unknown = [name for name in selected_preaggregations if name not in allowed]
    if unknown:
        raise ValueError(f"unknown pre-aggregation names: {unknown}")
    if len(selected_preaggregations) > max_preaggregations:
        raise ValueError(
            f"too many pre-aggregations selected: {len(selected_preaggregations)} > {max_preaggregations}"
        )

    baseline_con = build_connection()
    candidate_con = build_connection()
    start_setup = time.perf_counter()
    for name in selected_preaggregations:
        candidate_con.execute(PREAGGREGATION_CANDIDATES[name])
    setup_runtime = time.perf_counter() - start_setup

    _, baseline_results = _run_preaggregation_reports(baseline_con, set(), manifest)
    _, candidate_results = _run_preaggregation_reports(candidate_con, set(selected_preaggregations), manifest)
    if any(not compare_results(left, right) for left, right in zip(candidate_results, baseline_results)):
        raise ValueError("candidate pre-aggregation selection changed the query results")

    repetitions = int(manifest.get("repetitions", 3))
    repeated_baseline_runtime = 0.0
    repeated_candidate_runtime = 0.0
    _run_preaggregation_reports(baseline_con, set(), manifest)
    _run_preaggregation_reports(candidate_con, set(selected_preaggregations), manifest)
    for _ in range(repetitions):
        extra_runtime, _ = _run_preaggregation_reports(baseline_con, set(), manifest)
        repeated_baseline_runtime += extra_runtime
        extra_runtime, _ = _run_preaggregation_reports(candidate_con, set(selected_preaggregations), manifest)
        repeated_candidate_runtime += extra_runtime

    return {
        "setup_runtime_s": float(setup_runtime),
        "candidate_workload_runtime_s": float(repeated_candidate_runtime),
        "candidate_total_runtime_s": float(setup_runtime + repeated_candidate_runtime),
        "baseline_total_runtime_s": float(repeated_baseline_runtime),
        "selected_preaggregation_count": len(selected_preaggregations),
    }


def measure_query_rewrite(sql: str, manifest: dict[str, Any]) -> dict[str, Any]:
    sql = str(sql).strip()
    if not sql:
        raise ValueError("query must not be empty")
    baseline_sql = str(manifest["baseline_sql"]).strip()
    repetitions = int(manifest.get("repetitions", 3))

    baseline_con = build_connection()
    candidate_con = build_connection()
    baseline_rows = baseline_con.execute(baseline_sql).fetchall()
    candidate_rows = candidate_con.execute(sql).fetchall()
    if not compare_results(candidate_rows, baseline_rows):
        raise ValueError("candidate query result does not match the baseline result")

    baseline_con.execute(baseline_sql).fetchall()
    baseline_start = time.perf_counter()
    for _ in range(repetitions):
        baseline_con.execute(baseline_sql).fetchall()
    baseline_runtime = time.perf_counter() - baseline_start

    candidate_con.execute(sql).fetchall()
    candidate_start = time.perf_counter()
    for _ in range(repetitions):
        candidate_rows = candidate_con.execute(sql).fetchall()
    candidate_runtime = time.perf_counter() - candidate_start

    return {
        "baseline_runtime_s": float(baseline_runtime),
        "candidate_runtime_s": float(candidate_runtime),
        "row_count": len(candidate_rows),
    }
