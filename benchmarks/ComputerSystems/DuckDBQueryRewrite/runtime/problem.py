from __future__ import annotations

try:
    from .duckdb_local_workload import measure_query_rewrite
except ImportError:
    from benchmarks.ComputerSystems.duckdb_local_workload import measure_query_rewrite


PUBLIC_CASES = (
    {
        "case_id": "public_quarter_join",
        "query_id": "quarter_join",
        "baseline_sql": """
WITH revenue AS (
  SELECT date_trunc('quarter', o.o_orderdate) AS quarter_bucket,
         c.c_mktsegment AS segment,
         sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue
  FROM customer c
  JOIN orders o ON o.o_custkey = c.c_custkey
  JOIN lineitem l ON l.l_orderkey = o.o_orderkey
  WHERE c.c_mktsegment IN ('BUILDING', 'AUTOMOBILE', 'HOUSEHOLD')
  GROUP BY 1, 2
),
order_counts AS (
  SELECT date_trunc('quarter', o.o_orderdate) AS quarter_bucket,
         c.c_mktsegment AS segment,
         count(DISTINCT o.o_orderkey) AS order_count
  FROM customer c
  JOIN orders o ON o.o_custkey = c.c_custkey
  JOIN lineitem l ON l.l_orderkey = o.o_orderkey
  WHERE c.c_mktsegment IN ('BUILDING', 'AUTOMOBILE', 'HOUSEHOLD')
  GROUP BY 1, 2
)
SELECT r.quarter_bucket, r.segment, r.revenue, o.order_count
FROM revenue r
JOIN order_counts o USING (quarter_bucket, segment)
ORDER BY quarter_bucket, segment
""".strip(),
        "repetitions": 3,
    },
    {
        "case_id": "public_shipmode_month",
        "query_id": "shipmode_month",
        "baseline_sql": """
WITH revenue AS (
  SELECT date_trunc('month', l_shipdate) AS month_bucket,
         l_shipmode AS shipmode,
         sum(l_extendedprice * (1 - l_discount)) AS revenue
  FROM lineitem
  WHERE l_shipdate >= DATE '1997-01-01'
  GROUP BY 1, 2
),
counts AS (
  SELECT date_trunc('month', l_shipdate) AS month_bucket,
         l_shipmode AS shipmode,
         count(*) AS line_count
  FROM lineitem
  WHERE l_shipdate >= DATE '1997-01-01'
  GROUP BY 1, 2
)
SELECT r.month_bucket, r.shipmode, r.revenue, c.line_count
FROM revenue r
JOIN counts c USING (month_bucket, shipmode)
ORDER BY month_bucket, shipmode
""".strip(),
        "repetitions": 3,
    },
    {
        "case_id": "public_customer_year",
        "query_id": "customer_year",
        "baseline_sql": """
WITH rev AS (
  SELECT year(o.o_orderdate) AS revenue_year,
         c.c_custkey AS customer_key,
         sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue
  FROM customer c
  JOIN orders o ON o.o_custkey = c.c_custkey
  JOIN lineitem l ON l.l_orderkey = o.o_orderkey
  GROUP BY 1, 2
),
orders_seen AS (
  SELECT year(o.o_orderdate) AS revenue_year,
         c.c_custkey AS customer_key,
         count(DISTINCT o.o_orderkey) AS order_count
  FROM customer c
  JOIN orders o ON o.o_custkey = c.c_custkey
  JOIN lineitem l ON l.l_orderkey = o.o_orderkey
  GROUP BY 1, 2
)
SELECT rev.revenue_year, rev.customer_key, rev.revenue, orders_seen.order_count
FROM rev
JOIN orders_seen USING (revenue_year, customer_key)
WHERE rev.revenue_year = 1998
ORDER BY rev.revenue DESC, rev.customer_key
LIMIT 80
""".strip(),
        "repetitions": 2,
    },
)

HIDDEN_CASES = (
    {
        "case_id": "hidden_quarter_join_recent",
        "query_id": "quarter_join_recent",
        "baseline_sql": """
WITH revenue AS (
  SELECT date_trunc('quarter', o.o_orderdate) AS quarter_bucket,
         c.c_mktsegment AS segment,
         sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue
  FROM customer c
  JOIN orders o ON o.o_custkey = c.c_custkey
  JOIN lineitem l ON l.l_orderkey = o.o_orderkey
  WHERE c.c_mktsegment IN ('BUILDING', 'AUTOMOBILE') AND o.o_orderdate >= DATE '1997-01-01'
  GROUP BY 1, 2
),
order_counts AS (
  SELECT date_trunc('quarter', o.o_orderdate) AS quarter_bucket,
         c.c_mktsegment AS segment,
         count(DISTINCT o.o_orderkey) AS order_count
  FROM customer c
  JOIN orders o ON o.o_custkey = c.c_custkey
  JOIN lineitem l ON l.l_orderkey = o.o_orderkey
  WHERE c.c_mktsegment IN ('BUILDING', 'AUTOMOBILE') AND o.o_orderdate >= DATE '1997-01-01'
  GROUP BY 1, 2
)
SELECT r.quarter_bucket, r.segment, r.revenue, o.order_count
FROM revenue r
JOIN order_counts o USING (quarter_bucket, segment)
ORDER BY quarter_bucket, segment
""".strip(),
        "repetitions": 2,
    },
    {
        "case_id": "hidden_shipmode_recent",
        "query_id": "shipmode_recent",
        "baseline_sql": """
WITH revenue AS (
  SELECT date_trunc('month', l_shipdate) AS month_bucket,
         l_shipmode AS shipmode,
         sum(l_extendedprice * (1 - l_discount)) AS revenue
  FROM lineitem
  WHERE l_shipdate >= DATE '1998-01-01'
  GROUP BY 1, 2
),
counts AS (
  SELECT date_trunc('month', l_shipdate) AS month_bucket,
         l_shipmode AS shipmode,
         count(*) AS line_count
  FROM lineitem
  WHERE l_shipdate >= DATE '1998-01-01'
  GROUP BY 1, 2
)
SELECT r.month_bucket, r.shipmode, r.revenue, c.line_count
FROM revenue r
JOIN counts c USING (month_bucket, shipmode)
ORDER BY month_bucket, shipmode
""".strip(),
        "repetitions": 3,
    },
    {
        "case_id": "hidden_customer_year_1997",
        "query_id": "customer_year_1997",
        "baseline_sql": """
WITH rev AS (
  SELECT year(o.o_orderdate) AS revenue_year,
         c.c_custkey AS customer_key,
         sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue
  FROM customer c
  JOIN orders o ON o.o_custkey = c.c_custkey
  JOIN lineitem l ON l.l_orderkey = o.o_orderkey
  GROUP BY 1, 2
),
orders_seen AS (
  SELECT year(o.o_orderdate) AS revenue_year,
         c.c_custkey AS customer_key,
         count(DISTINCT o.o_orderkey) AS order_count
  FROM customer c
  JOIN orders o ON o.o_custkey = c.c_custkey
  JOIN lineitem l ON l.l_orderkey = o.o_orderkey
  GROUP BY 1, 2
)
SELECT rev.revenue_year, rev.customer_key, rev.revenue, orders_seen.order_count
FROM rev
JOIN orders_seen USING (revenue_year, customer_key)
WHERE rev.revenue_year = 1997
ORDER BY rev.revenue DESC, rev.customer_key
LIMIT 60
""".strip(),
        "repetitions": 2,
    },
    {
        "case_id": "hidden_segment_rollup",
        "query_id": "segment_rollup",
        "baseline_sql": """
WITH revenue AS (
  SELECT c.c_mktsegment AS segment,
         sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue
  FROM customer c
  JOIN orders o ON o.o_custkey = c.c_custkey
  JOIN lineitem l ON l.l_orderkey = o.o_orderkey
  WHERE o.o_orderdate >= DATE '1996-01-01'
  GROUP BY 1
),
counts AS (
  SELECT c.c_mktsegment AS segment,
         count(DISTINCT o.o_orderkey) AS order_count
  FROM customer c
  JOIN orders o ON o.o_custkey = c.c_custkey
  JOIN lineitem l ON l.l_orderkey = o.o_orderkey
  WHERE o.o_orderdate >= DATE '1996-01-01'
  GROUP BY 1
)
SELECT r.segment, r.revenue, c.order_count
FROM revenue r
JOIN counts c USING (segment)
ORDER BY r.segment
""".strip(),
        "repetitions": 2,
    },
    {
        "case_id": "hidden_priority_rollup",
        "query_id": "priority_rollup",
        "baseline_sql": """
WITH revenue AS (
  SELECT o.o_orderpriority AS priority_bucket,
         sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue
  FROM orders o
  JOIN lineitem l ON l.l_orderkey = o.o_orderkey
  WHERE o.o_orderdate >= DATE '1997-01-01'
  GROUP BY 1
),
counts AS (
  SELECT o.o_orderpriority AS priority_bucket,
         count(*) AS order_count
  FROM orders o
  JOIN lineitem l ON l.l_orderkey = o.o_orderkey
  WHERE o.o_orderdate >= DATE '1997-01-01'
  GROUP BY 1
)
SELECT r.priority_bucket, r.revenue, c.order_count
FROM revenue r
JOIN counts c USING (priority_bucket)
ORDER BY r.priority_bucket
""".strip(),
        "repetitions": 2,
    },
)

WORKLOAD_MANIFEST = dict(PUBLIC_CASES[0])
ORIGINAL_QUERY_SQL = WORKLOAD_MANIFEST["baseline_sql"]


def load_instance():
    return {"sql": ORIGINAL_QUERY_SQL, "manifest": dict(WORKLOAD_MANIFEST)}


def evaluate_query(value, manifest: dict | None = None):
    manifest = WORKLOAD_MANIFEST if manifest is None else dict(manifest)
    if isinstance(value, dict):
        if "sql" not in value:
            raise ValueError("missing sql")
        value = value["sql"]
    return measure_query_rewrite(str(value), manifest)
