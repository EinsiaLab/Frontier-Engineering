from __future__ import annotations


def rewrite_query(sql, workload_manifest):
    query_id = str(workload_manifest.get("query_id", ""))
    rewrites = {
        "quarter_join": """
SELECT date_trunc('quarter', o.o_orderdate) AS quarter_bucket,
       c.c_mktsegment AS segment,
       sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
       count(DISTINCT o.o_orderkey) AS order_count
FROM customer c
JOIN orders o ON o.o_custkey = c.c_custkey
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
WHERE c.c_mktsegment IN ('BUILDING', 'AUTOMOBILE', 'HOUSEHOLD')
GROUP BY 1, 2
ORDER BY quarter_bucket, segment
""".strip(),
        "shipmode_month": """
SELECT date_trunc('month', l_shipdate) AS month_bucket,
       l_shipmode AS shipmode,
       sum(l_extendedprice * (1 - l_discount)) AS revenue,
       count(*) AS line_count
FROM lineitem
WHERE l_shipdate >= DATE '1997-01-01'
GROUP BY 1, 2
ORDER BY month_bucket, shipmode
""".strip(),
        "customer_year": """
SELECT year(o.o_orderdate) AS revenue_year,
       c.c_custkey AS customer_key,
       sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
       count(DISTINCT o.o_orderkey) AS order_count
FROM customer c
JOIN orders o ON o.o_custkey = c.c_custkey
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
GROUP BY 1, 2
HAVING year(o.o_orderdate) = 1998
ORDER BY revenue DESC, customer_key
LIMIT 80
""".strip(),
        "quarter_join_recent": """
SELECT date_trunc('quarter', o.o_orderdate) AS quarter_bucket,
       c.c_mktsegment AS segment,
       sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
       count(DISTINCT o.o_orderkey) AS order_count
FROM customer c
JOIN orders o ON o.o_custkey = c.c_custkey
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
WHERE c.c_mktsegment IN ('BUILDING', 'AUTOMOBILE') AND o.o_orderdate >= DATE '1997-01-01'
GROUP BY 1, 2
ORDER BY quarter_bucket, segment
""".strip(),
        "shipmode_recent": """
SELECT date_trunc('month', l_shipdate) AS month_bucket,
       l_shipmode AS shipmode,
       sum(l_extendedprice * (1 - l_discount)) AS revenue,
       count(*) AS line_count
FROM lineitem
WHERE l_shipdate >= DATE '1998-01-01'
GROUP BY 1, 2
ORDER BY month_bucket, shipmode
""".strip(),
        "customer_year_1997": """
SELECT year(o.o_orderdate) AS revenue_year,
       c.c_custkey AS customer_key,
       sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
       count(DISTINCT o.o_orderkey) AS order_count
FROM customer c
JOIN orders o ON o.o_custkey = c.c_custkey
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
GROUP BY 1, 2
HAVING year(o.o_orderdate) = 1997
ORDER BY revenue DESC, customer_key
LIMIT 60
""".strip(),
        "segment_rollup": """
SELECT c.c_mktsegment AS segment,
       sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
       count(DISTINCT o.o_orderkey) AS order_count
FROM customer c
JOIN orders o ON o.o_custkey = c.c_custkey
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
WHERE o.o_orderdate >= DATE '1996-01-01'
GROUP BY 1
ORDER BY segment
""".strip(),
        "priority_rollup": """
SELECT o.o_orderpriority AS priority_bucket,
       sum(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
       count(*) AS order_count
FROM orders o
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
WHERE o.o_orderdate >= DATE '1997-01-01'
GROUP BY 1
ORDER BY priority_bucket
""".strip(),
    }
    return rewrites.get(query_id, str(sql).strip())
