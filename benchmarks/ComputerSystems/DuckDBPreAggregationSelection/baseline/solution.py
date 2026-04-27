from __future__ import annotations


def select_preaggregations(workload_manifest):
    max_preaggregations = int(workload_manifest.get("max_preaggregations", 1))
    limit_rows = int(workload_manifest.get("limit_rows", 100))
    min_shipdate = str(workload_manifest.get("min_shipdate", "1997-01-01"))
    choices = ["agg_quarter_segment_revenue"]
    if limit_rows <= 60:
        choices.insert(0, "agg_customer_year_revenue")
    if "1998" in min_shipdate or max_preaggregations >= 2:
        choices.append("agg_month_shipmode_revenue")
    out = []
    for name in choices:
        if name not in out:
            out.append(name)
    return out[:max_preaggregations]
