from __future__ import annotations

try:
    from .duckdb_local_workload import PREAGGREGATION_CANDIDATES, measure_preaggregation_design, normalize_name_list
except ImportError:
    from benchmarks.ComputerSystems.duckdb_local_workload import PREAGGREGATION_CANDIDATES, measure_preaggregation_design, normalize_name_list


PUBLIC_CASES = (
    {
        "case_id": "public_all_reports",
        "candidate_preaggregations": tuple(sorted(PREAGGREGATION_CANDIDATES)),
        "max_preaggregations": 2,
        "segment_filter": ("BUILDING", "AUTOMOBILE", "HOUSEHOLD"),
        "min_shipdate": "1997-01-01",
        "revenue_year": 1998,
        "limit_rows": 100,
        "repetitions": 3,
    },
    {
        "case_id": "public_focus_segments",
        "candidate_preaggregations": tuple(sorted(PREAGGREGATION_CANDIDATES)),
        "max_preaggregations": 1,
        "segment_filter": ("BUILDING", "AUTOMOBILE"),
        "min_shipdate": "1998-01-01",
        "revenue_year": 1997,
        "limit_rows": 50,
        "repetitions": 3,
    },
    {
        "case_id": "public_long_horizon",
        "candidate_preaggregations": tuple(sorted(PREAGGREGATION_CANDIDATES)),
        "max_preaggregations": 2,
        "segment_filter": ("HOUSEHOLD", "FURNITURE", "MACHINERY"),
        "min_shipdate": "1996-01-01",
        "revenue_year": 1996,
        "limit_rows": 75,
        "repetitions": 2,
    },
)

HIDDEN_CASES = (
    {
        "case_id": "hidden_shipmode_recent",
        "candidate_preaggregations": tuple(sorted(PREAGGREGATION_CANDIDATES)),
        "max_preaggregations": 2,
        "segment_filter": ("BUILDING", "HOUSEHOLD"),
        "min_shipdate": "1998-06-01",
        "revenue_year": 1998,
        "limit_rows": 40,
        "repetitions": 3,
    },
    {
        "case_id": "hidden_segment_mix",
        "candidate_preaggregations": tuple(sorted(PREAGGREGATION_CANDIDATES)),
        "max_preaggregations": 2,
        "segment_filter": ("AUTOMOBILE", "FURNITURE"),
        "min_shipdate": "1997-04-01",
        "revenue_year": 1997,
        "limit_rows": 60,
        "repetitions": 2,
    },
    {
        "case_id": "hidden_customer_topn",
        "candidate_preaggregations": tuple(sorted(PREAGGREGATION_CANDIDATES)),
        "max_preaggregations": 1,
        "segment_filter": ("BUILDING", "AUTOMOBILE", "HOUSEHOLD"),
        "min_shipdate": "1997-01-01",
        "revenue_year": 1998,
        "limit_rows": 25,
        "repetitions": 3,
    },
    {
        "case_id": "hidden_wide_reports",
        "candidate_preaggregations": tuple(sorted(PREAGGREGATION_CANDIDATES)),
        "max_preaggregations": 3,
        "segment_filter": ("BUILDING", "AUTOMOBILE", "HOUSEHOLD", "FURNITURE"),
        "min_shipdate": "1995-01-01",
        "revenue_year": 1995,
        "limit_rows": 90,
        "repetitions": 2,
    },
    {
        "case_id": "hidden_narrow_reports",
        "candidate_preaggregations": tuple(sorted(PREAGGREGATION_CANDIDATES)),
        "max_preaggregations": 1,
        "segment_filter": ("MACHINERY",),
        "min_shipdate": "1998-01-01",
        "revenue_year": 1998,
        "limit_rows": 20,
        "repetitions": 3,
    },
)

WORKLOAD_MANIFEST = dict(PUBLIC_CASES[0])


def load_instance():
    return dict(WORKLOAD_MANIFEST)


def evaluate_selection(selection, manifest: dict | None = None):
    manifest = WORKLOAD_MANIFEST if manifest is None else dict(manifest)
    return measure_preaggregation_design(normalize_name_list(selection, "preaggregations"), manifest)
