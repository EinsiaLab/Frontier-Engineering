from __future__ import annotations

from benchmarks.ComputerSystems.duckdb_local_workload import INDEX_CANDIDATES, measure_index_design, normalize_name_list


PUBLIC_CASES = (
    {
        "case_id": "public_customer_join",
        "candidate_indexes": tuple(sorted(INDEX_CANDIDATES)),
        "max_indexes": 2,
        "customer_sample": 60,
        "order_sample": 50,
        "urgent_customer_sample": 30,
        "priority_value": "1-URGENT",
        "min_order_date": "1997-01-01",
        "repetitions": 3,
    },
    {
        "case_id": "public_order_lookup",
        "candidate_indexes": tuple(sorted(INDEX_CANDIDATES)),
        "max_indexes": 2,
        "customer_sample": 40,
        "order_sample": 80,
        "urgent_customer_sample": 20,
        "priority_value": "2-HIGH",
        "min_order_date": "1996-01-01",
        "repetitions": 3,
    },
    {
        "case_id": "public_priority_mix",
        "candidate_indexes": tuple(sorted(INDEX_CANDIDATES)),
        "max_indexes": 3,
        "customer_sample": 90,
        "order_sample": 40,
        "urgent_customer_sample": 50,
        "priority_value": "1-URGENT",
        "min_order_date": "1998-01-01",
        "repetitions": 2,
    },
)

HIDDEN_CASES = (
    {
        "case_id": "hidden_deep_history",
        "candidate_indexes": tuple(sorted(INDEX_CANDIDATES)),
        "max_indexes": 2,
        "customer_sample": 55,
        "order_sample": 70,
        "urgent_customer_sample": 35,
        "priority_value": "3-MEDIUM",
        "min_order_date": "1995-06-01",
        "repetitions": 3,
    },
    {
        "case_id": "hidden_recent_priority",
        "candidate_indexes": tuple(sorted(INDEX_CANDIDATES)),
        "max_indexes": 2,
        "customer_sample": 75,
        "order_sample": 60,
        "urgent_customer_sample": 45,
        "priority_value": "1-URGENT",
        "min_order_date": "1998-06-01",
        "repetitions": 2,
    },
    {
        "case_id": "hidden_lookup_heavy",
        "candidate_indexes": tuple(sorted(INDEX_CANDIDATES)),
        "max_indexes": 2,
        "customer_sample": 25,
        "order_sample": 120,
        "urgent_customer_sample": 20,
        "priority_value": "5-LOW",
        "min_order_date": "1997-01-01",
        "repetitions": 3,
    },
    {
        "case_id": "hidden_balanced",
        "candidate_indexes": tuple(sorted(INDEX_CANDIDATES)),
        "max_indexes": 3,
        "customer_sample": 70,
        "order_sample": 70,
        "urgent_customer_sample": 40,
        "priority_value": "2-HIGH",
        "min_order_date": "1996-07-01",
        "repetitions": 2,
    },
    {
        "case_id": "hidden_customer_focus",
        "candidate_indexes": tuple(sorted(INDEX_CANDIDATES)),
        "max_indexes": 2,
        "customer_sample": 110,
        "order_sample": 35,
        "urgent_customer_sample": 60,
        "priority_value": "1-URGENT",
        "min_order_date": "1997-04-01",
        "repetitions": 2,
    },
)

WORKLOAD_MANIFEST = dict(PUBLIC_CASES[0])


def load_instance():
    return dict(WORKLOAD_MANIFEST)


def evaluate_selection(selection, manifest: dict | None = None):
    manifest = WORKLOAD_MANIFEST if manifest is None else dict(manifest)
    return measure_index_design(normalize_name_list(selection, "indexes"), manifest)
