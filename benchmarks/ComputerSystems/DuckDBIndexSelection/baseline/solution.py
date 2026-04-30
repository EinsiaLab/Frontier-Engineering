from __future__ import annotations


def select_indexes(workload_manifest):
    max_indexes = int(workload_manifest.get("max_indexes", 2))
    priority_value = str(workload_manifest.get("priority_value", "1-URGENT"))
    order_sample = int(workload_manifest.get("order_sample", 0))
    customer_sample = int(workload_manifest.get("customer_sample", 0))
    choices = []
    if customer_sample >= order_sample:
        choices.append("idx_orders_cust")
    if order_sample >= customer_sample:
        choices.append("idx_lineitem_order")
    if priority_value in {"1-URGENT", "2-HIGH"}:
        choices.append("idx_orders_priority")
    if "1998" in str(workload_manifest.get("min_order_date", "")):
        choices.append("idx_orders_date")
    if max_indexes >= 3:
        choices.append("idx_customer_segment")
    out = []
    for name in choices:
        if name not in out:
            out.append(name)
    return out[:max_indexes]
