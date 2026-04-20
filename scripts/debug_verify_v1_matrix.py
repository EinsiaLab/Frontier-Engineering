# Temporary v1 matrix verification; writes NDJSON to debug-e710d8.log (debug session e710d8).
# Run from repo root: python scripts/debug_verify_v1_matrix.py
# region agent log
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

LOG_PATH = ROOT / "debug-e710d8.log"
V1 = ROOT / "frontier_eval" / "conf" / "batch" / "v1.yaml"


def _log(hypothesis_id: str, message: str, data: dict | None = None) -> None:
    payload = {
        "sessionId": "e710d8",
        "timestamp": int(time.time() * 1000),
        "hypothesisId": hypothesis_id,
        "location": "scripts/debug_verify_v1_matrix.py",
        "message": message,
        "data": data or {},
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# endregion


def main() -> None:
    # H1: YAML loads and resolves without OPENAI_* (defaults in oc.env)
    try:
        cfg = OmegaConf.load(str(V1))
        resolved = OmegaConf.to_container(cfg, resolve=True)
        _log("H1", "OmegaConf load+resolve ok", {"keys": list(resolved.keys()) if isinstance(resolved, dict) else None})
    except Exception as e:
        _log("H1", "OmegaConf resolve FAILED", {"error": str(e)})
        raise

    # H2: _parse_tasks succeeds, no duplicate labels
    from frontier_eval.batch import _parse_tasks

    raw_tasks = resolved.get("tasks") if isinstance(resolved, dict) else None
    try:
        specs = _parse_tasks(list(raw_tasks))
        labels = [s.label for s in specs]
        dup = [x for x in labels if labels.count(x) > 1]
        _log("H2", "parse_tasks", {"count": len(specs), "duplicate_labels": sorted(set(dup))})
    except Exception as e:
        _log("H2", "_parse_tasks FAILED", {"error": str(e)})
        raise

    # H3: validate.sh run_cpu_batch exclude list ⊆ matrix labels
    exclude_validate = [
        "Robotics/QuadrupedGaitOptimization",
        "Robotics/RobotArmCycleTimeOptimization",
        "Aerodynamics/CarAerodynamicsSensing",
        "KernelEngineering/MLA",
        "KernelEngineering/TriMul",
        "KernelEngineering/FlashAttention",
        "engdesign",
    ]
    mset = set(labels)
    unknown_ex = [x for x in exclude_validate if x not in mset]
    _log("H3", "validate exclude-tasks ⊆ matrix", {"unknown_exclude": unknown_ex})

    # H4: validate.sh GPU --tasks ⊆ matrix
    gpu_pick = [
        "Robotics/QuadrupedGaitOptimization",
        "Robotics/RobotArmCycleTimeOptimization",
        "Aerodynamics/CarAerodynamicsSensing",
    ]
    unknown_gpu = [x for x in gpu_pick if x not in mset]
    _log("H4", "validate gpu --tasks ⊆ matrix", {"unknown_gpu": unknown_gpu})

    # H5: simulated CPU batch selection non-empty
    remain = [t for t in specs if t.label not in set(exclude_validate)]
    _log("H5", "cpu simulation after exclude", {"remaining_count": len(remain)})

    # H6: file text still references OPENAI_MODEL (OmegaConf may resolve llms[].model on access)
    text = V1.read_text(encoding="utf-8")
    _log(
        "H6",
        "v1.yaml llm model line",
        {
            "has_openai_model_env_in_file": "OPENAI_MODEL" in text,
            "has_oc_env_model": "oc.env:OPENAI_MODEL" in text.replace(" ", ""),
        },
    )


if __name__ == "__main__":
    main()
