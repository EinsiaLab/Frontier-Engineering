import json
import math
import sys
from pathlib import Path


EXPECTED_NUM_RINGS = 20
RING_WIDTH_MM = 10.0


def _load_constants(task_dir: Path) -> dict:
    return json.loads((task_dir / "reference" / "constants.json").read_text(encoding="utf-8"))


def _fail(message: str) -> dict:
    return {"status": "failed", "message": message}


def _normalize_rings(data: list[dict]) -> list[dict] | dict:
    ring_ids: list[int] = []
    for idx, ring in enumerate(data):
        if not isinstance(ring, dict):
            return _fail(f"Ring {idx} must be a JSON object.")
        if "ring_id" not in ring:
            return _fail(f"Ring {idx} is missing required key 'ring_id'.")
        ring_id = ring["ring_id"]
        if not isinstance(ring_id, int):
            return _fail(f"Ring {idx} has non-integer ring_id={ring_id!r}.")
        ring_ids.append(ring_id)

    expected = list(range(EXPECTED_NUM_RINGS))
    if sorted(ring_ids) != expected:
        return _fail("ring_id values must be unique and cover exactly 0..19.")

    data_by_id = {ring["ring_id"]: ring for ring in data}
    return [data_by_id[i] for i in expected]


def evaluate(solution_path: Path) -> dict:
    task_dir = Path(__file__).resolve().parents[1]
    constants = _load_constants(task_dir)

    if not solution_path.exists():
        return _fail(f"Solution file not found: {solution_path}")

    try:
        data = json.loads(solution_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _fail(f"Failed to parse JSON: {exc}")

    if not isinstance(data, list):
        return _fail("JSON must be a list of ring dictionaries.")
    if len(data) != EXPECTED_NUM_RINGS:
        return _fail(f"Expected exactly {EXPECTED_NUM_RINGS} rings, got {len(data)}.")

    normalized = _normalize_rings(data)
    if isinstance(normalized, dict):
        return normalized

    search_space = constants["search_space"]
    physics = constants["physics"]
    budget = constants["budget"]
    scoring = constants["scoring"]

    total_volume = 0.0
    total_sensitivity = 0.0
    total_resolution_gamma = 0.0

    for ring in normalized:
        try:
            radius = float(ring["R"])
            thickness = float(ring["H"])
            width = float(ring["W"])
        except Exception:
            return _fail(f"Ring {ring['ring_id']} must contain finite numeric R/H/W.")

        if not (math.isfinite(radius) and math.isfinite(thickness) and math.isfinite(width)):
            return _fail(f"Ring {ring['ring_id']} contains non-finite geometry values.")
        if not (search_space["ring_radius"]["min"] <= radius <= search_space["ring_radius"]["max"]):
            return _fail(f"Ring {ring['ring_id']} has out-of-range R={radius}.")
        if not (
            search_space["crystal_thickness"]["min"]
            <= thickness
            <= search_space["crystal_thickness"]["max"]
        ):
            return _fail(f"Ring {ring['ring_id']} has out-of-range H={thickness}.")
        if not (search_space["crystal_width"]["min"] <= width <= search_space["crystal_width"]["max"]):
            return _fail(f"Ring {ring['ring_id']} has out-of-range W={width}.")

        total_volume += math.pi * (((radius + thickness) ** 2) - radius**2) * RING_WIDTH_MM

        z_pos = (ring["ring_id"] - (EXPECTED_NUM_RINGS - 1) / 2.0) * RING_WIDTH_MM
        distance = math.sqrt(radius**2 + z_pos**2)
        solid_angle_factor = RING_WIDTH_MM / distance
        stopping_power = (
            1.0 - math.exp(-physics["lyso_attenuation_coefficient_mm_inv"] * thickness)
        ) ** 2
        total_sensitivity += solid_angle_factor * stopping_power

        gamma = math.sqrt(width**2 + (physics["doi_parallax_factor"] * thickness / radius) ** 2)
        total_resolution_gamma += gamma

    avg_resolution_gamma = total_resolution_gamma / EXPECTED_NUM_RINGS
    cost_penalty = max(
        0.0,
        (total_volume - budget["max_lyso_volume_mm3"]) * budget["volume_penalty_rate"],
    )

    sensitivity_score = total_sensitivity * scoring["sensitivity_weight"]
    resolution_penalty = avg_resolution_gamma * scoring["resolution_penalty_weight"]
    total_score = sensitivity_score - resolution_penalty - cost_penalty

    return {
        "status": "success",
        "score": total_score,
        "metrics": {
            "volume_mm3": total_volume,
            "sensitivity_factor": total_sensitivity,
            "resolution_gamma": avg_resolution_gamma,
            "cost_penalty": cost_penalty,
        },
    }


if __name__ == "__main__":
    target_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("solution.json")
    print(json.dumps(evaluate(target_file)))
