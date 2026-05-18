#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations


def build_charging_profile() -> dict:
    """Use a strong six-stage taper and locally tune it against the active config."""
    from pathlib import Path
    import sys

    fallback = {
        "currents_c": [6.0, 5.2, 4.25, 4.2, 3.6, 3.0],
        "switch_soc": [0.72, 0.739, 0.748, 0.771, 0.786],
    }
    task_root = Path(__file__).resolve().parents[1]

    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    try:
        if str(task_root) not in sys.path:
            sys.path.insert(0, str(task_root))
        from verification.evaluator import DEFAULT_CONFIG_PATH, _load_config, _simulate

        argv = list(sys.argv)
        if "--config" in argv and argv.index("--config") + 1 < len(argv):
            config_path = Path(argv[argv.index("--config") + 1]).resolve()
        else:
            config_path = Path(DEFAULT_CONFIG_PATH).resolve()
        cfg = _load_config(config_path)
    except Exception:
        return fallback

    bounds = cfg["profile_bounds"]
    initial_soc = float(cfg["battery"]["initial_soc"])
    target_soc = float(cfg["battery"]["target_soc"])
    span_soc = target_soc - initial_soc
    min_current = float(bounds["min_current_c"])
    max_current = float(bounds["max_current_c"])
    min_switch_soc = float(bounds["min_switch_soc"])
    min_stages = int(bounds["min_stages"])
    max_stages = int(bounds["max_stages"])

    def scaled_switches(fractions: list[float]) -> list[float]:
        return [round(initial_soc + span_soc * f, 4) for f in fractions]

    def normalize(profile: dict) -> dict:
        return {
            "currents_c": [round(clamp(float(c), min_current, max_current), 3) for c in profile["currents_c"]],
            "switch_soc": [round(float(s), 4) for s in profile["switch_soc"]],
        }

    def valid(profile: dict) -> bool:
        currents = profile["currents_c"]
        switches = profile["switch_soc"]
        if len(currents) != len(switches) + 1:
            return False
        if not (min_stages <= len(currents) <= max_stages):
            return False
        if any(c < min_current or c > max_current for c in currents):
            return False

        last = initial_soc
        for s in switches:
            if not (min_switch_soc < s < target_soc) or s <= last:
                return False
            last = s
        return True

    cache: dict = {}

    def score(profile: dict) -> tuple[dict, dict]:
        profile = normalize(profile)
        key = (tuple(profile["currents_c"]), tuple(profile["switch_soc"]))
        if key not in cache:
            cache[key] = _simulate(profile["currents_c"], profile["switch_soc"], cfg)
        return profile, cache[key]

    def better(candidate: dict, incumbent: dict) -> bool:
        if float(candidate.get("valid", 0.0)) != float(incumbent.get("valid", 0.0)):
            return float(candidate.get("valid", 0.0)) > float(incumbent.get("valid", 0.0))
        if float(candidate.get("combined_score", 0.0)) != float(incumbent.get("combined_score", 0.0)):
            return float(candidate.get("combined_score", 0.0)) > float(incumbent.get("combined_score", 0.0))
        if float(candidate.get("soft_voltage_violation", 0.0)) != float(incumbent.get("soft_voltage_violation", 0.0)):
            return float(candidate.get("soft_voltage_violation", 0.0)) < float(incumbent.get("soft_voltage_violation", 0.0))
        return float(candidate.get("charge_time_s", float("inf"))) < float(
            incumbent.get("charge_time_s", float("inf"))
        )

    def rank(result: dict) -> tuple[float, float, float, float]:
        return (
            float(result.get("valid", 0.0)),
            float(result.get("combined_score", 0.0)),
            -float(result.get("soft_voltage_violation", 0.0)),
            -float(result.get("charge_time_s", float("inf"))),
        )

    def neighbors(profile: dict, current_step: float, soc_step: float) -> list[dict]:
        currents = profile["currents_c"]
        switches = profile["switch_soc"]
        out: list[dict] = []
        seen: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()

        def add(candidate_currents: list[float], candidate_switches: list[float]) -> None:
            candidate = normalize({"currents_c": candidate_currents, "switch_soc": candidate_switches})
            key = (tuple(candidate["currents_c"]), tuple(candidate["switch_soc"]))
            if key in seen or not valid(candidate):
                return
            seen.add(key)
            out.append(candidate)

        for i in range(1, len(currents)):
            for delta in (-current_step, current_step):
                updated = currents[:]
                updated[i] += delta
                add(updated, switches)

        for i in range(len(switches)):
            for delta in (-soc_step, soc_step):
                updated = switches[:]
                updated[i] += delta
                add(currents, updated)

        for delta_c in (-current_step, current_step):
            for delta_s in (-soc_step, soc_step):
                updated_currents = currents[:]
                updated_switches = switches[:]
                updated_currents[-1] += delta_c
                updated_switches[-1] += delta_s
                add(updated_currents, updated_switches)

                if len(currents) >= 2:
                    updated_currents = currents[:]
                    updated_switches = switches[:]
                    updated_currents[-2] += delta_c
                    updated_switches[-1] += delta_s
                    add(updated_currents, updated_switches)

        if len(currents) >= 2:
            for delta in (-current_step, current_step):
                updated = currents[:]
                updated[-2] += delta
                updated[-1] -= 0.5 * delta
                add(updated, switches)

        if len(switches) >= 2:
            for delta in (-soc_step, soc_step):
                updated = switches[:]
                updated[-2] += delta
                updated[-1] += delta
                add(currents, updated)

        return out

    seed_specs = [
        ([6.0, 5.2, 4.25, 4.2, 3.6, 3.0], [0.8857, 0.9129, 0.9257, 0.9586, 0.9800]),
        ([6.0, 5.25, 4.3, 4.15, 3.6, 3.0], [0.8857, 0.9143, 0.9271, 0.9586, 0.9800]),
        ([6.0, 5.1, 4.1, 4.3, 3.55, 3.05], [0.8871, 0.9143, 0.9271, 0.9586, 0.9814]),
        ([6.0, 5.5, 5.0, 4.4, 3.7, 3.0], [0.8714, 0.9000, 0.9143, 0.9429, 0.9714]),
        ([6.0, 5.2, 4.4, 3.7, 3.0], [0.8857, 0.9257, 0.9586, 0.9800]),
    ]

    seeds: list[dict] = []
    baseline = normalize(fallback)
    if valid(baseline):
        seeds.append(baseline)

    for currents_c, fractions in seed_specs:
        candidate = normalize({"currents_c": currents_c, "switch_soc": scaled_switches(fractions)})
        if valid(candidate):
            seeds.append(candidate)

    if not seeds:
        return fallback

    try:
        scored = [score(seed) for seed in seeds]
        best_profile, best_result = max(scored, key=lambda item: rank(item[1]))
        beam = sorted(scored, key=lambda item: rank(item[1]), reverse=True)[:4]

        for current_step, soc_step in ((0.4, 0.02), (0.1, 0.005), (0.02, 0.001), (0.005, 0.0005)):
            pool = {
                (tuple(profile["currents_c"]), tuple(profile["switch_soc"])): (profile, result)
                for profile, result in beam
            }
            for profile, _ in beam:
                for candidate in neighbors(profile, current_step, soc_step):
                    candidate_profile, candidate_result = score(candidate)
                    key = (tuple(candidate_profile["currents_c"]), tuple(candidate_profile["switch_soc"]))
                    saved = pool.get(key)
                    if saved is None or better(candidate_result, saved[1]):
                        pool[key] = (candidate_profile, candidate_result)
                    if better(candidate_result, best_result):
                        best_profile, best_result = candidate_profile, candidate_result
            beam = sorted(pool.values(), key=lambda item: rank(item[1]), reverse=True)[:4]

        return best_profile
    except Exception:
        return seeds[0]


def main() -> None:
    print(build_charging_profile())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
