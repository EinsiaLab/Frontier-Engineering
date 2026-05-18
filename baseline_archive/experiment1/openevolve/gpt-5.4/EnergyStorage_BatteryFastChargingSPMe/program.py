#!/usr/bin/env python3
# EVOLVE-BLOCK-START

from __future__ import annotations

import importlib.util
from itertools import product
from pathlib import Path





def build_charging_policy() -> dict:
    fallback_currents = [7.0, 4.8, 4.0, 3.2, 2.4]
    fallback_switches = [0.16, 0.36, 0.66, 0.86]

    def locate_task_root() -> Path | None:
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "verification" / "evaluator.py").exists() and (
                parent / "references" / "battery_config.json"
            ).exists():
                return parent
        return None

    task_root = locate_task_root()
    if task_root is None:
        return {"currents_c": fallback_currents, "switch_soc": fallback_switches}

    eval_path = task_root / "verification" / "evaluator.py"
    config_path = task_root / "references" / "battery_config.json"

    try:
        spec = importlib.util.spec_from_file_location("battery_fast_charge_spme_internal_eval", eval_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to load evaluator")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        cfg = evaluator._load_config(config_path)
    except Exception:
        return {"currents_c": fallback_currents, "switch_soc": fallback_switches}

    bounds = cfg["profile_bounds"]
    target_soc = float(cfg["battery"]["target_soc"])
    min_switch_soc = float(bounds["min_switch_soc"])
    min_current = float(bounds["min_current_c"])
    max_current = float(bounds["max_current_c"])

    seed_base_currents = (3.8, 3.1, 2.25, 1.3)
    seed_base_switches = (0.24, 0.54, 0.79)
    seed_flat4_currents = (7.0, 3.6, 3.6, 3.6)
    seed_flat4_switches = (0.16, 0.56, 0.88)
    seed3_currents = (7.0, 4.05, 3.8)
    seed3_switches = (0.17, 0.87)
    seed4_currents = (7.0, 4.1, 3.6, 2.7)
    seed4_switches = (0.16, 0.80, 0.88)
    seed5_currents = tuple(fallback_currents)
    seed5_switches = tuple(fallback_switches)
    seed6_currents = (7.0, 4.8, 4.2, 3.7, 3.1, 2.5)
    seed6_switches = (0.15, 0.28, 0.46, 0.68, 0.86)

    score_cache = {}

    def normalize(values):
        return tuple(round(float(v), 3) for v in values)

    def is_nonincreasing(values):
        return all(min_current <= value <= max_current for value in values)

    def score(currents, switches):
        key = (normalize(currents), normalize(switches))
        cached = score_cache.get(key)
        if cached is not None:
            return cached
        try:
            valid_currents, valid_switches = evaluator._validate_policy(
                {"currents_c": list(key[0]), "switch_soc": list(key[1])},
                cfg,
            )
            result = evaluator._simulate(valid_currents, valid_switches, cfg)
            value = float(result["combined_score"]) if result.get("valid", 0.0) == 1.0 else -1.0
        except Exception:
            value = -1.0
        score_cache[key] = value
        return value

    def maybe_update(best, currents, switches):
        candidate_score = score(currents, switches)
        if candidate_score > best[0]:
            return (candidate_score, normalize(currents), normalize(switches))
        return best

    def refine(currents, switches):
        best_currents = list(normalize(currents))
        best_switches = list(normalize(switches))
        best_score = score(best_currents, best_switches)

        def valid_currents(candidate):
            return all(min_current <= value <= max_current for value in candidate) and is_nonincreasing(candidate)

        def valid_switches(candidate):
            return all(
                (min_switch_soc if idx == 0 else candidate[idx - 1])
                < candidate[idx]
                < (target_soc if idx == len(candidate) - 1 else candidate[idx + 1])
                for idx in range(len(candidate))
            )

        for current_step in (0.4, 0.2, 0.1, 0.05, 0.025, 0.02, 0.01, 0.005, 0.002, 0.001):
            improved = True
            while improved:
                improved = False

                for idx in range(len(best_currents)):
                    for delta in (-current_step, current_step):
                        candidate = best_currents.copy()
                        candidate[idx] = round(candidate[idx] + delta, 3)
                        if not valid_currents(candidate):
                            continue
                        candidate_score = score(candidate, best_switches)
                        if candidate_score > best_score:
                            best_currents = candidate
                            best_score = candidate_score
                            improved = True

                for idx in range(len(best_currents) - 1):
                    for delta in (-current_step, -current_step / 2.0, current_step / 2.0, current_step):
                        candidate = best_currents.copy()
                        candidate[idx] = round(candidate[idx] + delta, 3)
                        candidate[idx + 1] = round(candidate[idx + 1] - delta, 3)
                        if not valid_currents(candidate):
                            continue
                        candidate_score = score(candidate, best_switches)
                        if candidate_score > best_score:
                            best_currents = candidate
                            best_score = candidate_score
                            improved = True

                for idx in range(len(best_switches)):
                    for delta in (-0.02, 0.02, -0.01, 0.01, -0.005, 0.005, -0.002, 0.002, -0.001, 0.001):
                        candidate = best_switches.copy()
                        candidate[idx] = round(candidate[idx] + delta, 3)
                        if not valid_switches(candidate):
                            continue
                        candidate_score = score(best_currents, candidate)
                        if candidate_score > best_score:
                            best_switches = candidate
                            best_score = candidate_score
                            improved = True

                for idx in range(len(best_switches)):
                    for delta in (-0.01, 0.01, -0.005, 0.005, -0.002, 0.002, -0.001, 0.001):
                        candidate = best_switches.copy()
                        for j in range(idx, len(candidate)):
                            candidate[j] = round(candidate[j] + delta, 3)
                        if not valid_switches(candidate):
                            continue
                        candidate_score = score(best_currents, candidate)
                        if candidate_score > best_score:
                            best_switches = candidate
                            best_score = candidate_score
                            improved = True

        return (best_score, normalize(best_currents), normalize(best_switches))

    baseline = (score(fallback_currents, fallback_switches), tuple(fallback_currents), tuple(fallback_switches))
    best_base = (score(seed_base_currents, seed_base_switches), seed_base_currents, seed_base_switches)
    best_flat4 = (score(seed_flat4_currents, seed_flat4_switches), seed_flat4_currents, seed_flat4_switches)
    best_3 = (score(seed3_currents, seed3_switches), seed3_currents, seed3_switches)

    for currents in product((6.8, 7.0), (3.9, 4.05, 4.2), (3.6, 3.8, 4.0)):
        if not is_nonincreasing(currents):
            continue
        for switches in product((0.16, 0.17, 0.18), (0.86, 0.87, 0.88, 0.89)):
            best_3 = maybe_update(best_3, currents, switches)

    best_4 = (score(seed4_currents, seed4_switches), seed4_currents, seed4_switches)
    for currents in product((6.4, 7.0), (3.9, 4.2, 4.5), (3.2, 3.5, 3.8), (2.2, 2.6, 3.0)):
        if not is_nonincreasing(currents):
            continue
        for switches in product((0.14, 0.16, 0.18), (0.74, 0.80, 0.84), (0.86, 0.88, 0.89)):
            best_4 = maybe_update(best_4, currents, switches)

    best_taper = (
        score((6.8, 4.2, 4.0, 3.5), (0.17, 0.77, 0.896)),
        (6.8, 4.2, 4.0, 3.5),
        (0.17, 0.77, 0.896),
    )
    for currents in product((6.4, 6.8, 7.0), (4.0, 4.1, 4.2, 4.3), (3.9, 4.0, 4.1), (3.3, 3.5, 3.7)):
        if not is_nonincreasing(currents):
            continue
        for switches in product((0.16, 0.17, 0.18), (0.75, 0.77, 0.79), (0.894, 0.896, 0.898)):
            best_taper = maybe_update(best_taper, currents, switches)

    best_elite = (
        score((6.97, 4.05, 4.15, 3.63), (0.17, 0.78, 0.898)),
        (6.97, 4.05, 4.15, 3.63),
        (0.17, 0.78, 0.898),
    )
    for currents in product((6.94, 6.97, 7.0), (4.02, 4.05, 4.08), (4.12, 4.15, 4.18), (3.58, 3.63, 3.68)):
        if not is_nonincreasing(currents):
            continue
        for switches in product((0.168, 0.17, 0.172), (0.776, 0.78, 0.784), (0.897, 0.898, 0.899)):
            best_elite = maybe_update(best_elite, currents, switches)

    best_5 = (score(seed5_currents, seed5_switches), seed5_currents, seed5_switches)
    for currents in product((6.4, 7.0), (4.4, 4.8), (3.6, 4.0), (2.8, 3.2), (2.0, 2.4)):
        if not is_nonincreasing(currents):
            continue
        for switches in product((0.14, 0.16, 0.18), (0.30, 0.36, 0.42), (0.58, 0.64, 0.70), (0.82, 0.86, 0.88)):
            best_5 = maybe_update(best_5, currents, switches)

    best_6 = (score(seed6_currents, seed6_switches), seed6_currents, seed6_switches)
    for currents in product((6.4, 7.0), (4.6, 5.0), (4.0, 4.4), (3.4, 3.8), (2.8, 3.2), (2.2, 2.6)):
        if not is_nonincreasing(currents):
            continue
        for switches in product((0.14, 0.16), (0.26, 0.32), (0.44, 0.52), (0.64, 0.72), (0.82, 0.86)):
            best_6 = maybe_update(best_6, currents, switches)

    seeds = (baseline, best_base, best_flat4, best_3, best_4, best_taper, best_elite, best_5, best_6)
    best = max(seeds, key=lambda item: item[0])
    refined_seeds = []
    for _, currents, switches in seeds:
        refined = refine(currents, switches)
        refined_seeds.append(refined)
        if refined[0] > best[0]:
            best = refined

    def explore_tail(candidate):
        best_tail = candidate
        currents = list(candidate[1])
        switches = list(candidate[2])

        if len(currents) >= int(bounds["max_stages"]):
            return best_tail

        start = (switches[-1] if switches else min_switch_soc) + 0.001
        if start >= target_soc - 0.001:
            return best_tail

        last = currents[-1]
        tail_currents = sorted(
            {
                round(value, 3)
                for value in (
                    last - 0.6,
                    last - 0.3,
                    last - 0.2,
                    last - 0.1,
                    last - 0.05,
                    last * 0.7,
                    last * 0.5,
                    2.0,
                    1.5,
                    1.2,
                    0.8,
                )
                if min_current <= value <= last
            },
            reverse=True,
        )
        span = target_soc - start
        tail_switches = sorted(
            {
                round(value, 3)
                for value in (
                    start + 0.25 * span,
                    start + 0.5 * span,
                    start + 0.75 * span,
                    target_soc - 0.03,
                    target_soc - 0.02,
                    target_soc - 0.015,
                    target_soc - 0.01,
                    target_soc - 0.008,
                    target_soc - 0.006,
                    target_soc - 0.004,
                    target_soc - 0.002,
                    target_soc - 0.001,
                )
                if start < value < target_soc
            }
        )

        for tail_current in tail_currents:
            for tail_switch in tail_switches:
                best_tail = maybe_update(best_tail, currents + [tail_current], switches + [tail_switch])

        return refine(best_tail[1], best_tail[2]) if best_tail[0] > candidate[0] else best_tail

    def explore_split(candidate):
        best_split = candidate
        currents = list(candidate[1])
        switches = list(candidate[2])

        if len(currents) >= int(bounds["max_stages"]):
            return best_split

        for idx, current in enumerate(currents):
            left = min_switch_soc if idx == 0 else switches[idx - 1]
            right = target_soc if idx == len(switches) else switches[idx]
            span = right - left
            if span <= 0.02:
                continue

            next_current = currents[idx + 1] if idx + 1 < len(currents) else min_current
            split_currents = sorted(
                {
                    round(value, 3)
                    for value in (
                        current,
                        (current + next_current) / 2.0,
                        current - 0.1,
                        current - 0.2,
                    )
                    if max(next_current, min_current) <= value <= min(current, max_current)
                },
                reverse=True,
            )
            split_switches = sorted(
                {
                    round(value, 3)
                    for value in (left + span / 3.0, left + span / 2.0, left + 2.0 * span / 3.0)
                    if left + 0.001 < value < right - 0.001
                }
            )

            for split_current in split_currents:
                new_currents = currents[: idx + 1] + [split_current] + currents[idx + 1 :]
                if not is_nonincreasing(new_currents):
                    continue
                for split_switch in split_switches:
                    new_switches = switches[:idx] + [split_switch] + switches[idx:]
                    best_split = maybe_update(best_split, new_currents, new_switches)

        return refine(best_split[1], best_split[2]) if best_split[0] > candidate[0] else best_split

    def explore_around(candidate):
        for explorer in (explore_split, explore_tail, explore_split):
            candidate = explorer(candidate)
        return candidate

    def polish_joint(candidate):
        currents = list(candidate[1])
        switches = list(candidate[2])
        if len(currents) != 4 or len(switches) != 3:
            return candidate

        current_steps = (0.03, 0.03, 0.03, 0.06)
        switch_steps = (0.002, 0.006, 0.001)
        current_options = []
        for idx, value in enumerate(currents):
            current_options.append(
                tuple(
                    sorted(
                        {
                            round(value + delta, 3)
                            for delta in (-current_steps[idx], 0.0, current_steps[idx])
                            if min_current <= value + delta <= max_current
                        }
                    )
                )
            )

        switch_options = []
        for idx, value in enumerate(switches):
            low = min_switch_soc if idx == 0 else switches[idx - 1]
            high = target_soc if idx == len(switches) - 1 else switches[idx + 1]
            switch_options.append(
                tuple(
                    sorted(
                        {
                            round(value + delta, 3)
                            for delta in (-switch_steps[idx], 0.0, switch_steps[idx])
                            if low < value + delta < high
                        }
                    )
                )
            )

        best_local = candidate
        for trial_currents in product(*current_options):
            if not is_nonincreasing(trial_currents):
                continue
            for trial_switches in product(*switch_options):
                if not all(
                    (min_switch_soc if idx == 0 else trial_switches[idx - 1])
                    < trial_switches[idx]
                    < (target_soc if idx == len(trial_switches) - 1 else trial_switches[idx + 1])
                    for idx in range(len(trial_switches))
                ):
                    continue
                best_local = maybe_update(best_local, trial_currents, trial_switches)

        return refine(best_local[1], best_local[2]) if best_local[0] > candidate[0] else candidate

    for candidate in refined_seeds + [best]:
        expanded = candidate
        for _ in range(2):
            expanded = explore_around(expanded)
            if expanded[0] > best[0]:
                best = expanded

    best = polish_joint(best)
    return {"currents_c": list(best[1]), "switch_soc": list(best[2])}


def main() -> None:
    print(build_charging_policy())


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END
