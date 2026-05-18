"""Simple baseline for the Reizman Suzuki emulator task."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _is_repo_root(path: Path) -> bool:
    return (path / "benchmarks").is_dir() and (path / "frontier_eval").is_dir()


def _ensure_domain_on_path() -> None:
    env_root = (os.environ.get("FRONTIER_ENGINEERING_ROOT") or "").strip()
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    here = Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])

    repo_root = next((path for path in candidates if _is_repo_root(path)), None)
    if repo_root is None:
        raise RuntimeError("Could not locate repository root for ReactionOptimisation.")

    domain_root = repo_root / "benchmarks" / "ReactionOptimisation"
    if not domain_root.is_dir():
        raise RuntimeError(f"ReactionOptimisation directory not found under {repo_root}.")

    domain_root_str = str(domain_root)
    if domain_root_str not in sys.path:
        sys.path.insert(0, domain_root_str)


_ensure_domain_on_path()

from reizman_suzuki_pareto import task
from shared.utils import dump_json, seed_everything


# EVOLVE-BLOCK-START
def _scalarize(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        keys = [k for k in value if isinstance(k, str)]
        for tag in ("hypervolume", "hv", "score", "reward", "value", "objective", "yield", "best"):
            for key in keys:
                if tag in key.lower():
                    out = _scalarize(value[key])
                    if out is not None:
                        return out
        vals = [v for v in (_scalarize(v) for v in value.values()) if v is not None]
        return float(np.mean(vals)) if vals else None
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [v for v in (_scalarize(v) for v in value) if v is not None]
        return float(np.mean(vals)) if vals else None
    return None


def _summary_value(records: list[dict]) -> float:
    if not records:
        return 0.0
    try:
        out = _scalarize(task.summarize(records))
        if out is not None:
            return out
    except Exception:
        pass
    out = _scalarize(records[-1])
    return 0.0 if out is None else out


def _flatten(value, path: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        out: list[tuple[str, object]] = []
        for key in sorted(value):
            next_path = f"{path}.{key}" if path else str(key)
            out.extend(_flatten(value[key], next_path))
        return out
    if isinstance(value, np.ndarray) and value.ndim > 0:
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out = []
        for i, item in enumerate(value):
            out.extend(_flatten(item, f"{path}[{i}]"))
        return out
    return [(path, value)]


def _encode(candidate, cats: dict[str, dict[str, float]]) -> np.ndarray:
    feats: list[float] = []
    for path, value in _flatten(candidate):
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            feats.append(float(value))
        else:
            table = cats.setdefault(path, {})
            key = repr(value)
            if key not in table:
                table[key] = float(len(table))
            feats.append(table[key])
    return np.asarray(feats, dtype=float)


def _pad(vectors: list[np.ndarray], width: int) -> np.ndarray:
    out = np.zeros((len(vectors), width), dtype=float)
    for i, vec in enumerate(vectors):
        out[i, : len(vec)] = vec
    return out


def _acquisition(X: np.ndarray, y: np.ndarray, P: np.ndarray) -> np.ndarray:
    scale = X.std(axis=0) + 1e-6
    y_scale = y.std() + 1e-6
    k = min(len(y), max(3, int(np.sqrt(len(y)))))
    out = np.empty(len(P), dtype=float)
    for i, x in enumerate(P):
        d2 = np.sum(((X - x) / scale) ** 2, axis=1)
        idx = np.argpartition(d2, k - 1)[:k]
        w = 1.0 / (d2[idx] + 1e-6)
        mu = float(np.dot(w, y[idx]) / w.sum())
        var = float(np.dot(w, (y[idx] - mu) ** 2) / w.sum())
        out[i] = mu + 0.25 * np.sqrt(var) + 0.05 * y_scale * float(np.sqrt(d2.min()))
    return out


def _clip_candidate(candidate: dict) -> dict:
    out = {"catalyst": str(candidate["catalyst"])}
    for name, (low, high) in task.BOUNDS.items():
        out[name] = float(np.clip(float(candidate[name]), low, high))
    return out


def _vectorize(candidate: dict) -> np.ndarray:
    return np.asarray(
        [
            (float(candidate[name]) - low) / (high - low)
            for name, (low, high) in task.BOUNDS.items()
        ],
        dtype=float,
    )


def _template_candidate(catalyst: str, weight: float, base: dict) -> dict:
    candidate = {
        "catalyst": catalyst,
        "t_res": float(base.get("t_res", 360.0)),
        "temperature": float(base.get("temperature", 100.0)),
        "catalyst_loading": float(base.get("catalyst_loading", 2.0)),
    }
    if catalyst == "P1-L4":
        if weight < 0.3:
            candidate.update(t_res=360.0, temperature=96.0, catalyst_loading=2.5)
        elif weight > 0.7:
            candidate.update(t_res=268.0, temperature=110.0, catalyst_loading=2.5)
        else:
            candidate.update(t_res=330.0, temperature=100.0, catalyst_loading=2.5)
    elif catalyst == "P2-L1":
        if weight < 0.35:
            candidate.update(t_res=340.0, temperature=103.0, catalyst_loading=2.5)
        else:
            candidate.update(t_res=323.0, temperature=110.0, catalyst_loading=2.5)
    elif catalyst == "P1-L7":
        candidate.update(
            t_res=190.0 if weight > 0.7 else 300.0,
            temperature=90.0 if weight > 0.4 else 100.0,
            catalyst_loading=2.1,
        )
    else:
        candidate.update(
            t_res=330.0,
            temperature=108.0 if weight > 0.7 else 100.0,
            catalyst_loading=2.3,
        )
    return _clip_candidate(candidate)


def _mutate_fixed(catalyst: str, base: dict, rng: np.random.Generator, scale: float) -> dict:
    candidate = {"catalyst": catalyst}
    for name, (low, high) in task.BOUNDS.items():
        span = high - low
        center = float(base[name])
        if rng.random() < 0.12:
            value = rng.uniform(low, high)
        else:
            value = center + rng.normal(0.0, scale * span)
        if name == "catalyst_loading" and rng.random() < 0.5:
            value = max(value, high - 0.12 * span * abs(float(rng.normal())))
        if name == "temperature" and catalyst in {"P1-L4", "P2-L1"} and rng.random() < 0.35:
            value = max(value, high - 0.18 * span * abs(float(rng.normal())))
        if rng.random() < 0.08:
            value = low if rng.random() < 0.15 else high
        candidate[name] = float(np.clip(value, low, high))
    return candidate


def _blend_fixed(catalyst: str, a: dict, b: dict, rng: np.random.Generator) -> dict:
    candidate = {"catalyst": catalyst}
    for name, (low, high) in task.BOUNDS.items():
        span = high - low
        mix = float(rng.uniform(0.2, 0.8))
        value = mix * float(a[name]) + (1.0 - mix) * float(b[name]) + rng.normal(0.0, 0.04 * span)
        candidate[name] = float(np.clip(value, low, high))
    return candidate


def _suggest_fixed(
    catalyst: str,
    records: list[dict],
    weight: float,
    rng: np.random.Generator,
) -> dict:
    ranked = sorted(records, key=lambda row: task.scalarize(row, weight), reverse=True)
    top = ranked[: max(1, min(4, len(ranked)))]
    pool: list[dict] = []
    pool_vecs: list[np.ndarray] = []
    signatures: set[str] = set()
    attempts = 0
    target = _vectorize(_template_candidate(catalyst, weight, top[0]))

    while len(pool) < 64 and attempts < 256:
        attempts += 1
        mode = rng.random()
        if mode < 0.15:
            candidate = _template_candidate(catalyst, weight, top[0])
        elif mode < 0.75:
            candidate = _mutate_fixed(
                catalyst,
                top[int(rng.integers(len(top)))],
                rng,
                max(0.04, 0.18 / np.sqrt(len(records) + 1.0)),
            )
        elif mode < 0.9 and len(top) > 1:
            a = top[int(rng.integers(len(top)))]
            b = top[int(rng.integers(len(top)))]
            candidate = _blend_fixed(catalyst, a, b, rng)
        else:
            candidate = task.sample_candidate(rng)
            candidate["catalyst"] = catalyst
            candidate = _clip_candidate(candidate)

        sig = repr(candidate)
        if sig in signatures:
            continue
        signatures.add(sig)
        pool.append(candidate)
        pool_vecs.append(_vectorize(candidate))

    if not pool:
        return _mutate_fixed(catalyst, top[0], rng, 0.15)

    X = np.vstack([_vectorize(row) for row in records])
    y = np.asarray([task.scalarize(row, weight) for row in records], dtype=float)
    P = np.vstack(pool_vecs)
    acq = _acquisition(X, y, P)
    novelty = np.sqrt(((P[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)).min(axis=1)

    prior = -np.abs(P[:, 0] - target[0]) - 0.6 * np.abs(P[:, 1] - target[1]) - 0.5 * np.abs(P[:, 2] - target[2])
    if catalyst in {"P1-L4", "P2-L1"}:
        prior += 0.15 * P[:, 1] + 0.2 * P[:, 2]
    if weight > 0.7:
        prior += 0.2 * P[:, 1] + 0.1 * P[:, 2]
    elif weight < 0.3:
        prior += 0.08 * P[:, 2]

    return pool[int(np.argmax(acq + 0.08 * novelty + 0.1 * prior))]


def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []
    seen: set[str] = set()
    current_hv = 0.0

    if budget <= 0:
        return {
            "task_name": task.TASK_NAME,
            "algorithm_name": "screened_branch_search",
            "seed": seed,
            "budget": budget,
            "history": history,
            "summary": task.summarize(history),
        }

    def run(candidate: dict) -> dict:
        nonlocal current_hv
        actual = _clip_candidate(candidate)
        for _ in range(8):
            if repr(actual) not in seen:
                break
            actual = _mutate_fixed(str(actual["catalyst"]), actual, rng, 0.05)
        try:
            record = task.evaluate(experiment, actual)
        except Exception:
            actual = _clip_candidate(task.sample_candidate(rng))
            while repr(actual) in seen:
                actual = _clip_candidate(task.sample_candidate(rng))
            record = task.evaluate(experiment, actual)
        seen.add(repr(actual))
        history.append(record)
        current_hv = task.summarize(history)["hypervolume"]
        return record

    screening: list[dict] = []
    screen_point = {"t_res": 360.0, "temperature": 100.0, "catalyst_loading": 2.0}
    for catalyst in task.CATEGORIES["catalyst"]:
        if len(history) >= budget:
            break
        screening.append(run({"catalyst": catalyst, **screen_point}))

    if len(history) >= budget or not screening:
        return {
            "task_name": task.TASK_NAME,
            "algorithm_name": "screened_branch_search",
            "seed": seed,
            "budget": budget,
            "history": history,
            "summary": task.summarize(history),
        }

    weights = [0.15, 0.5, 0.85]
    branches: list[dict] = []
    used_catalysts: set[str] = set()

    for weight in weights:
        ranked = sorted(screening, key=lambda row: task.scalarize(row, weight), reverse=True)
        seed_record = ranked[0]
        if str(seed_record["catalyst"]) in used_catalysts:
            alt = next((row for row in ranked if str(row["catalyst"]) not in used_catalysts), None)
            if alt is not None and task.scalarize(alt, weight) + 0.03 >= task.scalarize(seed_record, weight):
                seed_record = alt
        used_catalysts.add(str(seed_record["catalyst"]))

        branch = {
            "weight": weight,
            "catalyst": str(seed_record["catalyst"]),
            "records": [seed_record],
            "gains": [],
            "spent": 0,
            "target": 5 if weight > 0.7 else 4,
        }

        if len(history) < budget:
            template = _template_candidate(branch["catalyst"], weight, seed_record)
            seed_candidate = {
                "catalyst": str(seed_record["catalyst"]),
                "t_res": float(seed_record["t_res"]),
                "temperature": float(seed_record["temperature"]),
                "catalyst_loading": float(seed_record["catalyst_loading"]),
            }
            if repr(template) != repr(seed_candidate):
                before = current_hv
                branch["records"].append(run(template))
                branch["gains"].append(max(0.0, current_hv - before))
                branch["spent"] += 1

        branches.append(branch)

    while len(history) < budget:
        pending = [branch for branch in branches if branch["spent"] < branch["target"]]

        if pending:
            branch = max(
                pending,
                key=lambda row: (
                    row["target"] - row["spent"],
                    max(task.scalarize(r, row["weight"]) for r in row["records"]),
                ),
            )
        else:
            step = len(history) - len(screening)
            branch_scores = []
            for branch in branches:
                merit = max(task.scalarize(row, branch["weight"]) for row in branch["records"])
                recent = float(np.mean(branch["gains"][-2:])) if branch["gains"] else 0.0
                best_gain = max(branch["gains"][-4:], default=0.0)
                explore = 0.03 * np.sqrt(np.log(step + 3.0) / (branch["spent"] + 1.0))
                branch_scores.append(merit + 3.0 * recent + 2.0 * best_gain + explore)
            branch = branches[int(np.argmax(branch_scores))]

        candidate = _suggest_fixed(branch["catalyst"], branch["records"], branch["weight"], rng)
        before = current_hv
        branch["records"].append(run(candidate))
        branch["gains"].append(max(0.0, current_hv - before))
        branch["spent"] += 1

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "screened_branch_search",
        "seed": seed,
        "budget": budget,
        "history": history,
        "summary": task.summarize(history),
    }
# EVOLVE-BLOCK-END


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=task.DEFAULT_BUDGET)
    args = parser.parse_args()
    print(dump_json(solve(seed=args.seed, budget=args.budget)))


if __name__ == "__main__":
    main()
