#!/usr/bin/env python
# EVOLVE-BLOCK-START
"""Hybrid ensemble multi-strategy optimizer for Dammann grating transitions.

Architecture:
- Four fundamentally different optimization strategies
- Dual Annealing: stochastic global search with temperature schedule
- Basin-hopping: Monte Carlo with local minimization cycles
- Multi-start L-BFGS-B: gradient-based local search from random seeds
- Differential Evolution: population-based evolutionary optimizer
- Best-of selection across all strategies
- Symmetry-aware parameterization preserved
- Caching for efficient objective evaluation
- Hybrid parameters from crossover optimization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
from scipy.optimize import dual_annealing, minimize, basinhopping, differential_evolution

from diffractio import um, mm
from diffractio.scalar_masks_X import Scalar_mask_X


DEFAULT_CONFIG: Dict[str, Any] = {
    "period_size": 40 * um,
    "wavelength": 0.6328 * um,
    "period_pixels": 256,
    "num_transitions": 14,
    "num_repetitions": 10,
    "focal": 1 * mm,
    "lens_radius": 1 * mm,
    "order_min": -3,
    "order_max": 3,
    "order_window_halfwidth_px": 3,
}


class EnsembleOptimizer:
    """Ensemble optimizer combining multiple fundamentally different strategies.

    Each strategy uses a completely different approach to explore the search space:
    1. Dual Annealing - Temperature-based stochastic acceptance
    2. Basin-hopping - Random perturbation + local minimization cycles
    3. Multi-start L-BFGS-B - Multiple gradient-based local searches
    4. Differential Evolution - Population-based evolutionary optimization
    """

    def __init__(self, problem: Dict[str, Any]):
        self.problem = problem
        self.cfg = problem["cfg"]
        self.period = self.cfg["period_size"]
        self.n_trans = self.cfg["num_transitions"]
        self.n_half = self.n_trans // 2
        self._min_spacing = 0.01 * self.period
        self._bounds: Optional[List[Tuple[float, float]]] = None
        # Cache for objective function evaluations
        self._cache: Dict[str, float] = {}
        self._cache_hits = 0

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """Get optimization bounds with margin."""
        if self._bounds is None:
            margin = 0.02 * self.period
            self._bounds = [(margin, 0.48 * self.period) for _ in range(self.n_half)]
        return self._bounds

    def params_to_transitions(self, params: np.ndarray) -> np.ndarray:
        """Convert symmetric half-parameters to full transition array."""
        sorted_params = np.sort(np.abs(params))
        negative_side = -sorted_params[::-1]
        positive_side = sorted_params
        transitions = np.concatenate([negative_side, positive_side])
        return np.sort(transitions)

    def enforce_spacing_constraint(self, transitions: np.ndarray) -> np.ndarray:
        """Enforce minimum spacing between adjacent transitions."""
        result = transitions.copy()
        for i in range(1, len(result)):
            if result[i] - result[i-1] < self._min_spacing:
                result[i] = result[i-1] + self._min_spacing
        return result

    def compute_score(self, transitions: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Compute the exact scoring metrics for given transitions."""
        try:
            field = build_incident_field(self.problem, transitions)
            focus_field = field.RS(z=self.cfg["focal"], new_field=True, verbose=False)
            intensity = np.abs(focus_field.u) ** 2
            metrics = evaluate_orders(self.problem, intensity, focus_field.x)

            cv = metrics["cv_orders"]
            eff = metrics["efficiency"]
            min_max = metrics["min_to_max"]

            uniform_score = max(0.0, min(1.0, 1.0 - cv / 0.9))
            efficiency_score = max(0.0, min(1.0, (eff - 0.003) / 0.177))
            balance_score = max(0.0, min(1.0, (min_max - 0.15) / 0.75))

            total_score = (
                0.60 * uniform_score +
                0.30 * efficiency_score +
                0.10 * balance_score
            )

            return total_score, metrics
        except Exception:
            return 0.0, {}

    def objective(self, params: np.ndarray) -> float:
        """Objective function for minimization: negative of the score."""
        # Create cache key with quantization for approximate matching
        key = tuple(np.round(params / (self.period * 1e-6)).astype(np.int64))
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        transitions = self.params_to_transitions(params)
        transitions = self.enforce_spacing_constraint(transitions)
        score, _ = self.compute_score(transitions)
        result = -score

        # Store in cache (limit size to prevent memory issues)
        if len(self._cache) < 50000:
            self._cache[key] = result

        return result

    def _strategy_dual_annealing(self) -> Tuple[np.ndarray, float]:
        """Dual annealing: stochastic global search with temperature schedule.

        Uses Metropolis acceptance criterion with adaptive temperature.
        """
        bounds = self._get_bounds()
        result = dual_annealing(
            self.objective,
            bounds,
            maxiter=400,
            initial_temp=5230.0,
            restart_temp_ratio=2e-5,
            visit=2.62,
            accept=-5.0,
            seed=42,
            no_local_search=False,
        )
        return result.x, -result.fun

    def _strategy_differential_evolution(self) -> Tuple[np.ndarray, float]:
        """Differential evolution: population-based evolutionary optimizer.

        Hybrid parameters combining wider mutation from Program 1 with
        larger population from Program 2 for better exploration.
        """
        bounds = self._get_bounds()
        result = differential_evolution(
            self.objective,
            bounds,
            maxiter=300,
            popsize=15,
            mutation=(0.6, 1.2),  # Wider range from Program 1 for better exploration
            recombination=0.75,   # Hybrid value between both programs
            strategy='best1bin',
            seed=42,
            polish=True,
            tol=1e-7,
            atol=1e-8,           # Better tolerance from Program 1
            workers=1,
            updating='deferred',
        )
        return result.x, -result.fun

    def _strategy_basin_hopping(self) -> Tuple[np.ndarray, float]:
        """Basin-hopping: Monte Carlo perturbations with local minimization.

        Uses random displacement + local minimization cycles.
        """
        bounds = self._get_bounds()
        bounds_arr = np.array(bounds)
        x0 = (bounds_arr[:, 0] + bounds_arr[:, 1]) / 2

        # Custom step-taking with bounds awareness
        class BoundedTakeStep:
            def __init__(self, stepsize, bounds):
                self.stepsize = stepsize
                self.bounds = np.array(bounds)

            def __call__(self, x):
                x_new = x + np.random.uniform(-self.stepsize, self.stepsize, len(x))
                return np.clip(x_new, self.bounds[:, 0], self.bounds[:, 1])

        step_taker = BoundedTakeStep(0.05 * self.period, bounds)

        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "options": {"maxiter": 80}
        }

        result = basinhopping(
            self.objective,
            x0,
            niter=120,
            T=0.5,
            stepsize=0.05 * self.period,
            minimizer_kwargs=minimizer_kwargs,
            take_step=step_taker,
            seed=42,
        )

        # Final local refinement
        refined = minimize(
            self.objective,
            result.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )

        best_x = refined.x if refined.fun < result.fun else result.x
        best_score = -min(refined.fun, result.fun)
        return best_x, best_score

    def _strategy_multistart_local(self, n_starts: int = 8) -> Tuple[np.ndarray, float]:
        """Multi-start L-BFGS-B: gradient-based local search from random seeds.

        Uses deterministic gradient-based optimization from multiple starting points.
        """
        bounds = self._get_bounds()
        bounds_arr = np.array(bounds)

        best_score = -np.inf
        best_x = None

        # Use Latin hypercube-like sampling for diverse starting points
        np.random.seed(42)
        for i in range(n_starts):
            # Stratified sampling within bounds
            x0 = bounds_arr[:, 0] + (bounds_arr[:, 1] - bounds_arr[:, 0]) * (
                i + 0.5
            ) / n_starts + np.random.uniform(-0.1, 0.1, self.n_half) * (
                bounds_arr[:, 1] - bounds_arr[:, 0]
            )
            x0 = np.clip(x0, bounds_arr[:, 0], bounds_arr[:, 1])

            try:
                result = minimize(
                    self.objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-9}
                )
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_x = result.x.copy()
            except Exception:
                continue

        if best_x is None:
            best_x = (bounds_arr[:, 0] + bounds_arr[:, 1]) / 2

        return best_x, best_score

    def optimize(self) -> np.ndarray:
        """Execute all strategies and return the best transitions found."""
        candidates: List[Tuple[np.ndarray, float, str]] = []

        # Strategy 1: Dual Annealing
        try:
            x1, score1 = self._strategy_dual_annealing()
            candidates.append((x1, score1, "dual_annealing"))
        except Exception:
            pass

        # Strategy 2: Basin-hopping
        try:
            x2, score2 = self._strategy_basin_hopping()
            candidates.append((x2, score2, "basin_hopping"))
        except Exception:
            pass

        # Strategy 3: Multi-start local
        try:
            x3, score3 = self._strategy_multistart_local(n_starts=8)
            candidates.append((x3, score3, "multistart_local"))
        except Exception:
            pass

        # Strategy 4: Differential Evolution
        try:
            x4, score4 = self._strategy_differential_evolution()
            candidates.append((x4, score4, "differential_evolution"))
        except Exception:
            pass

        # Select best across all strategies
        if not candidates:
            # Fallback to center of bounds
            bounds = self._get_bounds()
            best_params = np.array([(b[0] + b[1]) / 2 for b in bounds])
        else:
            best_params, best_score, _ = max(candidates, key=lambda c: c[1])

        # Final local refinement on best candidate
        bounds = self._get_bounds()
        try:
            refined = minimize(
                self.objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if refined.fun < -best_score:
                best_params = refined.x
                best_score = -refined.fun
        except Exception:
            pass

        # Additional Powell refinement for final polish (crossover addition)
        try:
            refined_powell = minimize(
                self.objective,
                best_params,
                method='Powell',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            if refined_powell.fun < -best_score:
                best_params = refined_powell.x
        except Exception:
            pass

        transitions = self.params_to_transitions(best_params)
        return self.enforce_spacing_constraint(transitions)


class OpticalFieldBuilder:
    """Builder class for constructing optical fields with Dammann gratings."""

    def __init__(self, problem: Dict[str, Any]):
        self.problem = problem
        self.cfg = problem["cfg"]
        self.x_period = problem["x_period"]

    def build(self, transitions: np.ndarray) -> Scalar_mask_X:
        """Build the complete optical field with grating and lens."""
        period = Scalar_mask_X(x=self.x_period, wavelength=self.cfg["wavelength"])
        period.binary_code_positions(x_transitions=transitions, start="down", has_draw=False)
        period.u = np.exp(1j * np.pi * period.u)

        dammann = period.repeat_structure(
            num_repetitions=self.cfg["num_repetitions"],
            position="center",
            new_field=True,
        )

        lens = Scalar_mask_X(x=dammann.x, wavelength=self.cfg["wavelength"])
        lens.lens(x0=0.0, focal=self.cfg["focal"], radius=self.cfg["lens_radius"])

        return dammann * lens


class OrderEvaluator:
    """Evaluates diffraction order energies and computes metrics."""

    def __init__(self, problem: Dict[str, Any]):
        self.problem = problem
        self.cfg = problem["cfg"]
        self.order_spacing = self.cfg["focal"] * self.cfg["wavelength"] / self.cfg["period_size"]

    def evaluate(self, intensity_x: np.ndarray, x: np.ndarray) -> Dict[str, Any]:
        """Compute order energies and derived metrics."""
        orders = np.arange(self.cfg["order_min"], self.cfg["order_max"] + 1, dtype=int)
        energies = []
        positions = []

        hw = int(self.cfg["order_window_halfwidth_px"])
        for m in orders:
            x_m = m * self.order_spacing
            ix = int(np.argmin(np.abs(x - x_m)))
            i0 = max(0, ix - hw)
            i1 = min(len(x), ix + hw + 1)
            energies.append(float(intensity_x[i0:i1].sum()))
            positions.append(float(x_m))

        energies = np.asarray(energies, dtype=float)
        cv = float(energies.std() / (energies.mean() + 1e-12))
        norm = energies / (energies.max() + 1e-12)
        efficiency = float(energies.sum() / (intensity_x.sum() + 1e-12))

        return {
            "orders": orders.tolist(),
            "order_positions": positions,
            "order_energies": energies.tolist(),
            "order_energies_norm": norm.tolist(),
            "cv_orders": cv,
            "efficiency": efficiency,
            "min_to_max": float(norm.min()),
        }


def build_problem(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build problem dictionary with configuration."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    x_period = np.linspace(
        -cfg["period_size"] / 2,
        cfg["period_size"] / 2,
        cfg["period_pixels"]
    )

    return {
        "cfg": cfg,
        "x_period": x_period,
    }


def build_incident_field(problem: Dict[str, Any], transitions: np.ndarray) -> Scalar_mask_X:
    """Build the incident optical field with Dammann grating and lens."""
    builder = OpticalFieldBuilder(problem)
    return builder.build(transitions)


def evaluate_orders(problem: Dict[str, Any], intensity_x: np.ndarray, x: np.ndarray) -> Dict[str, Any]:
    """Evaluate diffraction order energies and metrics."""
    evaluator = OrderEvaluator(problem)
    return evaluator.evaluate(intensity_x, x)


def baseline_transitions(problem: Dict[str, Any]) -> np.ndarray:
    """Generate optimized transition positions using ensemble optimizer."""
    optimizer = EnsembleOptimizer(problem)
    return optimizer.optimize()


def solve_baseline(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Solve the Dammann grating optimization problem."""
    transitions = baseline_transitions(problem)
    field = build_incident_field(problem, transitions)
    focus_field = field.RS(z=problem["cfg"]["focal"], new_field=True, verbose=False)

    intensity = np.abs(focus_field.u) ** 2
    metrics = evaluate_orders(problem, intensity, focus_field.x)

    return {
        "transitions": transitions,
        "x_focus": focus_field.x,
        "intensity_focus": intensity,
        "metrics": metrics,
    }


def save_solution(path: Path, solution: Dict[str, Any], problem: Dict[str, Any]) -> None:
    """Save solution to NPZ file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        transitions=solution["transitions"].astype(np.float32),
        x_focus=solution["x_focus"].astype(np.float32),
        intensity_focus=solution["intensity_focus"].astype(np.float32),
        period_size=np.float32(problem["cfg"]["period_size"]),
        wavelength=np.float32(problem["cfg"]["wavelength"]),
        focal=np.float32(problem["cfg"]["focal"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task03 ensemble Dammann transition solver")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "baseline_solution.npz",
        help="Output NPZ path",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON config overriding defaults",
    )
    args = parser.parse_args()

    config = None
    if args.config_json is not None:
        config = json.loads(args.config_json.read_text(encoding="utf-8"))

    problem = build_problem(config)
    solution = solve_baseline(problem)
    save_solution(args.output, solution, problem)

    print("[Task03/Ensemble] solution saved:", args.output)
    print("[Task03/Ensemble] cv_orders={:.6f}, efficiency={:.6f}".format(
        solution["metrics"]["cv_orders"],
        solution["metrics"]["efficiency"]
    ))


if __name__ == "__main__":
    main()
# EVOLVE-BLOCK-END