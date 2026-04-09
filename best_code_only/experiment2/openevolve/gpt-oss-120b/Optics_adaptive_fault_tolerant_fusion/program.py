# EVOLVE-BLOCK-START
import numpy as np


def fuse_and_compute_dm_commands(
    slopes_multi: np.ndarray,
    reconstructor: np.ndarray,
    control_model: dict,
    prev_commands: np.ndarray | None = None,
    max_voltage: float = 0.50,
) -> np.ndarray:
    """
    Baseline: naive average over all WFS channels.

    Sensitive to corrupted sensors.
    """
    # ----------------------------------------------------------------------
    # Robust fusion of slopes from multiple wavefront sensors.
    # 1️⃣  If explicit per‑sensor weights are supplied, use them directly.
    # 2️⃣  Otherwise compute a median, measure each sensor’s deviation,
    #     keep the most “in‑lier” sensors and weight them with a soft‑max‑like
    #     scheme (the same idea used by the reference controller).
    # ----------------------------------------------------------------------
    weights = control_model.get('weights')
    if weights is not None:
        # Explicit static weights supplied by the caller.
        # Convert to a 1‑D float array and normalise safely.
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr = weights_arr / np.maximum(weights_arr.sum(), 1e-12)

        # Use a dot‑product for the weighted mean – mathematically identical
        # to ``np.average`` but slightly faster and avoids the internal loop.
        fused = np.dot(weights_arr, slopes_multi) / np.maximum(weights_arr.sum(), 1e-12)
    else:
        # --------------------------------------------------------------
        # Robust fusion – combine model‑based and deviation‑based estimates.
        # --------------------------------------------------------------
        model = control_model.get("anomaly_model")
        # Parameters (same defaults as the reference controller)
        inlier_frac = float(control_model.get('inlier_fraction', 0.4))
        score_temp   = float(control_model.get('score_temperature', 0.08))

        # Number of sensors to keep as in‑liers
        n_keep = max(1, int(np.ceil(inlier_frac * slopes_multi.shape[0])))

        if model is not None and slopes_multi.shape[0] >= 2:
            # ------------------------------
            # 1️⃣  Model‑based fusion
            # ------------------------------
            scores = model.decision_function(slopes_multi)          # higher ⇒ more normal
            keep_idx_model = np.argsort(scores)[-n_keep:]           # top‑scoring sensors
            kept_scores = scores[keep_idx_model]
            # soft‑max weighting (stable)
            kept_scores = kept_scores - np.max(kept_scores)
            w_model = np.exp(kept_scores / (score_temp + 1e-12))
            w_model /= np.sum(w_model) + 1e-12
            fused_model = np.sum(slopes_multi[keep_idx_model] * w_model[:, None], axis=0)

            # ------------------------------
            # 2️⃣  Deviation‑based fusion (median + L1 deviation)
            # ------------------------------
            median = np.median(slopes_multi, axis=0)
            # Use Median Absolute Deviation (MAD) for a more robust estimate.
            deviations = np.median(np.abs(slopes_multi - median), axis=1)
            keep_idx_dev = np.argsort(deviations)[:n_keep]
            sel_dev = deviations[keep_idx_dev]
            w_dev = np.exp(-sel_dev / (score_temp + 1e-12))
            w_dev /= np.sum(w_dev) + 1e-12
            fused_dev = np.sum(slopes_multi[keep_idx_dev] * w_dev[:, None], axis=0)

            # ------------------------------
            # 3️⃣  Blend the two estimates
            #     Higher variance of the model scores ⇒ trust the model more.
            # ------------------------------
            score_var = np.var(scores)
            # Use a smaller denominator so that the blend leans more heavily
            # on the model when the scores exhibit noticeable variance.
            blend_w = score_var / (score_var + 0.5)          # in [0, 1]
            fused = blend_w * fused_model + (1.0 - blend_w) * fused_dev
        else:
            # ----------------------------------------------------------
            # No model available – fall back to pure deviation‑based fusion
            # ----------------------------------------------------------
            median = np.median(slopes_multi, axis=0)
            # Use Median Absolute Deviation (MAD) for a more robust estimate.
            deviations = np.median(np.abs(slopes_multi - median), axis=1)
            keep_idx = np.argsort(deviations)[:n_keep]
            sel_dev = deviations[keep_idx]
            w = np.exp(-sel_dev / (score_temp + 1e-12))
            w /= np.sum(w) + 1e-12
            fused = np.sum(slopes_multi[keep_idx] * w[:, None], axis=0)

    # Compute raw DM commands
    u_raw = reconstructor @ fused

    # Optional temporal blending with previous commands.
    # The reference controller uses the key `temporal_blend`; we also accept the
    # older `alpha` key for backward compatibility.
    # Blend factor is expected to be in [0, 1]; clamp to avoid accidental overflow.
    blend = float(control_model.get('temporal_blend',
                                   control_model.get('alpha', 0.0)))
    blend = min(max(blend, 0.0), 1.0)

    if prev_commands is not None and blend > 0.0:
        u_raw = blend * u_raw + (1.0 - blend) * prev_commands

    # Clip to voltage limits
    return np.clip(u_raw, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
