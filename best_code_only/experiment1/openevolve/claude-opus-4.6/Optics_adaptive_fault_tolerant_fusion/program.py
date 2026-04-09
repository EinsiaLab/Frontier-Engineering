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
    Anomaly-aware weighted fusion using control_model's anomaly detector,
    with median fallback. Mirrors reference approach.
    """
    n_wfs = slopes_multi.shape[0]
    model = control_model.get("anomaly_model")
    inlier_frac = float(control_model.get("inlier_fraction", 0.4))
    temperature = float(control_model.get("score_temperature", 0.08))

    if n_wfs >= 3:
        # Pairwise RMS distance matrix
        pw = np.zeros((n_wfs, n_wfs))
        for i in range(n_wfs):
            for j in range(i + 1, n_wfs):
                d = np.sqrt(np.mean((slopes_multi[i] - slopes_multi[j]) ** 2))
                pw[i, j] = d
                pw[j, i] = d

        # Consensus score: for each channel, median distance to others
        # Clean channels cluster together; corrupted ones are far from everyone
        med_dist = np.median(pw, axis=1)

        # IsolationForest scores as secondary signal
        if model is not None:
            iso_scores = model.decision_function(slopes_multi)
            # Normalize both to comparable scales
            md_range = med_dist.max() - med_dist.min() + 1e-12
            md_norm = (med_dist - med_dist.min()) / md_range  # 0=best
            iso_range = iso_scores.max() - iso_scores.min() + 1e-12
            iso_norm = (iso_scores - iso_scores.min()) / iso_range  # 1=best
            # Combined: lower is better (invert iso so lower=better)
            combined = 0.7 * md_norm + 0.3 * (1.0 - iso_norm)
        else:
            combined = med_dist

        # Keep the 2 most consistent channels
        n_keep = max(2, n_wfs - 3)
        keep_idx = np.argsort(combined)[:n_keep]
        kept = slopes_multi[keep_idx]

        # Weight by inverse combined score
        c_kept = combined[keep_idx]
        w = 1.0 / (c_kept + 1e-8)
        w = w / (np.sum(w) + 1e-12)
        fused = np.sum(kept * w[:, None], axis=0)
    elif n_wfs == 2:
        if model is not None:
            scores = model.decision_function(slopes_multi)
            fused = slopes_multi[np.argmax(scores)]
        else:
            fused = np.mean(slopes_multi, axis=0)
    elif n_wfs == 1:
        fused = slopes_multi[0]
    else:
        fused = np.mean(slopes_multi, axis=0)

    u = reconstructor @ fused
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
