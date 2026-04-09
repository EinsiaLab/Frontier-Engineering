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
    Robust sensor fusion using pairwise agreement + multi-criteria scoring.

    Key idea: with 5 channels and 3 corrupted, the 2 good channels agree
    with each other far more than with corrupted ones. We use pairwise
    distances to identify the most mutually-consistent subset.
    """
    n_wfs = slopes_multi.shape[0]

    if n_wfs == 1:
        fused = slopes_multi[0]
        u = reconstructor @ fused
        return np.clip(u, -max_voltage, max_voltage)

    if n_wfs == 2:
        fused = np.mean(slopes_multi, axis=0)
        u = reconstructor @ fused
        return np.clip(u, -max_voltage, max_voltage)

    # === Dual-space robust fusion ===
    # Work in both slope space and command space for better outlier detection

    # Project all channels to command space
    commands_all = slopes_multi @ reconstructor.T  # (n_wfs, n_act)

    # --- Step 1: Geometric median in command space (IRLS) ---
    # Initialize with coordinate-wise median
    gm_cmd = np.median(commands_all, axis=0)

    for _ in range(15):
        diffs = commands_all - gm_cmd[np.newaxis, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        dists = np.maximum(dists, 1e-10)
        weights_gm = 1.0 / dists
        weights_gm /= np.sum(weights_gm)
        gm_cmd_new = weights_gm @ commands_all
        if np.sum((gm_cmd_new - gm_cmd) ** 2) < 1e-16:
            gm_cmd = gm_cmd_new
            break
        gm_cmd = gm_cmd_new

    # --- Step 2: Geometric median in slope space ---
    gm_slope = np.median(slopes_multi, axis=0)

    for _ in range(15):
        diffs = slopes_multi - gm_slope[np.newaxis, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        dists = np.maximum(dists, 1e-10)
        weights_gm = 1.0 / dists
        weights_gm /= np.sum(weights_gm)
        gm_slope_new = weights_gm @ slopes_multi
        if np.sum((gm_slope_new - gm_slope) ** 2) < 1e-16:
            gm_slope = gm_slope_new
            break
        gm_slope = gm_slope_new

    # --- Step 3: Compute distances in both spaces ---
    dists_cmd = np.sqrt(np.sum((commands_all - gm_cmd[np.newaxis, :]) ** 2, axis=1))
    dists_slope = np.sqrt(np.sum((slopes_multi - gm_slope[np.newaxis, :]) ** 2, axis=1))

    # Normalize distances
    max_dc = np.max(dists_cmd) + 1e-12
    max_ds = np.max(dists_slope) + 1e-12
    norm_dists = 0.5 * (dists_cmd / max_dc) + 0.5 * (dists_slope / max_ds)

    # --- Step 4: Find best pair for anchoring ---
    norms_sq = np.sum(slopes_multi ** 2, axis=1)
    dot_products = slopes_multi @ slopes_multi.T
    pw_dists_sq = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - 2.0 * dot_products
    np.maximum(pw_dists_sq, 0.0, out=pw_dists_sq)
    np.fill_diagonal(pw_dists_sq, np.inf)

    flat_idx = np.argmin(pw_dists_sq)
    best_i, best_j = divmod(flat_idx, n_wfs)

    # --- Step 5: Select inliers - keep channels closest to geometric median ---
    # Sort by combined normalized distance
    sorted_idx = np.argsort(norm_dists)

    # Always keep at least the best pair; keep channels that are
    # significantly closer than the worst ones
    # With 5 channels, 3 corrupted: keep top 2 (or maybe 3 if one is close)
    n_keep_min = 2

    # Adaptive threshold: gap detection
    sorted_dists = norm_dists[sorted_idx]

    # Find the biggest gap in sorted distances
    n_keep = n_keep_min
    if n_wfs > 2:
        gaps = np.diff(sorted_dists)
        # Look for a significant gap after the first 2 channels
        for g_idx in range(1, len(gaps)):
            # If gap is large relative to the distances so far, cut here
            if gaps[g_idx] > 0.15 and g_idx >= n_keep_min - 1:
                n_keep = g_idx + 1
                break
        else:
            # No big gap found - check if all channels are similar
            if sorted_dists[-1] - sorted_dists[0] < 0.1:
                n_keep = n_wfs  # all channels seem OK
            else:
                n_keep = n_keep_min

    # Ensure best pair is included
    inlier_set_sorted = set(sorted_idx[:n_keep].tolist())
    inlier_set_sorted.add(best_i)
    inlier_set_sorted.add(best_j)
    inlier_indices = np.array(sorted(inlier_set_sorted))

    inlier_slopes = slopes_multi[inlier_indices]
    inlier_dists = norm_dists[inlier_indices]
    n_keep = len(inlier_indices)

    # --- Step 6: Weighted fusion with exponential weighting ---
    if n_keep == 1:
        fused = inlier_slopes[0]
    else:
        # Exponential weighting: channels closer to geometric median get much more weight
        # Use negative distance as score with temperature scaling
        scores = -inlier_dists
        temp = 0.05  # sharp weighting
        exp_s = np.exp((scores - np.max(scores)) / temp)
        w = exp_s / np.sum(exp_s)
        fused = w @ inlier_slopes

    # --- Step 7: Iterative refinement toward geometric median of inliers ---
    for _ in range(3):
        diffs = inlier_slopes - fused[np.newaxis, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        dists = np.maximum(dists, 1e-10)
        w_ref = 1.0 / dists
        w_ref /= np.sum(w_ref)
        fused = w_ref @ inlier_slopes

    # === Compute DM commands ===
    u = reconstructor @ fused

    # Optional: leaky integrator if prev_commands available
    gain = control_model.get("gain", 0.3) if control_model else 0.3
    leak = control_model.get("leak", 0.99) if control_model else 0.99

    if prev_commands is not None:
        u = leak * prev_commands + gain * u

    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END