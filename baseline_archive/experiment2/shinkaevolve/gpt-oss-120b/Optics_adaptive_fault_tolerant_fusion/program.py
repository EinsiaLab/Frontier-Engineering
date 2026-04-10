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
    Consensus-based robust fusion using minimum-pairwise-distance selection.
    
    Core insight: With 3 corruptions out of 5 channels, the two good channels
    should be closest to each other. Uses minimum pairwise distance as primary
    signal, validated by median pairwise distance.
    
    Algorithm:
    1. Compute all pairwise L2 distances between channels
    2. For each channel, compute minimum distance (closest neighbor)
    3. For each channel, compute median distance (consensus with others)
    4. Combine ranks with 0.7/0.3 weighting
    5. Select top 2 channels and apply inverse-score weighting
    """
    n_wfs = slopes_multi.shape[0]

    if n_wfs <= 2:
        fused = np.median(slopes_multi, axis=0)
    else:
        # Compute pairwise L2 distances
        diff = slopes_multi[:, np.newaxis, :] - slopes_multi[np.newaxis, :, :]
        pairwise_dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(pairwise_dists, np.inf)
        
        # Metric 1: Minimum distance (good channels have a close neighbor)
        min_dists = np.min(pairwise_dists, axis=1)
        
        # Metric 2: Median distance (good channels agree with others on average)
        med_dists = np.median(pairwise_dists, axis=1)
        
        # Combine using ranks for scale invariance
        min_rank = np.argsort(np.argsort(min_dists)).astype(float)
        med_rank = np.argsort(np.argsort(med_dists)).astype(float)
        
        # Weight minimum distance more heavily (direct signal for good pair)
        combined = 0.7 * min_rank + 0.3 * med_rank
        
        # Select top 2 channels by combined score
        best_indices = np.argsort(combined)[:2]
        selected = slopes_multi[best_indices]
        
        # Weight by inverse combined score
        selected_scores = combined[best_indices]
        weights = 1.0 / (selected_scores + 1.0)
        weights /= np.sum(weights)
        fused = np.sum(selected * weights[:, np.newaxis], axis=0)

    u = reconstructor @ fused
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END