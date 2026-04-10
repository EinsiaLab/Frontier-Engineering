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
    Robust fusion by selecting the most similar pair of channels.
    With 5 WFS and 3 corrupted, the two good channels have near-identical
    slopes (differ only by tiny noise); corrupted channels differ wildly.
    This reliably recovers a clean fused slope without sklearn.
    """
    # When majority channels are corrupted (3/5 bad), median is unreliable.
    # Instead find the pair of channels with smallest L2 difference:
    # the two good channels are nearly identical (small sensor noise),
    # while corrupted channels have large random gain/bias/spikes/dropouts.
    # This reliably selects the two good sensors.
    n_wfs = slopes_multi.shape[0]
    min_dist = np.inf
    best_i, best_j = 0, 1
    for i in range(n_wfs):
        for j in range(i + 1, n_wfs):
            dist = np.sum((slopes_multi[i] - slopes_multi[j]) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_i, best_j = i, j
    fused = np.mean(slopes_multi[[best_i, best_j]], axis=0)

    u = reconstructor @ fused
    return np.clip(u, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
