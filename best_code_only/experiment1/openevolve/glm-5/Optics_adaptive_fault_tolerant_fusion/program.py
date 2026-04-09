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
    Iterative consensus fusion: refine sensor selection through agreement.
    Uses pairwise consistency to identify the clean sensor cluster.
    """
    model = control_model.get("anomaly_model")
    n_sensors = slopes_multi.shape[0]
    
    # Vectorized correlation matrix computation
    slopes_centered = slopes_multi - slopes_multi.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(slopes_centered, axis=1) + 1e-12
    slopes_norm = slopes_centered / norms[:, None]
    corr_matrix = slopes_norm @ slopes_norm.T
    np.fill_diagonal(corr_matrix, 0)
    
    # Consensus score = mean absolute correlation (robust to sign)
    consensus = np.abs(corr_matrix).mean(axis=1)
    
    if model is not None:
        scores = model.decision_function(slopes_multi)
        s_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        c_norm = (consensus - consensus.min()) / (consensus.max() - consensus.min() + 1e-12)
        quality = 0.5 * s_norm + 0.5 * c_norm
    else:
        quality = consensus
    
    # Direct selection of top 2 sensors
    keep_idx = np.argsort(quality)[-2:]
    
    # Weight by quality scores
    w = quality[keep_idx]
    w = np.maximum(w, 0.01)
    w /= w.sum()
    fused = np.sum(slopes_multi[keep_idx] * w[:, None], axis=0)
    
    return np.clip(reconstructor @ fused, -max_voltage, max_voltage)
# EVOLVE-BLOCK-END
