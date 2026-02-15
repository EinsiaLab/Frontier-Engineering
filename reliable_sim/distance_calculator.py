import numpy as np
from itertools import permutations

def sparse_distance_calculator(xs, y_idx):
    """
    Calculate the distance between vector x and a set of sparse vectors y.
    y is 1 in the dimensions indicated by y_idx, and 0 elsewhere.
    (x-y)^2 = (x^2 - 2xy + y^2) = x^2 - 2\sum_{i\in y_idx} x_i + d
    therefore, it only needs bs*n multiplications instead of bs*n*L.

    Parameters:
    xs ((bs, n) array): The dense vectors.
    y_idx ((L, d) array): The indices of the points.

    Returns:
    (bs, L) array: The distances between the dense vectors and the points.
    """
    is1d = xs.ndim == 1
    if is1d:
        xs = xs[None, :]
    # Get the number of points
    bs, n = xs.shape
    L, d = y_idx.shape
    
    # Calculate the distance
    xs2 = np.sum(xs**2, axis=-1)  # Shape: (bs,)
    xy_sum = np.sum(xs[:, y_idx], axis=-1)  # Shape: (bs, L)
    distances = xs2[:, None] - 2 * xy_sum + d
    
    if is1d:
        return distances[0]
    return distances

def sparse_distance_calculator_permuted(xs, y_idx, d_vec=None):
    """
    Calculate the distance between vector x and a set of sparse vectors y.
    If d_vec is provided: y_idx corresponds to d! vectors, where each vector has 
    a permutation of d_vec assigned to the positions in y_idx, and 0 elsewhere.
    therefore, it only needs bs*(n+L*d!*d) multiplications instead of bs*n*(L*d!).

    Parameters:
    xs ((bs, n) array): The dense vectors.
    y_idx ((L, d) array): The indices of the points.
    d_vec ((d,) array, optional): The vector to permute. If None, uses all 1s.

    Returns:
    (bs, L) or (bs, L*d!) array: The distances between the dense vectors and the points.
    """
    if d_vec is None:
        return sparse_distance_calculator(xs, y_idx)
    
    is1d = xs.ndim == 1
    if is1d:
        xs = xs[None, :]
    
    bs, n = xs.shape
    L, d = y_idx.shape
    
    # New behavior: permutation-based vectors
    d_vec = np.asarray(d_vec)
    assert len(d_vec) == d, f"d_vec length {len(d_vec)} must match y_idx width {d}"
    
    # Generate all permutations of d_vec
    perms = np.array(list(permutations(d_vec)))  # Shape: (d!, d)
    factorial_d = len(perms)
    
    # Calculate xs^2 once
    xs2 = np.sum(xs**2, axis=-1)  # Shape: (bs,)
    
    # Calculate sum of d_vec elements squared once
    d_vec_sum_sq = np.sum(d_vec**2)
    
    # Vectorized calculation for all y_idx at once
    # xs[:, y_idx] has shape (bs, L, d)
    xs_selected = xs[:, y_idx]  # Shape: (bs, L, d)
    
    # Calculate distances for all permutations and all y_idx
    distances = np.zeros((bs, L * factorial_d))
    
    for p_idx, perm in enumerate(perms):
        # Calculate 2 * sum(xs[y_idx] * perm) for all L simultaneously
        # xs_selected * perm[None, None, :] broadcasts to (bs, L, d)
        xy_sum = np.sum(xs_selected * perm[None, None, :], axis=-1)  # Shape: (bs, L)
        # Distance formula: ||xs||^2 - 2*<xs[y_idx], perm> + ||perm||^2
        dist = xs2[:, None] - 2 * xy_sum + d_vec_sum_sq  # Shape: (bs, L)
        # Store distances for this permutation
        distances[:, p_idx::factorial_d] = dist
    
    if is1d:
        return distances[0]
    return distances

def sparse_distance_calculator_4323(xs, y_idx):
    """
    Optimized distance calculator for d_vec = [4/3, 2/3, 2/3].
    Reduces 6 permutations to 3 unique patterns and minimizes multiplications.
    
    The three unique patterns are:
    - Pattern 1: [4/3, 2/3, 2/3] (4/3 at index 0)
    - Pattern 2: [2/3, 4/3, 2/3] (4/3 at index 1) 
    - Pattern 3: [2/3, 2/3, 4/3] (4/3 at index 2)
    
    Distance formula: ||x||² - 2⟨x, y⟩ + ||y||²
    Where ⟨x, y⟩ = 4/3 * x_k + 2/3 * (x_i + x_j) for pattern with 4/3 at position k
    
    Parameters:
    xs ((bs, n) array): The dense vectors.
    y_idx ((L, d) array): The indices of the points, where d=3.
    
    Returns:
    (bs, L*3) array: The distances between the dense vectors and the points.
    """
    is1d = xs.ndim == 1
    if is1d:
        xs = xs[None, :]
    
    bs, n = xs.shape
    L, d = y_idx.shape
    
    assert d == 3, f"This function only works for d=3, got d={d}"
    
    # Calculate xs^2 once
    xs2 = np.sum(xs**2, axis=-1)  # Shape: (bs,)
    
    # Calculate sum of d_vec elements squared: (4/3)^2 + 2*(2/3)^2 = 16/9 + 8/9 = 24/9 = 8/3
    d_vec_sum_sq = (4/3)**2 + 2*(2/3)**2
    
    # Get selected elements: xs[:, y_idx] has shape (bs, L, 3)
    xs_selected = xs[:, y_idx]  # Shape: (bs, L, 3)
    
    # Calculate distances for all 3 patterns
    distances = np.zeros((bs, L * 3))
    
    pattern_sum = np.sum(xs_selected, axis=-1)  # Shape: (bs, L)
    pattern1_sum = pattern_sum + xs_selected[:, :, 0]  # 4/3 at index 0
    pattern2_sum = pattern_sum + xs_selected[:, :, 1]  # 4/3 at index 1
    pattern3_sum = pattern_sum + xs_selected[:, :, 2]  # 4/3 at index 2

    dist_1 = xs2[:, None] - 4/3 * pattern1_sum + d_vec_sum_sq
    distances[:, 0::3] = dist_1

    dist_2 = xs2[:, None] - 4/3 * pattern2_sum + d_vec_sum_sq
    distances[:, 1::3] = dist_2
    
    dist_3 = xs2[:, None] - 4/3 * pattern3_sum + d_vec_sum_sq
    distances[:, 2::3] = dist_3
    
    if is1d:
        return distances[0]
    return distances

if __name__ == "__main__":
    np.set_printoptions(2, suppress=True)
    from code_linear import *
    
    # Test original function
    print("=== Testing original sparse_distance_calculator ===")
    r = 4
    code = HammingCode(r=r)
    neighbors_idx = code.get_nearest_neighbors_idx()
    neighbors = code.get_nearest_neighbors()

    x = np.random.rand(code.dim)  # Use correct dimension
    dist = sparse_distance_calculator(x, neighbors_idx)
    dist2 = np.sum((x[None, :] - neighbors)**2, axis=-1)
    print("Single vector error:", np.max(np.abs(dist-dist2)))

    xs = np.random.rand(2, code.dim)  # Use correct dimension
    dist_batch = sparse_distance_calculator(xs, neighbors_idx)
    dist_batch2 = np.sum((xs[:, None, :] - neighbors[None, :, :])**2, axis=-1)
    print("Batch vector error:", np.max(np.abs(dist_batch - dist_batch2)))

    # Test new permutation function
    print("\n=== Testing sparse_distance_calculator_permuted ===")
    
    # Create test data with valid dimensions
    d = 3
    L = 2
    n = 10
    
    # Create simple y_idx for testing - ensure indices are within bounds
    y_idx = np.array([[0, 1, 2], [3, 4, 5]])  # Shape: (2, 3)
    d_vec = np.array([1.0, 2.0, 3.0])  # Shape: (3,)
    
    # Test single vector
    x = np.random.rand(n)
    dist_perm = sparse_distance_calculator_permuted(x, y_idx, d_vec)
    
    # Verify by manual calculation
    from itertools import permutations
    perms = list(permutations(d_vec))
    
    # Manual calculation for verification
    expected_distances = []
    for l in range(L):
        for perm in perms:
            # Create full vector y
            y = np.zeros(n)
            y[y_idx[l]] = perm
            # Calculate distance
            dist_manual = np.sum((x - y)**2)
            expected_distances.append(dist_manual)
    
    expected_distances = np.array(expected_distances)
    print(f"Single vector permutation error: {np.max(np.abs(dist_perm - expected_distances))}")
    print(f"Expected shape: {expected_distances.shape}, Got shape: {dist_perm.shape}")
    
    # Test batch vectors
    xs = np.random.rand(2, n)
    dist_perm_batch = sparse_distance_calculator_permuted(xs, y_idx, d_vec)
    
    # Manual calculation for batch
    expected_distances_batch = []
    for b in range(2):
        batch_distances = []
        for l in range(L):
            for perm in perms:
                y = np.zeros(n)
                y[y_idx[l]] = perm
                dist_manual = np.sum((xs[b] - y)**2)
                batch_distances.append(dist_manual)
        expected_distances_batch.append(batch_distances)
    
    expected_distances_batch = np.array(expected_distances_batch)
    print(f"Batch vector permutation error: {np.max(np.abs(dist_perm_batch - expected_distances_batch))}")
    print(f"Expected shape: {expected_distances_batch.shape}, Got shape: {dist_perm_batch.shape}")
    
    # Test optimized function
    print("\n=== Testing optimized_distance_calculator_4_3_2_3 ===")
    
    # Test with d=3 and specific d_vec
    d_vec_special = np.array([4/3, 2/3, 2/3])
    
    # Test single vector
    x = np.random.rand(n)
    dist_optimized = sparse_distance_calculator_4323(x, y_idx)

    # Compare with original permuted function (only first 3 patterns)
    dist_original = sparse_distance_calculator_permuted(x, y_idx, d_vec_special)
    # Extract only the 3 unique patterns (indices 0, 1, 2 for each L)
    patterns_to_compare = []
    for l in range(L):
        patterns_to_compare.extend([l*6, l*6+2, l*6+3])  # First 3 permutations for each L
    dist_original_subset = dist_original[patterns_to_compare]
    
    print(f"Optimized vs Original error: {np.max(np.abs(dist_optimized - dist_original_subset))}")
    print(f"Optimized shape: {dist_optimized.shape}, Original subset shape: {dist_original_subset.shape}")
    
    # Test batch vectors
    xs = np.random.rand(2, n)
    dist_optimized_batch = sparse_distance_calculator_4323(xs, y_idx)
    dist_original_batch = sparse_distance_calculator_permuted(xs, y_idx, d_vec_special)
    
    # Extract patterns for batch
    dist_original_batch_subset = dist_original_batch[:, patterns_to_compare]
    
    print(f"Batch optimized vs original error: {np.max(np.abs(dist_optimized_batch - dist_original_batch_subset))}")
    print(f"Batch optimized shape: {dist_optimized_batch.shape}, Original batch subset shape: {dist_original_batch_subset.shape}")
    
    print("\nAll tests completed!")