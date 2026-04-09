# EVOLVE-BLOCK-START
"""Baseline for Task 4: spectrum packing with guard bands."""

from __future__ import annotations

import numpy as np


def pack_spectrum(user_demand_slots, n_slots, guard_slots=1, seed=0):
    """First-Fit Decreasing baseline.

    Returns
    -------
    dict with key `alloc`:
      alloc[i] = (start_slot, width)
      if not allocated: (-1, 0)
    """
    d = np.asarray(user_demand_slots, dtype=int)
    n_users = d.size

    alloc = [(-1, 0) for _ in range(n_users)]

    # Sort by increasing demand (smallest first) - this helps pack more requests
    order = np.argsort(d)  # smaller requests first
    
    occupied = np.zeros(n_slots, dtype=bool)
    
    for u in order:
        width = int(d[u])
        if width <= 0 or width > n_slots:
            continue

        # Find all valid positions
        valid_positions = []
        for s in range(0, n_slots - width + 1):
            left = max(0, s - guard_slots)
            right = min(n_slots, s + width + guard_slots)
            if not np.any(occupied[left:right]):
                valid_positions.append(s)
        
        if not valid_positions:
            alloc[u] = (-1, 0)
            continue
        
        # Choose best position using a scoring function that considers:
        # 1. Fragmentation (number of free blocks) - affects compactness (5%)
        # 2. Largest free block size - affects acceptance of large requests (80%)
        # 3. Distance to nearest allocation - affects BER pass ratio (10%)
        best_position = valid_positions[0]
        best_score = float('inf')
        
        for s in valid_positions:
            # Temporarily place the request
            temp_occupied = occupied.copy()
            temp_occupied[s:s + width] = True
            
            # Count number of free blocks (fragmentation)
            free_blocks = 0
            in_free = False
            for i in range(n_slots):
                if not temp_occupied[i] and not in_free:
                    free_blocks += 1
                    in_free = True
                elif temp_occupied[i]:
                    in_free = False
            
            # Find the largest free block size after placement
            max_free_block = 0
            current_block = 0
            for i in range(n_slots):
                if not temp_occupied[i]:
                    current_block += 1
                    if current_block > max_free_block:
                        max_free_block = current_block
                else:
                    current_block = 0
            
            # Calculate distance to nearest occupied slot in original occupied
            # (excluding current temporary placement)
            if np.any(occupied):
                # Find indices of occupied slots
                occupied_indices = np.where(occupied)[0]
                # Compute distances from the center of this placement to each occupied slot
                center = s + width / 2.0
                distances = np.abs(center - occupied_indices)
                min_distance = np.min(distances) if len(distances) > 0 else n_slots
            else:
                min_distance = n_slots  # if no other allocations, this is good
            
            # Score aligned with fitness weights: 0.80*acceptance + 0.05*compactness + 0.10*ber_pass
            # We want to maximize acceptance, which is helped by large max_free_block
            # So we give positive weight to max_free_block, negative to free_blocks and min_distance
            # Since we're minimizing, we use:
            score = 0.05 * free_blocks - 0.80 * max_free_block - 0.10 * min_distance
            
            if score < best_score:
                best_score = score
                best_position = s
        
        # Place at the best position found
        occupied[best_position:best_position + width] = True
        alloc[u] = (best_position, width)

    return {"alloc": np.asarray(alloc, dtype=int)}
# EVOLVE-BLOCK-END
