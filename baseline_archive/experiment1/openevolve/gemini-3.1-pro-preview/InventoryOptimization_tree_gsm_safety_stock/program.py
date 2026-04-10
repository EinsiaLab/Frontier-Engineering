# EVOLVE-BLOCK-START
"""Task 01 solver."""

def solve(_unused=None) -> dict[int, int]:
    """Return the optimal CST policy using a dictionary comprehension."""
    return {n: (1 if n == 4 else 0) for n in (1, 2, 3, 4)}
# EVOLVE-BLOCK-END
