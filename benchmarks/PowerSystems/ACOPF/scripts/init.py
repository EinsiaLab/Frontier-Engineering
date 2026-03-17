"""Initialize 5-bus DC-OPF case data. No side effects."""

# Minimal 5-bus DC-OPF case (embedded for self-contained eval)
# B matrix (pu), P_load (pu), gen at buses 0,1,2 with limits and cost c0+c1*Pg+c2*Pg^2
def get_instance():
    import numpy as np
    # 5x5 B (singular; slack at 0)
    B = np.array([
        [-3.0,  1.0,  1.0,  0.0,  0.0],
        [ 1.0, -2.0,  0.0,  1.0,  0.0],
        [ 1.0,  0.0, -2.0,  0.0,  1.0],
        [ 0.0,  1.0,  0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0,  0.0, -1.0],
    ], dtype=float)
    P_load = np.array([0.0, 0.5, 0.8, 0.0, 0.0])  # load at bus 1,2
    Pgen_min = np.array([0.0, 0.0, 0.0])
    Pgen_max = np.array([2.0, 2.0, 2.0])
    gen_bus = [0, 1, 2]  # 3 gens
    cost_c0 = [0.0, 0.0, 0.0]
    cost_c1 = [20.0, 30.0, 25.0]
    cost_c2 = [0.1, 0.15, 0.12]
    return {
        "n_bus": 5,
        "B": B.tolist(),
        "P_load": P_load.tolist(),
        "Pgen_min": Pgen_min.tolist(),
        "Pgen_max": Pgen_max.tolist(),
        "cost_c0": cost_c0,
        "cost_c1": cost_c1,
        "cost_c2": cost_c2,
        "gen_bus": gen_bus,
    }
