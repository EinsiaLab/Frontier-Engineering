import json
import numpy as np
from pathlib import Path

# EVOLVE-BLOCK-START
def generate_baseline():
    """
    Generate a heuristic baseline proton therapy plan.
    Strategy: Uniformly place spots within CTV, strictly avoiding the OAR margin.
    """
    spots = []
    
    c_ctv = np.array([0.0, 0.0, 50.0])
    r_ctv = 15.0
    
    c_oar = np.array([0.0, 20.0, 60.0])
    r_oar = 10.0
    
    # Candidate grid inside CTV
    xs = np.arange(-12, 13, 6)
    ys = np.arange(-12, 13, 6)
    zs = np.arange(38, 63, 6)
    
    for x in xs:
        for y in ys:
            for z in zs:
                pos = np.array([x, y, z])
                
                # Check if inside CTV (with slight margin, r=13)
                if np.linalg.norm(pos - c_ctv) <= 13.0:
                    dist_to_oar = np.linalg.norm(pos - c_oar)
                    
                    # Avoid OAR (radius 10 + 6mm safety margin)
                    if dist_to_oar < 16.0:
                        continue
                        
                    # Assign uniform initial weight
                    spots.append({
                        "x": float(round(x, 2)),
                        "y": float(round(y, 2)),
                        "z": float(round(z, 2)),
                        "w": 4.5
                    })
    
    # Max 100 spots
    spots = spots[:100]
    
    return {"spots": spots}
# EVOLVE-BLOCK-END

def _output_path() -> Path:
    # frontier_eval evaluates candidates from a temporary working directory,
    # so the contract is to always write `plan.json` in the current cwd.
    return Path("plan.json")

if __name__ == "__main__":
    plan_data = generate_baseline()
    output_path = _output_path()
    
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(plan_data, f, indent=2)
        print(f"Baseline plan successfully generated: {output_path.as_posix()}")
        print(f"Total spots placed: {len(plan_data['spots'])}")
    except Exception as e:
        print(f"Failed to generate baseline plan: {str(e)}")