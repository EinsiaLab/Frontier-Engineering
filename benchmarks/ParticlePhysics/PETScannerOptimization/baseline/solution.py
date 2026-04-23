import json
from pathlib import Path

# EVOLVE-BLOCK-START
def generate_scanner_design():
    """
    Generate an algorithmic heuristic for a non-uniform PET scanner.
    The scanner consists of 20 rings along the axial direction.
    
    Strategy: A naive baseline using a uniform allocation strategy.
    We use very thin crystals (10mm) to strictly avoid the volume budget 
    limit, at the severe expense of gamma-ray sensitivity.
    """
    num_rings = 20
    design = []
    
    for i in range(num_rings):
        # The AI agent should evolve this logic to allocate thicker crystals 
        # (larger H) to central rings (which have higher solid angle coverage)
        # while keeping edge rings thin to save LYSO volume budget.
        design.append({
            "ring_id": i,
            "R": 400.0,
            "H": 10.0,
            "W": 4.0
        })
        
    return design
# EVOLVE-BLOCK-END

def _output_path() -> Path:
    return Path("solution.json")

if __name__ == "__main__":
    design_data = generate_scanner_design()
    output_path = _output_path()
    
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(design_data, f, indent=2)
        print(f"Baseline algorithm successfully executed. Output saved to: {output_path.as_posix()}")
    except Exception as e:
        print(f"Failed to generate baseline design: {str(e)}")