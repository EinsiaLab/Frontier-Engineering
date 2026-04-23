import json
import sys
import math
from pathlib import Path

def evaluate(solution_path: Path) -> dict:
    if not solution_path.exists():
        return {"status": "failed", "message": f"Solution file not found: {solution_path}"}
        
    try:
        with open(solution_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {"status": "failed", "message": f"Failed to parse JSON: {e}"}

    
    if not isinstance(data, list):
        return {"status": "failed", "message": "JSON must be a list of rings (dictionaries)."}

    num_rings = len(data)
    if num_rings == 0:
        return {"status": "failed", "message": "Ring array is empty."}

    
    ring_width = 10.0
    
    total_volume = 0.0
    total_sensitivity = 0.0
    total_resolution_gamma = 0.0
    out_of_bounds_penalty = 0.0

    for i, ring in enumerate(data):
        try:
            R = float(ring.get("R", 400.0))
            H = float(ring.get("H", 10.0))
            W = float(ring.get("W", 4.0))
        except ValueError:
            return {"status": "failed", "message": f"Invalid variable types in ring {i}."}

       
        if not (300 <= R <= 500): out_of_bounds_penalty += 200
        if not (10 <= H <= 30): out_of_bounds_penalty += 200
        if not (2 <= W <= 6): out_of_bounds_penalty += 200

       
        total_volume += math.pi * ((R + H)**2 - R**2) * ring_width
        
       
        z_pos = (i - num_rings / 2.0 + 0.5) * ring_width
        distance = math.sqrt(R**2 + z_pos**2)
        
        solid_angle_factor = ring_width / distance
        stopping_power = (1.0 - math.exp(-0.087 * H))**2
        total_sensitivity += solid_angle_factor * stopping_power
        
       
        gamma = math.sqrt(W**2 + (200.0 * H / R)**2)
        total_resolution_gamma += gamma

   
    avg_resolution_gamma = total_resolution_gamma / num_rings

    
    MAX_VOLUME = 15000000.0
    
    cost_penalty = 0.0
    if total_volume > MAX_VOLUME:
        cost_penalty = (total_volume - MAX_VOLUME) * 0.002
        
    sensitivity_score = total_sensitivity * 20000.0
    resolution_penalty = avg_resolution_gamma * 500.0
    
    total_score = sensitivity_score - resolution_penalty - cost_penalty - out_of_bounds_penalty

    return {
        "status": "success",
        "score": total_score,
        "metrics": {
            "volume_mm3": total_volume,
            "sensitivity_factor": total_sensitivity,
            "resolution_gamma": avg_resolution_gamma,
            "cost_penalty": cost_penalty
        }
    }

if __name__ == "__main__":
    target_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("solution.json")
    result = evaluate(target_file)
    print(json.dumps(result))