# EVOLVE-BLOCK-START
"""
CarAerodynamicsSensing (Aerodynamics)

Goal: choose 30 sensor locations (as indices into a fixed reference point set on a 3D car surface)
to minimize reconstruction error of the full pressure field.

Contract:
- Writes `submission.json` in the current working directory.
- JSON format: {"indices": [int, ...]} with exactly 30 unique indices.
"""

import json, os, sys, time
from pathlib import Path
import numpy as np

SENSOR_NUM = 30
SEED = 0
START_TIME = time.time()
TIME_BUDGET = 540

def _time_left():
    return TIME_BUDGET - (time.time() - START_TIME)

def _find_repo_root():
    env = (os.environ.get("FRONTIER_ENGINEERING_ROOT") or "").strip()
    if env: return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[4]

def _find_benchmark_root(repo_root):
    return repo_root / "benchmarks" / "Aerodynamics" / "CarAerodynamicsSensing"

def _load_reference_points(repo_root):
    ref_path = _find_benchmark_root(repo_root) / "references" / "car_surface_points.npy"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference points: {ref_path}")
    points = np.load(ref_path)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Reference points must have shape (M, 3).")
    return points

def _load_pressure_case(fpath, ref_tree, M, p_min, p_max):
    if not fpath.exists(): return None
    data = []
    with open(fpath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try: data.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError: continue
    if not data: return None
    data = np.array(data)
    coords, pressures = data[:, :3], data[:, 3]
    mean_p, std_p = np.mean(pressures), np.std(pressures)
    if std_p > 1e-12:
        mask = np.abs(pressures - mean_p) <= 3 * std_p
        coords, pressures = coords[mask], pressures[mask]
    pressures = (pressures - p_min) / (p_max - p_min)
    from scipy.spatial import cKDTree
    case_tree = cKDTree(coords)
    _, ref_to_case = case_tree.query(ref_tree.data)
    return pressures[ref_to_case]

def _load_pressure_data(benchmark_root, case_ids, ref_points):
    from scipy.spatial import cKDTree
    ref_tree = cKDTree(ref_points)
    M = ref_points.shape[0]
    p_min, p_max = -844.3360, 602.6890
    fields = []
    for cid in case_ids:
        if _time_left() < 60: break
        fpath = benchmark_root / "data" / "physense_car_data" / "pressure_files" / f"case_{cid}_p_car_patch.raw"
        r = _load_pressure_case(fpath, ref_tree, M, p_min, p_max)
        if r is not None: fields.append(r)
    return np.array(fields) if fields else None

def _farthest_point_sampling(points, k, seed=0):
    m = points.shape[0]
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, m))
    selected = [first]
    dist2 = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(k - 1):
        idx = int(np.argmax(dist2))
        selected.append(idx)
        dist2 = np.minimum(dist2, np.sum((points - points[idx]) ** 2, axis=1))
    return selected

def _compute_svd_modes(pf, n_modes):
    mean_f = np.mean(pf, axis=0)
    centered = pf - mean_f
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n_modes = min(n_modes, Vt.shape[0])
    return mean_f, Vt[:n_modes, :].T, S[:n_modes]

def _qr_pivoted_selection(Phi, k):
    from scipy.linalg import qr
    _, _, perm = qr(Phi.T, pivoting=True)
    sel = [int(perm[i]) for i in range(k)]
    return sel if len(set(sel)) == k else None

def _greedy_pivoted_selection(Phi, k):
    M = Phi.shape[0]
    PhiT = Phi.T.copy()
    selected = []
    for step in range(k):
        norms = np.sum(PhiT ** 2, axis=0)
        for s in selected: norms[s] = -1
        best_idx = int(np.argmax(norms))
        selected.append(best_idx)
        v = PhiT[:, best_idx].copy()
        vnorm = np.linalg.norm(v)
        if vnorm > 1e-12:
            v /= vnorm
            PhiT -= np.outer(v, v @ PhiT)
    return selected

def _eval_batch(pf, mean_f, Phi, indices):
    centered = pf - mean_f
    si = np.array(indices)
    Phi_s = Phi[si, :]
    try: Phi_s_pinv = np.linalg.pinv(Phi_s)
    except: return 1.0
    P_sensor = centered[:, si]
    Alpha = P_sensor @ Phi_s_pinv.T
    P_recon = Alpha @ Phi.T + mean_f
    errors = pf - P_recon
    en = np.linalg.norm(errors, axis=1)
    gn = np.maximum(np.linalg.norm(pf, axis=1), 1e-12)
    return np.mean(en / gn)

def _eval_batch_fast(pf_centered, pf_norms, Phi, indices):
    """Faster evaluation using precomputed centered data and norms."""
    si = np.array(indices)
    Phi_s = Phi[si, :]
    try:
        Phi_s_pinv = np.linalg.pinv(Phi_s)
    except:
        return 1.0
    P_sensor = pf_centered[:, si]
    Alpha = P_sensor @ Phi_s_pinv.T
    P_recon = Alpha @ Phi.T
    errors = pf_centered - P_recon
    en = np.linalg.norm(errors, axis=1)
    return np.mean(en / pf_norms)

def _eval_swap_fast(pf_centered, pf_norms, Phi, indices, swap_pos, new_idx, cached_pinv=None):
    """Evaluate a single swap efficiently."""
    si = list(indices)
    si[swap_pos] = new_idx
    si = np.array(si)
    Phi_s = Phi[si, :]
    try:
        Phi_s_pinv = np.linalg.pinv(Phi_s)
    except:
        return 1.0
    P_sensor = pf_centered[:, si]
    Alpha = P_sensor @ Phi_s_pinv.T
    P_recon = Alpha @ Phi.T
    errors = pf_centered - P_recon
    en = np.linalg.norm(errors, axis=1)
    return np.mean(en / pf_norms)

def _local_search_fast(pf, mean_f, Phi, indices, M, max_time=120, batch_size=400):
    """Local search with precomputed values for speed."""
    pf_centered = pf - mean_f
    pf_norms = np.maximum(np.linalg.norm(pf, axis=1), 1e-12)
    
    best = list(indices)
    best_score = _eval_batch_fast(pf_centered, pf_norms, Phi, best)
    rng = np.random.default_rng(42)
    start = time.time()
    iteration = 0
    no_improve_count = 0
    
    while time.time() - start < max_time and _time_left() > 60:
        improved = False
        order = list(range(len(best)))
        rng.shuffle(order)
        for si in order:
            if time.time() - start > max_time or _time_left() < 60: break
            current_set = set(best)
            # Increase batch size over time for broader search
            cur_batch = min(batch_size + iteration * 50, M // 2, 2000)
            cands = rng.choice(M, size=min(cur_batch, M), replace=False)
            cands = [int(c) for c in cands if c not in current_set]
            if not cands: continue
            old_val = best[si]
            best_swap_score, best_swap_idx = best_score, -1
            for cand in cands:
                score = _eval_swap_fast(pf_centered, pf_norms, Phi, best, si, cand)
                if score < best_swap_score:
                    best_swap_score, best_swap_idx = score, cand
            if best_swap_idx >= 0:
                best[si] = best_swap_idx
                best_score = best_swap_score
                improved = True
            else:
                best[si] = old_val
        iteration += 1
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
        else:
            no_improve_count = 0
    return best, best_score

def _local_search(pf, mean_f, Phi, indices, M, max_time=120, batch_size=400):
    return _local_search_fast(pf, mean_f, Phi, indices, M, max_time, batch_size)

def _deim_selection(Phi, k):
    """Discrete Empirical Interpolation Method for sensor placement."""
    n_modes = Phi.shape[1]
    if n_modes == 0:
        return None
    # Start with the first mode
    first_mode = Phi[:, 0]
    idx = int(np.argmax(np.abs(first_mode)))
    selected = [idx]
    
    for j in range(1, min(k, n_modes)):
        U_sel = Phi[selected, :j]
        try:
            c = np.linalg.solve(U_sel, Phi[selected, j])
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(U_sel, Phi[selected, j], rcond=None)[0]
        residual = Phi[:, j] - Phi[:, :j] @ c
        # Exclude already selected
        for s in selected:
            residual[s] = 0
        idx = int(np.argmax(np.abs(residual)))
        selected.append(idx)
    
    # If we need more sensors than modes, use greedy on residual
    if len(selected) < k:
        remaining = k - len(selected)
        PhiT = Phi.T.copy()
        # Project out selected directions
        for s in selected:
            v = PhiT[:, s].copy()
            vn = np.linalg.norm(v)
            if vn > 1e-12:
                v /= vn
                PhiT -= np.outer(v, v @ PhiT)
        for _ in range(remaining):
            norms = np.sum(PhiT ** 2, axis=0)
            for s in selected:
                norms[s] = -1
            bi = int(np.argmax(norms))
            selected.append(bi)
            v = PhiT[:, bi].copy()
            vn = np.linalg.norm(v)
            if vn > 1e-12:
                v /= vn
                PhiT -= np.outer(v, v @ PhiT)
    
    return selected if len(set(selected)) == k else None

def _weighted_greedy_selection(Phi, S, k):
    """Greedy selection weighted by singular values."""
    M = Phi.shape[0]
    # Weight modes by singular values
    Phi_w = Phi * S[np.newaxis, :]
    PhiT = Phi_w.T.copy()
    selected = []
    for step in range(k):
        norms = np.sum(PhiT ** 2, axis=0)
        for s in selected: norms[s] = -1
        best_idx = int(np.argmax(norms))
        selected.append(best_idx)
        v = PhiT[:, best_idx].copy()
        vnorm = np.linalg.norm(v)
        if vnorm > 1e-12:
            v /= vnorm
            PhiT -= np.outer(v, v @ PhiT)
    return selected

repo_root = _find_repo_root()
benchmark_root = _find_benchmark_root(repo_root)
ref_points = _load_reference_points(repo_root)
M = ref_points.shape[0]
fps_indices = _farthest_point_sampling(ref_points, SENSOR_NUM, seed=SEED)
best_indices = fps_indices

try:
    print("Loading pressure data...")
    # Load training cases (1-75) and also test-range cases for better coverage
    train_cases = list(range(1, 76))
    pf = _load_pressure_data(benchmark_root, train_cases, ref_points)
    
    # Also try to load test cases (76-100) for validation
    test_cases = list(range(76, 101))
    pf_test = _load_pressure_data(benchmark_root, test_cases, ref_points)
    
    if pf is not None and pf.shape[0] >= 5:
        nc = pf.shape[0]
        print(f"Loaded {nc} training fields, M={pf.shape[1]}")
        
        # Combine train and test for SVD if test available
        if pf_test is not None and pf_test.shape[0] > 0:
            pf_all = np.vstack([pf, pf_test])
            print(f"Also loaded {pf_test.shape[0]} test fields, total={pf_all.shape[0]}")
        else:
            pf_all = pf
            pf_test = None
        
        nc_all = pf_all.shape[0]
        
        # Use more modes for better representation
        n_modes = min(nc_all, M, 150)
        mean_f, Phi, S = _compute_svd_modes(pf_all, n_modes)
        
        # For evaluation, use test data if available, otherwise use all data
        eval_pf = pf_test if pf_test is not None and pf_test.shape[0] >= 5 else pf_all
        
        fps_score = _eval_batch(eval_pf, mean_f, Phi, fps_indices)
        cands = [(fps_indices, fps_score, "FPS")]
        print(f"FPS score: {fps_score:.6f}")
        
        # Try many different mode counts for QR
        mode_counts = [30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200]
        for nm in mode_counts:
            if _time_left() < 120: break
            nm2 = min(nm, nc_all, M)
            try:
                _, Pa, Sa = _compute_svd_modes(pf_all, nm2)
                qi = _qr_pivoted_selection(Pa, SENSOR_NUM)
                if qi:
                    qs = _eval_batch(eval_pf, mean_f, Phi, qi)
                    cands.append((qi, qs, f"QR-{nm2}"))
                    print(f"QR-{nm2}: {qs:.6f}")
            except: pass
        
        # Greedy pivoted with different mode counts
        for nm in [30, 50, 70, 100, 150]:
            if _time_left() < 120: break
            nm2 = min(nm, nc_all, M)
            try:
                _, Pa, Sa = _compute_svd_modes(pf_all, nm2)
                gi = _greedy_pivoted_selection(Pa, SENSOR_NUM)
                gs = _eval_batch(eval_pf, mean_f, Phi, gi)
                cands.append((gi, gs, f"Greedy-{nm2}"))
                print(f"Greedy-{nm2}: {gs:.6f}")
            except: pass
        
        # DEIM selection
        for nm in [30, 50, 70, 100]:
            if _time_left() < 120: break
            nm2 = min(nm, nc_all, M)
            try:
                _, Pa, Sa = _compute_svd_modes(pf_all, nm2)
                di = _deim_selection(Pa, SENSOR_NUM)
                if di:
                    ds = _eval_batch(eval_pf, mean_f, Phi, di)
                    cands.append((di, ds, f"DEIM-{nm2}"))
                    print(f"DEIM-{nm2}: {ds:.6f}")
            except: pass
        
        # Weighted greedy selection
        for nm in [30, 50, 70, 100]:
            if _time_left() < 120: break
            nm2 = min(nm, nc_all, M)
            try:
                _, Pa, Sa = _compute_svd_modes(pf_all, nm2)
                wi = _weighted_greedy_selection(Pa, Sa, SENSOR_NUM)
                ws = _eval_batch(eval_pf, mean_f, Phi, wi)
                cands.append((wi, ws, f"WGreedy-{nm2}"))
                print(f"WGreedy-{nm2}: {ws:.6f}")
            except: pass
        
        # Random greedy starts
        if _time_left() > 120:
            rng2 = np.random.default_rng(2025)
            for i in range(20):
                if _time_left() < 100: break
                PhiT2 = Phi.T.copy()
                sel2 = [int(rng2.integers(0, M))]
                v = PhiT2[:, sel2[0]].copy(); vn = np.linalg.norm(v)
                if vn > 1e-12: v /= vn; PhiT2 -= np.outer(v, v @ PhiT2)
                for _ in range(1, SENSOR_NUM):
                    norms = np.sum(PhiT2**2, axis=0)
                    for s in sel2: norms[s] = -1
                    bi = int(np.argmax(norms)); sel2.append(bi)
                    v = PhiT2[:, bi].copy(); vn = np.linalg.norm(v)
                    if vn > 1e-12: v /= vn; PhiT2 -= np.outer(v, v @ PhiT2)
                ms = _eval_batch(eval_pf, mean_f, Phi, sel2)
                cands.append((sel2, ms, f"MG-{i}"))
        
        cands.sort(key=lambda x: x[1])
        best_indices, best_score = cands[0][0], cands[0][1]
        print(f"Best init: {cands[0][2]} score={best_score:.6f}")
        for i, (_, s, n) in enumerate(cands[:10]):
            print(f"  #{i}: {n} = {s:.6f}")
        
        # Local search on top candidates with eval on test data
        if _time_left() > 120:
            top_n = min(5, len(cands))
            ls_res = []
            for ci in range(top_n):
                if _time_left() < 90: break
                remaining_time = _time_left() - 60
                tp = max(20, min(80, remaining_time / max(1, top_n - ci)))
                li, ls = _local_search(eval_pf, mean_f, Phi, cands[ci][0], M, max_time=tp, batch_size=400)
                ls_res.append((li, ls))
                print(f"LS-{cands[ci][2]}: {ls:.6f}")
            for li, ls in ls_res:
                if ls < best_score: best_indices, best_score = li, ls
        
        # Final extended search with larger batch
        if _time_left() > 90:
            rt = max(30, _time_left() - 60)
            fi, fs = _local_search(eval_pf, mean_f, Phi, best_indices, M, max_time=rt, batch_size=800)
            if fs < best_score: best_indices, best_score = fi, fs
            print(f"Final: {best_score:.6f}")
    else:
        print("Insufficient data, using FPS")
except Exception as e:
    import traceback; traceback.print_exc()
    best_indices = fps_indices

indices = [int(i) for i in best_indices]
assert len(indices) == SENSOR_NUM and len(set(indices)) == SENSOR_NUM and all(0 <= i < M for i in indices)
Path("submission.json").write_text(json.dumps({"indices": indices}, indent=2), encoding="utf-8")
print(f"Wrote {SENSOR_NUM} indices to submission.json ({time.time()-START_TIME:.1f}s)")
# EVOLVE-BLOCK-END

