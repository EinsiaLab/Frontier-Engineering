# Car Aerodynamics Sensing

Select 30 sensor locations on a fixed 3D car surface to minimize the reconstruction error of the full pressure field.

## Files
- `Task.md`: full task description and rules
- `references/`: reference point set and helper script
- `verification/`: evaluator and environment files
- `baseline/`: random sampling baseline

## Quickstart
0) Ensure the PhySense repo (model code) is available (required for evaluation):
- Default: `third_party/PhySense/Car-Aerodynamics/`
- Alternatives:
  - `<workspace>/PhySense/Car-Aerodynamics/` (next to `Frontier-Engineering/`)
  - set `PHYSENSE_ROOT=/path/to/PhySense` (or to the `Car-Aerodynamics/` folder)

1) Download dataset and pretrained model from PhySense:
- Dataset: https://drive.google.com/drive/folders/1Xt395uC-RreWQ_bh5v_Bu-AOwV6SRt_w?usp=share_link
- Pretrained models: https://drive.google.com/drive/folders/1kODTzH5n_njQ7lgL-2i2aXbt33UjGKk9?usp=sharing

2) Place the files in the **fixed paths** below (relative to this task directory):
- Dataset root: `data/physense_car_data/`
  - must contain `pressure_files/` and `velocity_files/`
- Base model checkpoint: `data/physense_car_ckpt/physense_transolver_car_base.pth` (or `data/physense_car_ckpt/physense_transolver_car_best_base.pth`)

3) Generate the reference point set:

```bash
python references/extract_car_mesh.py \
  --data-dir data/physense_car_data \
  --output references/car_surface_points.npy
```

4) Create a baseline submission:

```bash
python baseline/solution.py --output submission.json
```

5) Evaluate:

```bash
python verification/evaluator.py --submission submission.json
```

## Docker (build context)
Build with the `verification/` directory as the context:

```bash
cd verification
docker build -f docker/Dockerfile -t car-aero-eval .
```

## Notes
- GPU is required. The evaluator uses CUDA and will fail on CPU.
- The evaluator samples K=10 cases from case_76 to case_100 with seed 2025.
