# Third-Party Notices and License Audit

Last audited: 2026-05-07.

This file records third-party repositories, datasets, checkpoints, and task-defining
open-source packages that are explicitly referenced by benchmark README/Task files
or by the repository asset bootstrap manifest. It is a provenance and notice file;
it does not replace the upstream license texts and it does not establish a license
for Frontier-Engineering itself.

General handling rules:

- Keep upstream source URLs, commit identifiers, and attribution notes with any
  bundled or derived task material.
- Do not vendor an external repository, dataset, checkpoint, or package into this
  repository unless its license text and required notices are added here or next to
  the vendored files.
- For assets that are downloaded at setup time, users must follow the upstream
  provider's license or terms. They are intentionally not redistributed by this
  repository unless noted below.
- Copyleft or share-alike entries are called out explicitly because downstream
  redistribution of those benchmark portions may trigger additional obligations.

## Bundled or Derived Benchmark Sources

| Benchmark path | Upstream source | License | Use in this repository | Required handling |
|---|---|---|---|---|
| `AdditiveManufacturing/DiffSimThermalControl` | [`mojtabamozaffar/differentiable-simulation-am`](https://github.com/mojtabamozaffar/differentiable-simulation-am) | MIT | Bundles upstream `0.k` and `toolpath.crs` in `references/original/` and derives cases in `references/cases.json`. | Retain MIT attribution and the upstream source URL with copied geometry/toolpath assets. |
| `Aerodynamics/DawnAircraftDesignOptimization` | [`peterdsharpe/DawnDesignTool`](https://github.com/peterdsharpe/DawnDesignTool) | MIT | Simplified benchmark inspired by `design_opt.py`; source note is in `references/mission_config.json`. | Keep attribution to DawnDesignTool and the simplified/inspired status clear. |
| `ComputerSystems/DuckDBWorkloadOptimization` | [`duckdb/duckdb`](https://github.com/duckdb/duckdb) at commit `ff4f70eeee83cfd3dae6577fc9b2b448d5fbdb35` | MIT | Copies official benchmark SQL/workload files under `references/duckdb_official/`. | Retain source commit, file list, and MIT attribution for copied benchmark files. |
| `ComputerSystems/MallocLab` | [`PKUanonym/REKCARC-TSC-UHT`](https://github.com/PKUanonym/REKCARC-TSC-UHT) | CC-BY-SA-4.0 | Bundles `malloclab-handout/` files and traces referenced by the task README. | Attribute the source. Treat adapted MallocLab handout materials as CC-BY-SA-4.0 share-alike material unless a more specific original license is documented. |
| `EngDesign` | [`AGI4Engineering/EngDesign`](https://github.com/AGI4Engineering/EngDesign) | MIT | Bundles selected EngDesign task prompts, rubrics, evaluation files, and assets. | Retain MIT attribution to EngDesign for selected task materials. |
| `EngDesign/YJ_02`, `EngDesign/YJ_03` | [`AJJLagerweij/topopt`](https://github.com/AJJLagerweij/topopt) | MIT | Referenced by topology-optimization task files. | Keep the upstream `topopt` attribution with topology-optimization derived code or references. |
| `JobShop/*` | [`Pabloo22/job_shop_lib`](https://github.com/Pabloo22/job_shop_lib) | MIT | Bundles `JobShop/data/benchmark_instances.json` sourced from `job_shop_lib`. | Retain MIT attribution and source URL for vendored JSSP instance data. |
| `KernelEngineering/*` | [`test-time-training/discover`](https://github.com/test-time-training/discover) | MIT | Kernel tasks are documented as originating from this repository. | Retain upstream attribution for task code/templates derived from `discover`. |
| `KernelEngineering/MLA`, `KernelEngineering/TriMul` | [`linkedin/Liger-Kernel`](https://github.com/linkedin/Liger-Kernel) | BSD-2-Clause | `baseline/utils.py` files are adapted from Liger-Kernel test utilities. | Retain BSD-2-Clause attribution for adapted utility code. |
| `KernelEngineering/MLA` | [`deepseek-ai/DeepSeek-V3`](https://github.com/deepseek-ai/DeepSeek-V3) | MIT for code (`LICENSE-CODE`) | Uses DeepSeek-V3 implementation dimensions as a reference source. | Keep the source URL; model weights/license are not bundled or used here. |
| `KernelEngineering/TriMul` | [`lucidrains/triangle-multiplicative-module`](https://github.com/lucidrains/triangle-multiplicative-module) | MIT | `baseline/reference.py` is documented as based on this implementation. | Retain MIT attribution for the reference implementation basis. |
| `PowerSystems/EV2GymSmartCharging` | [`StavrosOrf/EV2Gym`](https://github.com/StavrosOrf/EV2Gym) | MIT | Bundles upstream config/data copies under `references/upstream/` and evaluates with EV2Gym logic. | Retain MIT attribution and upstream source links for copied config/data. |
| `Robotics/CoFlyersVasarhelyiTuning` | [`micros-uav/CoFlyers`](https://github.com/micros-uav/CoFlyers) | GPL-3.0 | Extracts case parameters and reimplements the core Vasarhelyi control/evaluation structure in Python. | Treat this benchmark portion as GPL-3.0-derived material when redistributing. Keep source URLs and GPL notice with derived files. |
| `SingleCellAnalysis/denoising` | [`openproblems-bio/task_denoising`](https://github.com/openproblems-bio/task_denoising) | MIT | Task README instructs users to clone the upstream task; local submission template targets that workflow. | Keep upstream attribution. The upstream clone is not redistributed by this repository. |
| `SingleCellAnalysis/perturbation_prediction` | [`openproblems-bio/task_perturbation_prediction`](https://github.com/openproblems-bio/task_perturbation_prediction) | MIT | Benchmark adapts the task interface and evaluation metrics. | Retain MIT attribution and OpenProblems Bio source links. |
| `SingleCellAnalysis/predict_modality` | [`openproblems-bio/task_predict_modality`](https://github.com/openproblems-bio/task_predict_modality) | MIT | Benchmark adapts the task interface and evaluation metrics. | Retain MIT attribution and OpenProblems Bio source links. |

## Copyright Notices for Copied or Derived Materials

The following upstream copyright notices were resolved during this audit and
must be retained with the corresponding copied or derived benchmark materials.
Standard license texts are stored under `LICENSES/third_party/`.

| Source | Copyright / notice | License text |
|---|---|---|
| `mojtabamozaffar/differentiable-simulation-am` | Copyright (c) 2021 Mojtaba | `LICENSES/third_party/MIT.txt` |
| `peterdsharpe/DawnDesignTool` | Copyright (c) 2019-2021 Peter Sharpe | `LICENSES/third_party/MIT.txt` |
| `duckdb/duckdb` | Copyright 2018-2026 Stichting DuckDB Foundation | `LICENSES/third_party/MIT.txt` |
| `PKUanonym/REKCARC-TSC-UHT` | Source repository license: Creative Commons Attribution-ShareAlike 4.0 International | `LICENSES/third_party/CC-BY-SA-4.0.txt` |
| `AGI4Engineering/EngDesign` | Copyright (c) 2025 EngDesign Benchmark Team | `LICENSES/third_party/MIT.txt` |
| `AJJLagerweij/topopt` | Copyright (c) 2019 A.J.J. Lagerweij | `LICENSES/third_party/MIT.txt` |
| `Pabloo22/job_shop_lib` | Copyright (c) 2024 Pablo Arino | `LICENSES/third_party/MIT.txt` |
| `test-time-training/discover` | Copyright (c) 2025 Mert Yuksekgonul | `LICENSES/third_party/MIT.txt` |
| `linkedin/Liger-Kernel` | Copyright 2024 LinkedIn Corporation. All Rights Reserved. | `LICENSES/third_party/BSD-2-Clause.txt` |
| `deepseek-ai/DeepSeek-V3` code | Copyright (c) 2023 DeepSeek | `LICENSES/third_party/MIT.txt` |
| `lucidrains/triangle-multiplicative-module` | Copyright (c) 2021 Phil Wang | `LICENSES/third_party/MIT.txt` |
| `StavrosOrf/EV2Gym` | Copyright (c) 2024 Stavros Orfanoudakis | `LICENSES/third_party/MIT.txt` |
| `micros-uav/CoFlyers` | GPL-3.0 upstream license notice from `micros-uav/CoFlyers` | `LICENSES/third_party/GPL-3.0-only.txt` |
| `openproblems-bio/task_*` | Copyright (c) 2024 Open Problems in Single-Cell Analysis | `LICENSES/third_party/MIT.txt` |
| AlphaFold 3 article/supplementary screenshots in `KernelEngineering/TriMul/assets` | Open-access article licensed under Creative Commons Attribution 4.0 International; cite DOI `10.1038/s41586-024-07487-w` | `LICENSES/third_party/CC-BY-4.0.txt` |

## Contextual Open-Source References

These repositories are cited by task statements as technical context or
comparison points. This audit did not identify copied source from these entries
unless they are also listed in the bundled/derived table above.

| Benchmark path | Reference source | License | Handling |
|---|---|---|---|
| `KernelEngineering/TriMul` | [`Ligo-Biosciences/AlphaFold3`](https://github.com/Ligo-Biosciences/AlphaFold3) | Apache-2.0 | Contextual open-source AlphaFold 3 implementation reference; no vendored code identified. |
| `KernelEngineering/TriMul` | [`jwohlwend/boltz`](https://github.com/jwohlwend/boltz) | MIT | Contextual biomolecular-model implementation reference; no vendored code identified. |
| `KernelEngineering/TriMul` | [`chaidiscovery/chai-lab`](https://github.com/chaidiscovery/chai-lab) | Apache-2.0 | Contextual biomolecular-model implementation reference; no vendored code identified. |
| `KernelEngineering/TriMul` | [`NVIDIA/cuEquivariance`](https://github.com/NVIDIA/cuEquivariance) | Apache-2.0 | Contextual optimization reference; no vendored code identified. |

## Bundled CC-Licensed Publication Assets

| Benchmark path | Asset/source | License | Handling |
|---|---|---|---|
| `KernelEngineering/TriMul/assets/*.png` | Figures/screenshots adapted from the open-access AlphaFold 3 article and supplementary information, DOI [`10.1038/s41586-024-07487-w`](https://doi.org/10.1038/s41586-024-07487-w) | CC-BY-4.0 | Retain attribution to the article, DOI, CC-BY-4.0 license link, and the note that screenshots/crops may be adapted from the original. |

## External Assets Downloaded at Setup Time

| Benchmark or component | External asset | License / terms status | Handling |
|---|---|---|---|
| `Aerodynamics/CarAerodynamicsSensing` | [`thuml/PhySense`](https://github.com/thuml/PhySense) code, Google Drive car-aerodynamics dataset, and pretrained Transolver checkpoints | PhySense code is MIT. The task README does not declare a separate dataset/checkpoint license. | Code is cloned into `third_party/PhySense`; data/checkpoints are downloaded by users and are not redistributed here. Follow PhySense/provider terms before redistributing derived data or checkpoints. |
| `SustainableDataCenterControl/hand_written_control` | [`HewlettPackard/dc-rl`](https://github.com/HewlettPackard/dc-rl) at commit `a92b475` | MIT | Bootstrap clones this into `hand_written_control/sustaindc` and applies `patches/sustaindc_optional_runtime.patch`. Keep MIT attribution with any vendored or patched copy. |
| `SingleCellAnalysis/perturbation_prediction` | OpenProblems `openproblems-data` S3 `neurips-2023-data`; Kaggle competition materials | Dataset/provider terms are external to this repository. | Data is downloaded at evaluation time and not redistributed here. Follow OpenProblems and Kaggle competition terms before sharing cached data. |
| `SingleCellAnalysis/predict_modality` | OpenProblems `openproblems-data` S3 `openproblems_neurips2021/bmmc_cite/normal/log_cp10k` | OpenProblems task code is MIT; dataset terms are provider-specific. | Data is downloaded into a local cache and not redistributed here. Preserve dataset attribution when publishing derived results. |

## Task-Defining Open-Source Package Dependencies

These entries are not vendored task assets, but the README/Task files identify them
as task-defining runtimes, references, or oracle backends.

| Benchmark path | Package / repository | License | Handling |
|---|---|---|---|
| `Cryptographic/*` | [OpenSSL](https://www.openssl.org/) | Apache-2.0 | Verification-only system/library dependency; not vendored. |
| `InventoryOptimization/*` | [`stockpyl`](https://github.com/LarrySnyder/stockpyl) | MIT | Reference implementation dependency; not vendored. |
| `JobShop/*` | [OR-Tools](https://github.com/google/or-tools) | Apache-2.0 | Reference solver dependency; not vendored. |
| `MolecularMechanics/*` | [`openff-toolkit`](https://github.com/openforcefield/openff-toolkit), [`openff-forcefields`](https://github.com/openforcefield/openff-forcefields), RDKit, OpenMM, AmberTools | OpenFF Toolkit: MIT; OpenFF force fields: CC-BY-4.0; RDKit: BSD-3-Clause; OpenMM: MIT; AmberTools has mixed upstream terms. | Runtime environment dependency. Do not vendor the OpenFF/chemistry stack without carrying each package's license and data notices. |
| `Optics/adaptive_*` | [AOtools](https://github.com/AOtools/aotools) | LGPL-3.0 | Runtime dependency only. If vendored, include the exact upstream LGPL license text. |
| `Optics/fiber_*` | [`OptiCommPy`](https://github.com/edsonportosilva/OptiCommPy) | GPL-3.0 | Runtime dependency only. Avoid vendoring or statically incorporating without GPL-3.0 compliance. |
| `Optics/phase_dammann_uniform_orders` | [`diffractio`](https://github.com/optbrea/diffractio) | GPL-3.0 | Runtime dependency only. Avoid vendoring without GPL-3.0 compliance. |
| `Optics/phase_*`, `Optics/holographic_*` | [`slmsuite`](https://github.com/slmsuite/slmsuite) | MIT | Oracle/backend dependency; not vendored. |
| `Optics/holographic_*` | [`torchoptics`](https://github.com/matthewfilipovich/torchoptics) | MIT | Runtime dependency; not vendored. |
| `PyPortfolioOpt/*` | [`PyPortfolioOpt`](https://github.com/PyPortfolio/PyPortfolioOpt) | MIT | Solver workflow dependency; not vendored. |
| `QuantumComputing/*` | [`mqt.bench`](https://github.com/cda-tum/mqt-bench) | MIT | Runtime dependency for generated quantum benchmark circuits; not vendored. |
| `ReactionOptimisation/*` | [`summit`](https://pypi.org/project/summit/) | MIT | Benchmark/reference dependency; not vendored. |
| `Robotics/RobotArmCycleTimeOptimization` | PyBullet / `pybullet_data` | zlib/libpng-style PyBullet license; data provided by PyBullet package | Runtime package data dependency; not vendored. |
| `Robotics/QuadrupedGaitOptimization` | MuJoCo | Apache-2.0 | Runtime dependency. `references/ant.xml` is self-contained in this repository; if its provenance is later tied to an upstream Ant asset, add that source and license here. |
| `SingleCellAnalysis/denoising_ttt` | [`test-time-training/discover`](https://github.com/test-time-training/discover), [`czbiohub-sf/simscity`](https://github.com/czbiohub-sf/simscity), [`czbiohub-sf/molecular-cross-validation`](https://github.com/czbiohub-sf/molecular-cross-validation) | MIT | Denoising/TTT task and verification dependencies; not vendored except local benchmark adaptation files. |

## Framework Optional Third-Party Checkouts

These are framework/algorithm integrations rather than benchmark task assets, but
they are cloned by `scripts/bootstrap/fetch_task_assets.py`.

| Component | Upstream source | License | Handling |
|---|---|---|---|
| `abmcts` / TreeQuest | [`SakanaAI/treequest`](https://github.com/SakanaAI/treequest) | Apache-2.0 | Optional local checkout under `third_party/treequest`; not redistributed by default. |
| `shinkaevolve` | [`SakanaAI/ShinkaEvolve`](https://github.com/SakanaAI/ShinkaEvolve) | Apache-2.0 | Optional local checkout under `third_party/ShinkaEvolve`; local patching should preserve Apache-2.0 notices. |
| `openevolve` | [`algorithmicsuperintelligence/openevolve`](https://github.com/algorithmicsuperintelligence/openevolve) / PyPI `openevolve` | Apache-2.0 | Optional package or local checkout; not vendored by default. |

## Sources Requiring Follow-Up Before Redistribution

The following sources are clearly referenced by task docs or bundled files, but
their redistribution terms were not resolved fully from the task README alone:

- `Aerodynamics/CarAerodynamicsSensing` Google Drive dataset/checkpoint files from
  PhySense: license not separately declared in the task README.
- `SingleCellAnalysis/perturbation_prediction` Kaggle/OpenProblems data: governed
  by external competition and provider terms.
- `StructuralOptimization/ISCSO2015` and `StructuralOptimization/ISCSO2023`
  competition materials from Bright Optimizer: source is attributed, but no
  explicit open-source/data license is declared in the task README. Original
  section-list PDFs and winner images are not bundled.
- `EngDesign/CY_03` and `EngDesign/XY_05` reference external
  architecture/specification teaching materials (RISC-V, OASIS VIRTIO, UIUC
  slides/spec PDFs). These appear to be citations rather than open-source
  benchmark assets; review their terms before bundling additional copies.
- Publication/reference PDFs and competition images with unclear redistribution
  rights were removed from `benchmarks/*/references/`. Their folders now contain
  citation or external-reference notes only. Do not re-add local copies without
  documenting redistribution permission.
- `KernelEngineering` task statements reference GPU Mode leaderboard pages. The
  benchmark code origins are covered above where repository sources are declared;
  leaderboard page content should be treated as externally copyrighted website
  material unless explicit terms are documented.
