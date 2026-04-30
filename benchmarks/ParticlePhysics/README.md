# Particle Physics and Nuclear Engineering

English | [简体中文](./README_zh-CN.md)

## 1. Domain Background
Particle physics and nuclear technology involve the study of fundamental particles and their interactions. In modern large-scale scientific engineering (e.g., high-energy particle colliders, cosmic ray detection, nuclear radiation monitoring), the layout of detector arrays, the setting of data filtering thresholds, and the control of accelerator parameters often face extremely complex physical constraints and high economic costs.

Introducing AI Agents to solve open-ended optimization problems in this domain can significantly reduce engineering and trial-and-error costs while ensuring physical objectives (such as statistical significance and signal-to-noise ratio) are met.

## 2. Sub-task Index

Currently, this domain includes the following benchmark tasks:

* **[Muon Tomography Detector Placement Optimization](./MuonTomography/README.md)**
  * **Background**: Utilizing cosmic ray muons for internal transmission imaging of large structures.
  * **Objective**: Find the optimal spatial coordinates and angular layout of detectors under budget and excavation constraints to maximize the effective received flux in the target region.
* **[IMPT Dose Weight Optimization](./ProtonTherapyPlanning/README.md)**
  * **Background**: Optimizing proton therapy treatment plans using the Bragg peak effect of proton beams.
  * **Objective**: Optimize proton spot positions and weights under CTV coverage, OAR dose limits, and beam cost constraints.
* **[PET Scanner Geometry and Cost Pareto Optimization](./PETScannerOptimization/README.md)**
  * **Background**: PET detector ring design under scintillator budget and resolution constraints.
  * **Objective**: Optimize 20 detector rings to balance sensitivity, parallax error, and crystal volume budget.
