# First 20 Ideas Implementation Status

Updated on 2026-03-13.

See also: `docs/benchmark_ideas/frontier_first20_quality_audit.md` for the consolidated integrity and scoring audit across these tasks.

## Implemented and Smoke-Tested

These benchmarks have been created and passed direct evaluator runs. `EOQMOQ`, `CantileverTopologyOptimization`, `MultiRobotPriorityPlanning`, `DuckDBQueryRewrite`, and `ShipWeatherRoutingFuel` also passed `frontier_eval task=unified` smoke runs when forced to use the repository `.venv` Python.

Under `benchmarks/OperationsResearch/`:

1. `EOQMOQ`
2. `EOQAllUnitsDiscount`
3. `EOQIncrementalDiscount`
4. `RQPoissonServiceLevel`
5. `RQNormalServiceLevel95`
6. `FT10DispatchRule`
7. `LA16DispatchRule`
8. `FT10NeighborhoodMoves`
9. `LA16NeighborhoodMoves`
19. `ShipWeatherRoutingFuel`
20. `DynamicCurrentTimeRouting`

Under `benchmarks/StructuralOptimization/`:

16. `CantileverTopologyOptimization`
17. `MBBBeamTopologyOptimization`
18. `BridgeTopologyOptimization`

Under `benchmarks/Robotics/`:

13. `GridObstaclePathPlanning`
14. `NarrowPassagePathPlanning`
15. `MultiRobotPriorityPlanning`

Under `benchmarks/ComputerSystems/`:

10. `DuckDBIndexSelection`
11. `DuckDBQueryRewrite`
12. `DuckDBPreAggregationSelection`

Notes:

- JSSP benchmark instance data has been cloned from `job_shop_lib` into `/tmp/job_shop_lib`, and the canonical JSON payload is available locally.
- `pymoto` is installed in the repository `.venv`.
- `duckdb` is installed in the repository `.venv`.
- The DuckDB benchmarks intentionally use a benchmark-local deterministic SQL-generated workload with DuckDB/TPC-H schema lineage. They do not claim to redistribute official `dbgen` output.
- The maritime routing benchmarks intentionally use benchmark-local synthetic coastal fields with official WeatherRoutingTool / HALEM algorithm lineage. They do not claim to redistribute official hydrographic or weather rasters.
