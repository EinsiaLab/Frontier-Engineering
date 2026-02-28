# Agent Guide: Perfect Pull Request

## 1. Role

### Agent Responsibilities
- Implement verification code (`verification/evaluator.py`)
- Provide feasible solution (`scripts/init.py`)
- Create baseline solution (optional, `baseline/solution.py`)
- Write clean, minimal code with self-explanatory names

### Agent Restrictions
- DO NOT generate `README.md` or `Task.md` files (handled by maintainers)
- DO NOT over-document or over-comment code
- DO NOT include private information or absolute paths
- **If you must generate any documentation files**: Add identifier `<!-- AI_GENERATED -->` at the end of the file to ensure human review and manual removal

### Reference Structure
Reference: `benchmarks/Astrodynamics/MannedLunarLanding/`

Required structure:
```
<Task_Name>/
├── scripts/
│   └── init.py              # [Required] Feasible solution entry point
├── verification/
│   ├── evaluator.py         # [Required] Scoring script entry point
│   ├── requirements.txt     # [Required] Dependencies
│   └── docker/              # [Optional] Environment containerization
│       └── Dockerfile
├── baseline/                # [Optional] Baseline solution
│   ├── solution.py
│   └── result_log.txt
└── references/              # [Optional] References directory
    ├── constants.json
    └── manuals.pdf
```

## 2. Workflow

### Step 1: Understand Requirements
Task must satisfy:
1. **Reality Gap**: Close to reality, considering real-world factors, not purely abstract mathematics
2. **Economic Value**: Clear engineering or economic value upon solution
3. **Verifiability**: Executable verification program (Docker preferred) completing evaluation within acceptable time

### Step 2: Implement Solution
- Create `scripts/init.py` as a feasible solution that:
  - Passes all constraint checks
  - Produces valid output format
  - Completes evaluation without errors
  - Achieves non-zero score (valid, not necessarily optimal)
  - **IMPORTANT**: Include important logic and utility functions in `init.py`
    - Since algorithm iterations will work on `init.py`, ensure sufficient context is available
    - Include critical tool functions that are needed for the solution
    - For given utility functions, clearly annotate with comments:
      - Which parts can be modified
      - Which parts must NOT be modified (e.g., interface contracts, constraint checks)

### Step 3: Implement Verification
- Create `verification/evaluator.py` as scoring script entry point
- Create `verification/requirements.txt` with dependencies
- Optionally create Docker configuration

### Step 4: Test Locally
Run mandatory tests before submission:

**Test 1**: Basic functionality
```bash
python verification/evaluator.py scripts/init.py
```
Pass Criteria: Exit code 0, `valid=1.0`, all constraints satisfied, completes within acceptable time

**Test 2**: Framework integration
```bash
python -m frontier_eval task=<task_name> algorithm.iterations=0
```
Pass Criteria: Task name registered in domain README, framework loads without errors, completes at least 1 iteration

Note: Keep test commands short (ideally single-line). Testing is mandatory before submission.

### Step 5: Clean Up
Remove before submission:
- `.env` files, API keys, credentials
- IDE configs (`.vscode/`, `.idea/`)
- Temporary files (`*.log`, `temp/`, `__pycache__/`, `*.pyc`)
- Personal test scripts
- Absolute paths (use relative paths only)
- Large binary files (unless necessary)

Purpose: Avoid reproducibility issues and privacy leaks.

### Step 6: Submit PR
Create PR with description following template (see Requirements section).

## 3. Requirements

### Code Quality
- Self-explanatory variable and function names
- Minimal comments (only for complex logic)
- Follow PEP 8 (Python) or equivalent style guide
- Remove debug code, print statements, temporary files

### AI-Generated Documentation Identifier
If you must generate any documentation files (which is discouraged), add the identifier at the end:
```markdown
<!-- AI_GENERATED -->
```
This ensures human reviewers can identify and manually remove AI-generated content after review.

### init.py Requirements
- **Include Important Logic**: Place critical logic and utility functions directly in `init.py`
  - Algorithm iterations will modify `init.py`, so all necessary context must be present
  - Do not rely on external modules that may not be available during iteration
- **Single-File Closure (Required)**: `scripts/init.py` (and optional `baseline/solution.py`) must be self-contained to enable single-file optimization (e.g., OpenEvolve)
  - Do **not** import other Python modules from this benchmark repository (e.g., `benchmarks/...` or other `.py` files in the task folder)
  - Imports from the Python standard library and packages listed in `verification/requirements.txt` are allowed
- **Annotate Utility Functions**: For given utility functions, use comments to clearly mark:
  - `# MODIFIABLE: [description]` - Parts that can be changed during iteration
  - `# DO NOT MODIFY: [description]` - Parts that must remain unchanged (e.g., interface contracts, constraint validation, I/O format)
- **Example**:
  ```python
  def constraint_check(x):
      # DO NOT MODIFY: This function validates constraints required by evaluator
      # MODIFIABLE: Internal validation logic can be optimized
      ...
  ```

### Solution Requirements
- **Feasible**: Must satisfy all problem constraints
- **Valid**: Must produce correct output format
- **Runnable**: Must execute without errors
- **Reasonable**: Must complete within time limits

### Test Evidence in PR
REQUIRED: Include actual test outputs in PR description:

```markdown
## Test Evidence

### Basic Functionality Test
```bash
$ python verification/evaluator.py scripts/init.py
```
Output: `{"score": 0.75, "valid": 1.0, "runtime_s": 12.3}`
PASSED: Exit code 0, valid output, all constraints satisfied

### Framework Integration Test
```bash
$ python -m frontier_eval task=my_task algorithm.iterations=0
```
PASSED: Framework integration successful
```

### PR Description Template
```markdown
## Task Overview
[Brief description - 2-3 sentences]

## Domain
[Domain name]

## Task Name
[Registered identifier - must match domain README]

## Background & Source
[Explain the real-world problem, its engineering value, and source]

## Solution Approach
[Brief explanation of feasible solution - 1-2 paragraphs]

## How to Run Verification
[Step-by-step commands to run verification code]

## Test Evidence
[Actual test outputs - REQUIRED]

## Checklist
- [x] Solution is feasible (satisfies all constraints)
- [x] All mandatory tests passed with evidence
- [x] No README/Task.md generated
- [x] No private info or absolute paths
- [x] Task name registered in domain README
```

## 4. Termination Conditions

### Rejection Criteria
PR will be REJECTED if any of the following occur:

1. **Infeasible Solution**: Solution violates constraints
2. **Missing Test Evidence**: No proof of passing tests in PR description
3. **Generated Documentation**: README.md or Task.md files generated by agent (unless marked with `<!-- AI_GENERATED -->` identifier for human review)
4. **Test Failures**: Any mandatory test fails
5. **Invalid Output Format**: Output doesn't match specification
6. **Missing Task Requirements**: Task doesn't satisfy Reality Gap, Economic Value, or Verifiability requirements
7. **Private Information**: Contains API keys, credentials, or absolute paths
8. **Over-documentation**: Excessive comments or generated documentation files without `<!-- AI_GENERATED -->` identifier

### Success Criteria
PR is ready for maintainer review when:
- Solution is feasible and passes all tests
- Test evidence clearly documented in PR
- Code is minimal, clean, and well-named
- No documentation files generated
- All requirements satisfied
- Clean submission (no private info or temp files)

---

**Focus**: Feasible solution + test evidence. Documentation handled separately by maintainers.
