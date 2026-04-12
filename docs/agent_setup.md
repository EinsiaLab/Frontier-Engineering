# Optional Assistant Setup

## Skill sources

Project skill packages are under:

- `skill/source/frontier-evaluator`
- `skill/source/frontier-contributor`

Each package should be installed as one loadable project-level skill for your client/editor, including `SKILL.md` and any referenced relative assets (for example, scripts under `frontier-evaluator`).

There is no bundled CLI installer in this repository. Use your client/editor convention for project-local skills and follow each package's `SKILL.md` as the source of truth.

## Copy-paste prompt (optional)

Use this when you want your assistant to perform setup automatically:

```text
Please install the two skills under this repo’s skill/source/ as project-level skills for this repository (follow my editor’s / client’s convention for where project skills live, e.g. .cursor/skills/ for Cursor, .claude/skills/ for Claude Code, .codex/skills/ for Codex, etc.). The two packages are:

1. skill/source/frontier-evaluator — help run and debug frontier_eval evaluations and prepare per-benchmark runtime from each benchmark’s README;
2. skill/source/frontier-contributor — help add or update benchmarks following this repo’s contribution rules.

Install each folder as one loadable skill package: include SKILL.md plus any relative assets it references (e.g. scripts/ under frontier-evaluator) so everything works from the repository root after installation.
```
