# 可选 Assistant/Agent 配置

## Skill 来源

项目 skill 包位于：

- `skill/source/frontier-evaluator`
- `skill/source/frontier-contributor`

请将每个目录作为一个可加载的项目级 skill 包接入，并包含该目录下的 `SKILL.md` 及其中引用的相对路径资源（例如 `frontier-evaluator` 下的 scripts）。

本仓库不提供统一的 skill CLI 安装器。请按你当前客户端/编辑器的项目级 skill 目录约定接入，并以各包 `SKILL.md` 为准。

## 可复制提示词（可选）

当你希望让 Assistant 自动完成接入时，可使用下述提示词：

```text
请你把本仓库 skill/source/ 下的两个 skill 安装为**本仓库的项目级 skill**（按我当前客户端对项目级 skill 的目录约定处理，例如 Cursor 的 .cursor/skills/、Claude Code 的 .claude/skills/、Codex 的 .codex/skills/ 等），分别是：

1. skill/source/frontier-evaluator — 协助运行与调试 frontier_eval 评测、按各 benchmark README 准备运行环境与命令；
2. skill/source/frontier-contributor — 协助按本仓库规范贡献或更新 benchmark。

每个目录请以「一个可加载的 skill 包」接入：至少包含该目录下的 SKILL.md，以及其中引用的相对路径资源（例如 frontier-evaluator 下的 scripts/ 等），保证安装后在本仓库根目录下可正常使用。
```
