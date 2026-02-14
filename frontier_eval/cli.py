from __future__ import annotations

import asyncio
import os
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from frontier_eval.env import find_dotenv, load_dotenv
from frontier_eval.registry import get_algorithm, get_task


def _register_omegaconf_resolvers() -> None:
    def _basename(value) -> str:
        s = str(value or "").rstrip("/")
        return s.rsplit("/", 1)[-1]

    OmegaConf.register_new_resolver("fe.basename", _basename, replace=True)


_register_omegaconf_resolvers()


def _load_dotenv() -> None:
    cwd = Path.cwd().resolve()
    dotenv_path = find_dotenv(cwd)
    if dotenv_path is not None:
        load_dotenv(dotenv_path, override=False)
        return

    repo_root = os.environ.get("FRONTIER_ENGINEERING_ROOT", "")
    if not repo_root:
        return
    candidate = (Path(repo_root).expanduser().resolve() / ".env").resolve()
    if candidate.is_file():
        load_dotenv(candidate, override=False)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def _hydra_main(cfg: DictConfig) -> None:
    original_cwd = Path(get_original_cwd()).resolve()

    dotenv_path = find_dotenv(original_cwd)
    if dotenv_path is not None:
        load_dotenv(dotenv_path, override=False)

    repo_root = original_cwd
    if "paths" in cfg and "repo_root" in cfg.paths and cfg.paths.repo_root:
        repo_root = (repo_root / str(cfg.paths.repo_root)).resolve()
        if dotenv_path is None:
            repo_dotenv = repo_root / ".env"
            if repo_dotenv.is_file():
                load_dotenv(repo_dotenv, override=False)

    task = get_task(str(cfg.task.name))(cfg=cfg, repo_root=repo_root)
    algorithm = get_algorithm(str(cfg.algorithm.name))(cfg=cfg, repo_root=repo_root)

    cfg_view = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg_view, dict):
        llm_view = cfg_view.get("llm")
        if isinstance(llm_view, dict) and llm_view.get("api_key"):
            llm_view["api_key"] = "***REDACTED***"
    print(OmegaConf.to_yaml(OmegaConf.create(cfg_view), resolve=True))
    asyncio.run(algorithm.run(task=task))


def main() -> None:
    _load_dotenv()
    _hydra_main()


if __name__ == "__main__":
    main()
