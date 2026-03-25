from __future__ import annotations

import sys


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "skill":
        from frontier_eval.skill_cli import main as skill_main

        raise SystemExit(skill_main(sys.argv[2:]))

    from frontier_eval.cli import main

    main()

