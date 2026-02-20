# EVOLVE-BLOCK-START
"""
Minimal, fast program used by `frontier_eval` smoke tests.

Contract: running this file should exit 0 quickly.
"""


def main() -> int:
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# EVOLVE-BLOCK-END

