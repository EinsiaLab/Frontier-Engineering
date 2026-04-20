#!/usr/bin/env bash
# ------------------------------------------------------------
# clean_personal_paths.sh
# ------------------------------------------------------------
# This utility scans the repository for hard‑coded personal or internal
# absolute paths and replaces them with a generic placeholder. It is
# intended to be run after cloning the repo to ensure a clean, public‑
# ready code base.
# ------------------------------------------------------------

# List of path patterns to scrub (add more as needed)
PATTERNS=(
  "/data_storage/chihh2311/"
  "/mnt/shared-storage-user/p1-shared/luotianwei/"
  "litellm.nbdevenv.xiaoaojianghu.fun"
)

# Placeholder to replace the personal paths
PLACEHOLDER="/path/to/placeholder/"

# Find all text files (excluding binary) and replace patterns
for pattern in "${PATTERNS[@]}"; do
  echo "[clean] Replacing occurrences of ${pattern}"
  # Use grep to locate files containing the pattern, then sed to replace
  grep -rl --exclude-dir=.git --exclude-dir=__pycache__ --exclude='*.png' --exclude='*.jpg' "${pattern}" . | while read -r file; do
    # Ensure we don't modify binary files
    if file "$file" | grep -q "text"; then
      sed -i "s#${pattern}#${PLACEHOLDER}#g" "$file"
    fi
  done
done

echo "Cleaning complete."
