set -euo pipefail

OLD_PWD=$(pwd)
cd verification

g++ -std=c++17 -O3 ../baseline/SHA-256.cpp -o custom_sha
g++ -std=c++17 -O3 evaluate.cpp -o evaluate

./evaluate
cd "$OLD_PWD"
