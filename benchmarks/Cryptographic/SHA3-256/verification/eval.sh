set -euo pipefail

OLD_PWD=$(pwd)
cd verification

g++ -std=c++17 -O3 ../baseline/SHA3-256.cpp -o custom_sha3
g++ -std=c++17 -O3 evaluate.cpp -o evaluate

./evaluate
cd "$OLD_PWD"
