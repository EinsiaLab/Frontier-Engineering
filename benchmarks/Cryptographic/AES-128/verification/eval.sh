set -euo pipefail

OLD_PWD=$(pwd)
cd verification

g++ -std=c++17 -O3 ../baseline/AES-128.cpp -o custom_aes
g++ -std=c++17 -O3 evaluate.cpp -o evaluate

./evaluate
cd "$OLD_PWD"
