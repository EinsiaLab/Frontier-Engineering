# SHA-256 Algorithm Acceleration

We provide a basic C++ implementation `baseline/SHA-256.cpp` that does not use third-party libraries. You need to modify this file to improve the algorithm's throughput while maintaining correctness.

`references/SHA-256.pdf` is a detailed description of the algorithm.

Running `verification/valid.sh` will generate random data to verify the algorithm's correctness (using OpenSSL).

Running `verification/eval.sh` will perform multiple calculations using both 8Kbits and 8Mbits data streams to calculate the algorithm's efficiency.