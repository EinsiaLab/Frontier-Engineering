# SHA-256 算法加速

我们提供了不使用第三方库的基础 C++ 实现 `baseline/SHA-256.cpp`，你需要修改这个文件，在保持正确性的前提下提高算法吞吐率
`references/SHA-256.pdf` 是算法详细介绍文件
运行 `verification/valid.sh` 会生成随机数据对算法正确性进行验证（使用OpenSSL）。
运行 `verification/eval.sh` 会分别使用 8Kbits 和 8Mbits 两种数据流进行多次运算，计算算法实现效率。