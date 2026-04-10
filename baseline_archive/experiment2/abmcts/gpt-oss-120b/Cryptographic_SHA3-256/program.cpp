// EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>

class SHA3_256 {
private:
    static constexpr int RATE_BYTES = 136; // 1088 bits
    uint64_t state[25];
    uint8_t  block[RATE_BYTES];
    int      pos;

    inline uint64_t rotl(uint64_t a, int offset) {
        return (a << offset) | (a >> (64 - offset));
    }

    void keccak_f1600() {
        static const uint64_t RC[24] = {
            0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
            0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
            0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
            0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
            0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
            0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
            0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
            0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
        };

        static const int RHO[5][5] = {
            { 0, 36,  3, 41, 18},
            { 1, 44, 10, 45,  2},
            {62,  6, 43, 15, 61},
            {28, 55, 25, 21, 56},
            {27, 20, 39,  8, 14}
        };

        for (int round = 0; round < 24; ++round) {
            uint64_t C[5], D[5];
            for (int x = 0; x < 5; ++x) {
                C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
            }
            for (int x = 0; x < 5; ++x) {
                D[x] = C[(x + 4) % 5] ^ rotl(C[(x + 1) % 5], 1);
            }
            for (int x = 0; x < 5; ++x) {
                for (int y = 0; y < 5; ++y) {
                    state[x + 5 * y] ^= D[x];
                }
            }

            uint64_t B[25];
            for (int x = 0; x < 5; ++x) {
                for (int y = 0; y < 5; ++y) {
                    B[y + 5 * ((2 * x + 3 * y) % 5)] = rotl(state[x + 5 * y], RHO[x][y]);
                }
            }

            for (int y = 0; y < 5; ++y) {
                for (int x = 0; x < 5; ++x) {
                    state[x + 5 * y] = B[x + 5 * y] ^ (~B[((x + 1) % 5) + 5 * y] & B[((x + 2) % 5) + 5 * y]);
                }
            }

            state[0] ^= RC[round];
        }
    }

    void absorb_block() {
        for (int i = 0; i < RATE_BYTES / 8; ++i) {
            uint64_t v;
            std::memcpy(&v, block + i * 8, sizeof(v));
            state[i] ^= v;
        }
        keccak_f1600();
    }

    // Retained for compatibility; not used in the optimized path.
    void xor_byte(int byte_index, uint8_t byte_val) {
        int word_index = byte_index / 8;
        int shift = (byte_index % 8) * 8;
        state[word_index] ^= (static_cast<uint64_t>(byte_val) << shift);
    }

public:
    SHA3_256() { reset(); }

    void reset() {
        std::fill(state, state + 25, 0ULL);
        std::fill(block, block + RATE_BYTES, 0);
        pos = 0;
    }

    void update(const uint8_t* data, size_t len) {
        while (len) {
            size_t space = RATE_BYTES - pos;
            size_t chunk = std::min(space, len);
            std::memcpy(block + pos, data, chunk);
            pos += static_cast<int>(chunk);
            data += chunk;
            len -= chunk;

            if (pos == RATE_BYTES) {
                absorb_block();
                pos = 0;
                std::fill(block, block + RATE_BYTES, 0);
            }
        }
    }

    void update(const std::string& text) {
        update(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    }

    std::vector<uint8_t> finalize() {
        uint64_t saved_state[25];
        uint8_t  saved_block[RATE_BYTES];
        int      saved_pos = pos;
        std::memcpy(saved_state, state, sizeof(state));
        std::memcpy(saved_block, block, sizeof(block));

        // Pad
        block[pos] ^= 0x06;
        block[RATE_BYTES - 1] ^= 0x80;
        absorb_block();

        std::vector<uint8_t> hash(32);
        for (int i = 0; i < 32; ++i) {
            int word_index = i / 8;
            int shift = (i % 8) * 8;
            hash[i] = static_cast<uint8_t>((state[word_index] >> shift) & 0xFF);
        }

        // Restore original state
        std::memcpy(state, saved_state, sizeof(state));
        std::memcpy(block, saved_block, sizeof(block));
        pos = saved_pos;

        return hash;
    }

    std::string hexdigest() {
        std::vector<uint8_t> hash = finalize();
        std::ostringstream oss;
        for (uint8_t b : hash) {
            oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
        }
        return oss.str();
    }
};

std::string hash_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return "Error: Could not open file.";
    }

    SHA3_256 sha3;
    char buffer[8192];
    while (file.read(buffer, sizeof(buffer))) {
        sha3.update(reinterpret_cast<const uint8_t*>(buffer), static_cast<size_t>(file.gcount()));
    }
    if (file.gcount() > 0) {
        sha3.update(reinterpret_cast<const uint8_t*>(buffer), static_cast<size_t>(file.gcount()));
    }

    return sha3.hexdigest();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        return 1;
    }
    std::cout << hash_file(argv[1]);
    return 0;
}
// EVOLVE-BLOCK-END
