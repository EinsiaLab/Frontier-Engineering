// EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>

#include <fstream>


static constexpr uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

static constexpr int RHO[5][5] = {
    {0, 36, 3, 41, 18},
    {1, 44, 10, 45, 2},
    {62, 6, 43, 15, 61},
    {28, 55, 25, 21, 56},
    {27, 20, 39, 8, 14}
};

class SHA3_256 {
private:
    uint64_t state[25]; 
    int pos;            

    
    // Rotates left a 64‑bit word. Marked constexpr for potential compile‑time evaluation.
    static constexpr uint64_t rotl(uint64_t a, int offset) noexcept {
        return offset == 0 ? a : (a << offset) | (a >> (64 - offset));
    }

    
    void keccak_f1600() {
        // Round constants (RC) and rho offsets (RHO) are defined as file‑scope constexpr arrays.

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

    
    // Inline version of xor_byte – eliminates function‑call overhead.
    inline void xor_byte(int byte_index, uint8_t byte_val) noexcept {
        int word_index = byte_index / 8;
        int shift = (byte_index % 8) * 8;
        state[word_index] ^= (static_cast<uint64_t>(byte_val) << shift);
    }

public:
    SHA3_256() {
        reset();
    }

    
    void reset() {
        for (int i = 0; i < 25; ++i) {
            state[i] = 0;
        }
        pos = 0;
    }

    
    void update(const uint8_t* data, size_t len) {
        // Process data in blocks of the SHA‑3‑256 bitrate (136 bytes) when possible.
        constexpr int RATE_BYTES = 136;
        constexpr int WORDS_PER_BLOCK = RATE_BYTES / 8;   // 17 words
        size_t i = 0;

        // --------------------------------------------------------------------
        // Fast‑path 1: absorb whole 136‑byte blocks when the internal buffer is empty.
        // --------------------------------------------------------------------
        // Fast‑path: absorb whole 136‑byte blocks when the state is empty.
        // reinterpret_cast is safe on x86‑64 (unaligned loads are free) and lets
        // the compiler emit a single load per word, avoiding the memcpy overhead.
        while (i + RATE_BYTES <= len && pos == 0) {
            const uint64_t* src = reinterpret_cast<const uint64_t*>(data + i);
            for (int w = 0; w < WORDS_PER_BLOCK; ++w) {
                state[w] ^= src[w];
            }
            keccak_f1600();                // permute the full block
            i += RATE_BYTES;
        }

        // --------------------------------------------------------------------
        // Fast‑path 2: absorb remaining whole 8‑byte words while word‑aligned.
        // --------------------------------------------------------------------
        while (i + 8 <= len && (pos % 8 == 0)) {
            uint64_t chunk;
            std::memcpy(&chunk, data + i, sizeof(chunk));
            int word_index = pos / 8;
            state[word_index] ^= chunk;               // XOR whole word

            i += 8;
            pos += 8;
            if (pos == RATE_BYTES) {
                keccak_f1600();
                pos = 0;
            }
        }

        // --------------------------------------------------------------------
        // Slow path: handle the tail bytes (≤ 7) one‑by‑one.
        // --------------------------------------------------------------------
        for (; i < len; ++i) {
            xor_byte(pos, data[i]);
            ++pos;
            if (pos == RATE_BYTES) {
                keccak_f1600();
                pos = 0;
            }
        }
    }

    
    void update(const std::string& text) {
        update(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    }

    
    std::vector<uint8_t> finalize() {
        // Apply SHA‑3 padding and final permutation.
        xor_byte(pos, 0x06);
        xor_byte(135, 0x80);
        keccak_f1600();

        std::vector<uint8_t> hash(32);
        for (int i = 0; i < 32; ++i) {
            int word_index = i / 8;
            int shift = (i % 8) * 8;
            hash[i] = static_cast<uint8_t>((state[word_index] >> shift) & 0xFF);
        }
        return hash;
    }

    
    // Produce a hex string without allocating the intermediate vector used by
    // `finalize()`. This saves a heap allocation and a copy of the 32‑byte hash,
    // which speeds up the one‑shot use‑case (the program hashes a file once).
    std::string hexdigest() {
        // Apply SHA‑3 padding directly on the internal state.
        xor_byte(pos, 0x06);
        xor_byte(135, 0x80);
        keccak_f1600();

        static const char* hex_chars = "0123456789abcdef";
        std::string result;
        result.reserve(64);               // 32 bytes → 64 hex chars
        for (int i = 0; i < 32; ++i) {
            int word_index = i / 8;
            int shift = (i % 8) * 8;
            uint8_t byte = static_cast<uint8_t>((state[word_index] >> shift) & 0xFF);
            result.push_back(hex_chars[(byte >> 4) & 0xF]);
            result.push_back(hex_chars[byte & 0xF]);
        }
        return result;
    }
};




std::string hash_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return "Error: Could not open file.";
    }

    SHA3_256 sha3;
    // Use a 1 MiB stack buffer – large enough to amortise syscalls while staying
    // comfortably within typical L3 cache sizes.
    static constexpr std::size_t BUF_SIZE = 1 << 20;   // 1 MiB
    char buffer[BUF_SIZE];
    while (true) {
        file.read(buffer, sizeof(buffer));
        std::streamsize bytes = file.gcount();
        if (bytes == 0) break;
        sha3.update(reinterpret_cast<const uint8_t*>(buffer), static_cast<size_t>(bytes));
    }

    return sha3.hexdigest();
}

int main(int argc, char* argv[]) {
    // Fast‑IO: disable synchronization with C stdio and untie cin/cout.
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc != 2) {
        return 1;
    }

    std::cout << hash_file(argv[1]);
    return 0;
}
// EVOLVE-BLOCK-END
