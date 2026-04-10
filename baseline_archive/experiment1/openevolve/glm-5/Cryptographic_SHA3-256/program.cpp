// EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>

class SHA3_256 {
private:
    uint64_t state[25]; 
    int pos;            

    
    static inline uint64_t rotl(uint64_t a, int offset) {
        return (a << offset) | (a >> (64 - offset));
    }

    
    void keccak_f1600() {
        const uint64_t RC[24] = {
            0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
            0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
            0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
            0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
            0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
            0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
            0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
            0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
        };

        const int RHO[5][5] = {
            {0, 36, 3, 41, 18},
            {1, 44, 10, 45, 2},
            {62, 6, 43, 15, 61},
            {28, 55, 25, 21, 56},
            {27, 20, 39, 8, 14}
        };

        for (int round = 0; round < 24; ++round) {
            // Theta step - unrolled for performance
            uint64_t C0 = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
            uint64_t C1 = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
            uint64_t C2 = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
            uint64_t C3 = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
            uint64_t C4 = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];
            
            uint64_t D0 = C4 ^ rotl(C1, 1);
            uint64_t D1 = C0 ^ rotl(C2, 1);
            uint64_t D2 = C1 ^ rotl(C3, 1);
            uint64_t D3 = C2 ^ rotl(C4, 1);
            uint64_t D4 = C3 ^ rotl(C0, 1);
            
            state[0] ^= D0; state[5] ^= D0; state[10] ^= D0; state[15] ^= D0; state[20] ^= D0;
            state[1] ^= D1; state[6] ^= D1; state[11] ^= D1; state[16] ^= D1; state[21] ^= D1;
            state[2] ^= D2; state[7] ^= D2; state[12] ^= D2; state[17] ^= D2; state[22] ^= D2;
            state[3] ^= D3; state[8] ^= D3; state[13] ^= D3; state[18] ^= D3; state[23] ^= D3;
            state[4] ^= D4; state[9] ^= D4; state[14] ^= D4; state[19] ^= D4; state[24] ^= D4;

            // Rho and Pi steps
            uint64_t B[25];
            for (int x = 0; x < 5; ++x) {
                for (int y = 0; y < 5; ++y) {
                    B[y + 5 * ((2 * x + 3 * y) % 5)] = rotl(state[x + 5 * y], RHO[x][y]);
                }
            }

            // Chi step
            for (int y = 0; y < 5; ++y) {
                for (int x = 0; x < 5; ++x) {
                    state[x + 5 * y] = B[x + 5 * y] ^ (~B[((x + 1) % 5) + 5 * y] & B[((x + 2) % 5) + 5 * y]);
                }
            }

            // Iota step
            state[0] ^= RC[round];
        }
    }

    
    void xor_byte(int byte_index, uint8_t byte_val) {
        int word_index = byte_index / 8;
        int shift = (byte_index % 8) * 8;
        state[word_index] ^= ((uint64_t)byte_val << shift);
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
        const int RATE_BYTES = 136;
        size_t i = 0;
        
        while (i < len) {
            // Check if position is 8-byte aligned for fast path
            if ((pos & 7) != 0 || len - i < 8) {
                // Handle bytes individually when unaligned or small remaining
                while (i < len && pos < RATE_BYTES) {
                    state[pos >> 3] ^= ((uint64_t)data[i]) << ((pos & 7) << 3);
                    pos++;
                    i++;
                }
                if (pos == RATE_BYTES) {
                    keccak_f1600();
                    pos = 0;
                }
            } else {
                // Fast path: process 8-byte words directly
                while (i + 8 <= len && pos + 8 <= RATE_BYTES) {
                    uint64_t word;
                    std::memcpy(&word, data + i, 8);
                    state[pos >> 3] ^= word;
                    i += 8;
                    pos += 8;
                }
                if (pos == RATE_BYTES) {
                    keccak_f1600();
                    pos = 0;
                }
            }
        }
    }

    
    void update(const std::string& text) {
        update(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    }

    
    std::vector<uint8_t> finalize() {
        uint64_t saved_state[25];
        std::memcpy(saved_state, state, sizeof(state));
        int saved_pos = pos;

        // Padding: append 0x06 then 0x80 at end
        state[pos >> 3] ^= ((uint64_t)0x06) << ((pos & 7) << 3);
        state[16] ^= 0x8000000000000000ULL; // Set bit at position 1087 (last bit of rate)
        
        keccak_f1600();

        // Extract hash (first 32 bytes)
        std::vector<uint8_t> hash(32);
        std::memcpy(hash.data(), state, 32);

        std::memcpy(state, saved_state, sizeof(state));
        pos = saved_pos;

        return hash;
    }

    
    std::string hexdigest() {
        std::vector<uint8_t> hash = finalize();
        std::ostringstream oss;
        for (uint8_t b : hash) {
            oss << std::hex << std::setw(2) << std::setfill('0') << (int)b;
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
    alignas(8) char buffer[16384];
    while (file.read(buffer, sizeof(buffer))) {
        sha3.update(reinterpret_cast<const uint8_t*>(buffer), file.gcount());
    }
    if (file.gcount() > 0) {
        sha3.update(reinterpret_cast<const uint8_t*>(buffer), file.gcount());
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
