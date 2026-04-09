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

    
    inline uint64_t rotl(uint64_t a, int offset) {
        if (offset == 0) return a;
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
            
            // Compute theta step with unrolled loops for better performance
            uint64_t C[5];
            C[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
            C[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
            C[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
            C[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
            C[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];
            
            uint64_t D[5];
            D[0] = C[4] ^ rotl(C[1], 1);
            D[1] = C[0] ^ rotl(C[2], 1);
            D[2] = C[1] ^ rotl(C[3], 1);
            D[3] = C[2] ^ rotl(C[4], 1);
            D[4] = C[3] ^ rotl(C[0], 1);
            
            // Apply D to all columns
            for (int y = 0; y < 5; ++y) {
                state[0 + 5*y] ^= D[0];
                state[1 + 5*y] ^= D[1];
                state[2 + 5*y] ^= D[2];
                state[3 + 5*y] ^= D[3];
                state[4 + 5*y] ^= D[4];
            }

            
            uint64_t B[25];
            for (int x = 0; x < 5; ++x) {
                for (int y = 0; y < 5; ++y) {
                    B[y + 5 * ((2 * x + 3 * y) % 5)] = rotl(state[x + 5 * y], RHO[x][y]);
                }
            }

            
            // Unroll chi step for better performance
            for (int y = 0; y < 5; ++y) {
                int base = 5 * y;
                state[base]   = B[base]   ^ (~B[base + 1] & B[base + 2]);
                state[base+1] = B[base+1] ^ (~B[base + 2] & B[base + 3]);
                state[base+2] = B[base+2] ^ (~B[base + 3] & B[base + 4]);
                state[base+3] = B[base+3] ^ (~B[base + 4] & B[base]);
                state[base+4] = B[base+4] ^ (~B[base]     & B[base + 1]);
            }

            
            state[0] ^= RC[round];
        }
    }

    
    void xor_byte(int byte_index, uint8_t byte_val) {
        // Use bit operations instead of division and modulo
        int word_index = byte_index >> 3;  // Divide by 8
        int shift = (byte_index & 7) << 3;  // Modulo 8, then multiply by 8
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
        // Process word-aligned chunks when possible
        if (pos % 8 == 0) {
            while (i < len) {
                // Check if we can process a full word
                if ((len - i) >= 8 && (pos % 8) == 0 && pos + 8 <= RATE_BYTES) {
                    int word_index = pos / 8;
                    uint64_t word = 0;
                    // Read 8 bytes into a word (little-endian)
                    for (int j = 0; j < 8; ++j) {
                        word |= (static_cast<uint64_t>(data[i + j]) << (8 * j));
                    }
                    state[word_index] ^= word;
                    i += 8;
                    pos += 8;
                } else {
                    // Fall back to byte processing
                    xor_byte(pos, data[i]);
                    i++;
                    pos++;
                }
                if (pos == RATE_BYTES) {
                    keccak_f1600();
                    pos = 0;
                }
            }
        } else {
            // Original byte-by-byte processing when not aligned
            for (; i < len; ++i) {
                xor_byte(pos, data[i]);
                pos++;
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
        // Save state using memcpy for faster copying
        uint64_t saved_state[25];
        std::memcpy(saved_state, state, 25 * sizeof(uint64_t));
        int saved_pos = pos;

        // Apply padding
        xor_byte(pos, 0x06);
        xor_byte(135, 0x80);
        keccak_f1600();

        // Extract hash
        std::vector<uint8_t> hash(32);
        for (int i = 0; i < 32; ++i) {
            int word_index = i / 8;
            int shift = (i % 8) * 8;
            hash[i] = (uint8_t)((state[word_index] >> shift) & 0xFF);
        }

        // Restore state
        std::memcpy(state, saved_state, 25 * sizeof(uint64_t));
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
    char buffer[16384]; // Larger buffer for better I/O performance
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
