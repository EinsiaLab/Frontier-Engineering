// EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <fstream>

class SHA3_256 {
private:
    alignas(64) uint64_t state[25];
    int pos;

    static inline uint64_t rotl(uint64_t a, int offset) {
        return (a << offset) | (a >> (64 - offset));
    }

    static inline uint64_t load_u64_le(const uint8_t* p) {
        return ((uint64_t)p[0]) | ((uint64_t)p[1] << 8) |
               ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
               ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
               ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
    }

    #define KECCAK_ROUND(rc) do { \
        C0 = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20]; \
        C1 = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21]; \
        C2 = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22]; \
        C3 = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23]; \
        C4 = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24]; \
        D0 = C4 ^ rotl(C1, 1); D1 = C0 ^ rotl(C2, 1); D2 = C1 ^ rotl(C3, 1); \
        D3 = C2 ^ rotl(C4, 1); D4 = C3 ^ rotl(C0, 1); \
        s[0] ^= D0; s[5] ^= D0; s[10] ^= D0; s[15] ^= D0; s[20] ^= D0; \
        s[1] ^= D1; s[6] ^= D1; s[11] ^= D1; s[16] ^= D1; s[21] ^= D1; \
        s[2] ^= D2; s[7] ^= D2; s[12] ^= D2; s[17] ^= D2; s[22] ^= D2; \
        s[3] ^= D3; s[8] ^= D3; s[13] ^= D3; s[18] ^= D3; s[23] ^= D3; \
        s[4] ^= D4; s[9] ^= D4; s[14] ^= D4; s[19] ^= D4; s[24] ^= D4; \
        B[0] = rotl(s[0], 0); B[1] = rotl(s[6], 44); B[2] = rotl(s[12], 43); \
        B[3] = rotl(s[18], 21); B[4] = rotl(s[24], 14); B[5] = rotl(s[3], 28); \
        B[6] = rotl(s[9], 20); B[7] = rotl(s[10], 3); B[8] = rotl(s[16], 45); \
        B[9] = rotl(s[22], 61); B[10] = rotl(s[1], 1); B[11] = rotl(s[7], 6); \
        B[12] = rotl(s[13], 25); B[13] = rotl(s[19], 8); B[14] = rotl(s[20], 18); \
        B[15] = rotl(s[4], 27); B[16] = rotl(s[5], 36); B[17] = rotl(s[11], 10); \
        B[18] = rotl(s[17], 15); B[19] = rotl(s[23], 56); B[20] = rotl(s[2], 62); \
        B[21] = rotl(s[8], 55); B[22] = rotl(s[14], 39); B[23] = rotl(s[15], 41); \
        B[24] = rotl(s[21], 2); \
        s[0] = B[0] ^ (~B[1] & B[2]); s[1] = B[1] ^ (~B[2] & B[3]); \
        s[2] = B[2] ^ (~B[3] & B[4]); s[3] = B[3] ^ (~B[4] & B[0]); \
        s[4] = B[4] ^ (~B[0] & B[1]); s[5] = B[5] ^ (~B[6] & B[7]); \
        s[6] = B[6] ^ (~B[7] & B[8]); s[7] = B[7] ^ (~B[8] & B[9]); \
        s[8] = B[8] ^ (~B[9] & B[5]); s[9] = B[9] ^ (~B[5] & B[6]); \
        s[10] = B[10] ^ (~B[11] & B[12]); s[11] = B[11] ^ (~B[12] & B[13]); \
        s[12] = B[12] ^ (~B[13] & B[14]); s[13] = B[13] ^ (~B[14] & B[10]); \
        s[14] = B[14] ^ (~B[10] & B[11]); s[15] = B[15] ^ (~B[16] & B[17]); \
        s[16] = B[16] ^ (~B[17] & B[18]); s[17] = B[17] ^ (~B[18] & B[19]); \
        s[18] = B[18] ^ (~B[19] & B[15]); s[19] = B[19] ^ (~B[15] & B[16]); \
        s[20] = B[20] ^ (~B[21] & B[22]); s[21] = B[21] ^ (~B[22] & B[23]); \
        s[22] = B[22] ^ (~B[23] & B[24]); s[23] = B[23] ^ (~B[24] & B[20]); \
        s[24] = B[24] ^ (~B[20] & B[21]); s[0] ^= rc; \
    } while(0)

    __attribute__((always_inline)) inline void keccak_f1600() {
        uint64_t* s = state;
        uint64_t C0, C1, C2, C3, C4, D0, D1, D2, D3, D4;
        uint64_t B[25];

        KECCAK_ROUND(0x0000000000000001ULL);
        KECCAK_ROUND(0x0000000000008082ULL);
        KECCAK_ROUND(0x800000000000808aULL);
        KECCAK_ROUND(0x8000000080008000ULL);
        KECCAK_ROUND(0x000000000000808bULL);
        KECCAK_ROUND(0x0000000080000001ULL);
        KECCAK_ROUND(0x8000000080008081ULL);
        KECCAK_ROUND(0x8000000000008009ULL);
        KECCAK_ROUND(0x000000000000008aULL);
        KECCAK_ROUND(0x0000000000000088ULL);
        KECCAK_ROUND(0x0000000080008009ULL);
        KECCAK_ROUND(0x000000008000000aULL);
        KECCAK_ROUND(0x000000008000808bULL);
        KECCAK_ROUND(0x800000000000008bULL);
        KECCAK_ROUND(0x8000000000008089ULL);
        KECCAK_ROUND(0x8000000000008003ULL);
        KECCAK_ROUND(0x8000000000008002ULL);
        KECCAK_ROUND(0x8000000000000080ULL);
        KECCAK_ROUND(0x000000000000800aULL);
        KECCAK_ROUND(0x800000008000000aULL);
        KECCAK_ROUND(0x8000000080008081ULL);
        KECCAK_ROUND(0x8000000000008080ULL);
        KECCAK_ROUND(0x0000000080000001ULL);
        KECCAK_ROUND(0x8000000080008008ULL);
    }
    #undef KECCAK_ROUND

public:
    SHA3_256() { reset(); }

    void reset() {
        std::memset(state, 0, sizeof(state));
        pos = 0;
    }

    void update(const uint8_t* data, size_t len) {
        const int RATE_BYTES = 136;
        size_t i = 0;

        // Phase 1: Complete any partial block
        if (pos > 0) {
            while (pos < RATE_BYTES && i < len) {
                state[pos >> 3] ^= ((uint64_t)data[i] << ((pos & 7) << 3));
                pos++;
                i++;
            }
            if (pos == RATE_BYTES) {
                keccak_f1600();
                pos = 0;
            }
        }

        // Phase 2: Process full blocks
        while (i + RATE_BYTES <= len) {
            const uint8_t* p = data + i;
            state[0] ^= load_u64_le(p); p += 8;
            state[1] ^= load_u64_le(p); p += 8;
            state[2] ^= load_u64_le(p); p += 8;
            state[3] ^= load_u64_le(p); p += 8;
            state[4] ^= load_u64_le(p); p += 8;
            state[5] ^= load_u64_le(p); p += 8;
            state[6] ^= load_u64_le(p); p += 8;
            state[7] ^= load_u64_le(p); p += 8;
            state[8] ^= load_u64_le(p); p += 8;
            state[9] ^= load_u64_le(p); p += 8;
            state[10] ^= load_u64_le(p); p += 8;
            state[11] ^= load_u64_le(p); p += 8;
            state[12] ^= load_u64_le(p); p += 8;
            state[13] ^= load_u64_le(p); p += 8;
            state[14] ^= load_u64_le(p); p += 8;
            state[15] ^= load_u64_le(p); p += 8;
            state[16] ^= load_u64_le(p);
            keccak_f1600();
            i += RATE_BYTES;
        }

        // Phase 3: Process remaining bytes
        while (i < len) {
            state[pos >> 3] ^= ((uint64_t)data[i] << ((pos & 7) << 3));
            pos++;
            i++;
        }
    }

    void update(const std::string& text) {
        update(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    }

    std::vector<uint8_t> finalize() {
        uint64_t saved_state[25];
        std::memcpy(saved_state, state, sizeof(state));
        int saved_pos = pos;

        int word_idx = pos >> 3;
        int shift = (pos & 7) << 3;
        state[word_idx] ^= ((uint64_t)0x06 << shift);
        state[16] ^= ((uint64_t)0x80 << 56);

        keccak_f1600();

        std::vector<uint8_t> hash(32);
        for (int i = 0; i < 4; ++i) {
            uint64_t w = state[i];
            hash[i*8 + 0] = (uint8_t)(w);
            hash[i*8 + 1] = (uint8_t)(w >> 8);
            hash[i*8 + 2] = (uint8_t)(w >> 16);
            hash[i*8 + 3] = (uint8_t)(w >> 24);
            hash[i*8 + 4] = (uint8_t)(w >> 32);
            hash[i*8 + 5] = (uint8_t)(w >> 40);
            hash[i*8 + 6] = (uint8_t)(w >> 48);
            hash[i*8 + 7] = (uint8_t)(w >> 56);
        }

        std::memcpy(state, saved_state, sizeof(state));
        pos = saved_pos;

        return hash;
    }

    std::string hexdigest() {
        std::vector<uint8_t> hash = finalize();
        static const char hex_chars[] = "0123456789abcdef";
        std::string result;
        result.reserve(64);
        for (uint8_t b : hash) {
            result.push_back(hex_chars[b >> 4]);
            result.push_back(hex_chars[b & 0xf]);
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
    char buffer[65536];
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