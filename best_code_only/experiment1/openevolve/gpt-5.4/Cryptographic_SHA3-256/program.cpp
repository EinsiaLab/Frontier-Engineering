// EVOLVE-BLOCK-START
#include <iostream>
#include <string>
#include <cstdint>

#include <cstdio>
#include <cstring>

class SHA3_256 {
private:
    static constexpr int RATE_BYTES = 136;
    static constexpr int RATE_LANES = 17;
    uint64_t state[25];
    int pos;

    static inline uint64_t rotl(uint64_t a, int offset) {
        return offset ? ((a << offset) | (a >> (64 - offset))) : a;
    }

    static inline uint64_t load64(const uint8_t* p) {
        uint64_t v;
        std::memcpy(&v, p, sizeof(v));
        return v;
    }

    void keccak_f1600() {
        static const uint64_t RC[24] = {
            0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,
            0x8000000080008000ULL,0x000000000000808bULL,0x0000000080000001ULL,
            0x8000000080008081ULL,0x8000000000008009ULL,0x000000000000008aULL,
            0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
            0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,
            0x8000000000008003ULL,0x8000000000008002ULL,0x8000000000000080ULL,
            0x000000000000800aULL,0x800000008000000aULL,0x8000000080008081ULL,
            0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL
        };
        uint64_t* s = state;
        for (int round = 0; round < 24; ++round) {
            uint64_t C0 = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
            uint64_t C1 = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
            uint64_t C2 = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
            uint64_t C3 = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
            uint64_t C4 = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

            uint64_t D0 = C4 ^ rotl(C1, 1);
            uint64_t D1 = C0 ^ rotl(C2, 1);
            uint64_t D2 = C1 ^ rotl(C3, 1);
            uint64_t D3 = C2 ^ rotl(C4, 1);
            uint64_t D4 = C3 ^ rotl(C0, 1);

            s[0] ^= D0;  s[1] ^= D1;  s[2] ^= D2;  s[3] ^= D3;  s[4] ^= D4;
            s[5] ^= D0;  s[6] ^= D1;  s[7] ^= D2;  s[8] ^= D3;  s[9] ^= D4;
            s[10] ^= D0; s[11] ^= D1; s[12] ^= D2; s[13] ^= D3; s[14] ^= D4;
            s[15] ^= D0; s[16] ^= D1; s[17] ^= D2; s[18] ^= D3; s[19] ^= D4;
            s[20] ^= D0; s[21] ^= D1; s[22] ^= D2; s[23] ^= D3; s[24] ^= D4;

            uint64_t B0  = s[0];
            uint64_t B1  = rotl(s[6], 44);
            uint64_t B2  = rotl(s[12], 43);
            uint64_t B3  = rotl(s[18], 21);
            uint64_t B4  = rotl(s[24], 14);
            uint64_t B5  = rotl(s[3], 28);
            uint64_t B6  = rotl(s[9], 20);
            uint64_t B7  = rotl(s[10], 3);
            uint64_t B8  = rotl(s[16], 45);
            uint64_t B9  = rotl(s[22], 61);
            uint64_t B10 = rotl(s[1], 1);
            uint64_t B11 = rotl(s[7], 6);
            uint64_t B12 = rotl(s[13], 25);
            uint64_t B13 = rotl(s[19], 8);
            uint64_t B14 = rotl(s[20], 18);
            uint64_t B15 = rotl(s[4], 27);
            uint64_t B16 = rotl(s[5], 36);
            uint64_t B17 = rotl(s[11], 10);
            uint64_t B18 = rotl(s[17], 15);
            uint64_t B19 = rotl(s[23], 56);
            uint64_t B20 = rotl(s[2], 62);
            uint64_t B21 = rotl(s[8], 55);
            uint64_t B22 = rotl(s[14], 39);
            uint64_t B23 = rotl(s[15], 41);
            uint64_t B24 = rotl(s[21], 2);

            s[0]  = B0  ^ (~B1  & B2);
            s[1]  = B1  ^ (~B2  & B3);
            s[2]  = B2  ^ (~B3  & B4);
            s[3]  = B3  ^ (~B4  & B0);
            s[4]  = B4  ^ (~B0  & B1);
            s[5]  = B5  ^ (~B6  & B7);
            s[6]  = B6  ^ (~B7  & B8);
            s[7]  = B7  ^ (~B8  & B9);
            s[8]  = B8  ^ (~B9  & B5);
            s[9]  = B9  ^ (~B5  & B6);
            s[10] = B10 ^ (~B11 & B12);
            s[11] = B11 ^ (~B12 & B13);
            s[12] = B12 ^ (~B13 & B14);
            s[13] = B13 ^ (~B14 & B10);
            s[14] = B14 ^ (~B10 & B11);
            s[15] = B15 ^ (~B16 & B17);
            s[16] = B16 ^ (~B17 & B18);
            s[17] = B17 ^ (~B18 & B19);
            s[18] = B18 ^ (~B19 & B15);
            s[19] = B19 ^ (~B15 & B16);
            s[20] = B20 ^ (~B21 & B22);
            s[21] = B21 ^ (~B22 & B23);
            s[22] = B22 ^ (~B23 & B24);
            s[23] = B23 ^ (~B24 & B20);
            s[24] = B24 ^ (~B20 & B21);

            s[0] ^= RC[round];
        }
    }

    
    inline void xor_byte(int byte_index, uint8_t byte_val) {
        state[byte_index >> 3] ^= uint64_t(byte_val) << ((byte_index & 7) * 8);
    }

public:
    SHA3_256() {
        reset();
    }

    
    void reset() {
        std::memset(state, 0, sizeof(state));
        pos = 0;
    }

    void update(const uint8_t* data, size_t len) {
        size_t i = 0;

        if (pos) {
            while (pos < RATE_BYTES && i < len) xor_byte(pos++, data[i++]);
            if (pos == RATE_BYTES) {
                keccak_f1600();
                pos = 0;
            }
        }

        while (i + RATE_BYTES <= len) {
            for (int j = 0; j < RATE_LANES; ++j) state[j] ^= load64(data + i + (j << 3));
            keccak_f1600();
            i += RATE_BYTES;
        }

        while (i + 8 <= len) {
            state[pos >> 3] ^= load64(data + i);
            pos += 8;
            i += 8;
            if (pos == RATE_BYTES) {
                keccak_f1600();
                pos = 0;
            }
        }

        while (i < len) xor_byte(pos++, data[i++]);
    }



    std::string hexdigest() {
        uint64_t saved_state[25];
        std::memcpy(saved_state, state, sizeof(state));
        int saved_pos = pos;

        xor_byte(pos, 0x06);
        xor_byte(RATE_BYTES - 1, 0x80);
        keccak_f1600();

        static const char hex[] = "0123456789abcdef";
        std::string out(64, '0');
        for (int i = 0; i < 32; ++i) {
            uint8_t b = (state[i / 8] >> (8 * (i % 8))) & 0xFF;
            out[2 * i] = hex[b >> 4];
            out[2 * i + 1] = hex[b & 15];
        }

        std::memcpy(state, saved_state, sizeof(state));
        pos = saved_pos;
        return out;
    }
};




std::string hash_file(const std::string& filepath) {
    std::FILE* file = std::fopen(filepath.c_str(), "rb");
    if (!file) return "Error: Could not open file.";

    SHA3_256 sha3;
    uint8_t buffer[1 << 15];
    size_t n;
    while ((n = std::fread(buffer, 1, sizeof(buffer), file)) != 0) sha3.update(buffer, n);
    std::fclose(file);
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
