// EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>

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

static inline uint64_t rotl64(uint64_t a, int offset) {
    return (a << offset) | (a >> (64 - offset));
}

static void keccak_f1600(uint64_t* __restrict__ state) {
    uint64_t s00=state[0],  s01=state[1],  s02=state[2],  s03=state[3],  s04=state[4];
    uint64_t s05=state[5],  s06=state[6],  s07=state[7],  s08=state[8],  s09=state[9];
    uint64_t s10=state[10], s11=state[11], s12=state[12], s13=state[13], s14=state[14];
    uint64_t s15=state[15], s16=state[16], s17=state[17], s18=state[18], s19=state[19];
    uint64_t s20=state[20], s21=state[21], s22=state[22], s23=state[23], s24=state[24];

    for (int round = 0; round < 24; ++round) {
        // Theta
        uint64_t c0 = s00 ^ s05 ^ s10 ^ s15 ^ s20;
        uint64_t c1 = s01 ^ s06 ^ s11 ^ s16 ^ s21;
        uint64_t c2 = s02 ^ s07 ^ s12 ^ s17 ^ s22;
        uint64_t c3 = s03 ^ s08 ^ s13 ^ s18 ^ s23;
        uint64_t c4 = s04 ^ s09 ^ s14 ^ s19 ^ s24;

        uint64_t d0 = c4 ^ rotl64(c1, 1);
        uint64_t d1 = c0 ^ rotl64(c2, 1);
        uint64_t d2 = c1 ^ rotl64(c3, 1);
        uint64_t d3 = c2 ^ rotl64(c4, 1);
        uint64_t d4 = c3 ^ rotl64(c0, 1);

        s00 ^= d0; s01 ^= d1; s02 ^= d2; s03 ^= d3; s04 ^= d4;
        s05 ^= d0; s06 ^= d1; s07 ^= d2; s08 ^= d3; s09 ^= d4;
        s10 ^= d0; s11 ^= d1; s12 ^= d2; s13 ^= d3; s14 ^= d4;
        s15 ^= d0; s16 ^= d1; s17 ^= d2; s18 ^= d3; s19 ^= d4;
        s20 ^= d0; s21 ^= d1; s22 ^= d2; s23 ^= d3; s24 ^= d4;

        // Rho + Pi
        uint64_t t00 = s00;
        uint64_t t01 = rotl64(s06, 44);
        uint64_t t02 = rotl64(s12, 43);
        uint64_t t03 = rotl64(s18, 21);
        uint64_t t04 = rotl64(s24, 14);
        uint64_t t05 = rotl64(s03, 28);
        uint64_t t06 = rotl64(s09, 20);
        uint64_t t07 = rotl64(s10, 3);
        uint64_t t08 = rotl64(s16, 45);
        uint64_t t09 = rotl64(s22, 61);
        uint64_t t10 = rotl64(s01, 1);
        uint64_t t11 = rotl64(s07, 6);
        uint64_t t12 = rotl64(s13, 25);
        uint64_t t13 = rotl64(s19, 8);
        uint64_t t14 = rotl64(s20, 18);
        uint64_t t15 = rotl64(s04, 27);
        uint64_t t16 = rotl64(s05, 36);
        uint64_t t17 = rotl64(s11, 10);
        uint64_t t18 = rotl64(s17, 15);
        uint64_t t19 = rotl64(s23, 56);
        uint64_t t20 = rotl64(s02, 62);
        uint64_t t21 = rotl64(s08, 55);
        uint64_t t22 = rotl64(s14, 39);
        uint64_t t23 = rotl64(s15, 41);
        uint64_t t24 = rotl64(s21, 2);

        // Chi
        s00 = t00 ^ (~t01 & t02); s01 = t01 ^ (~t02 & t03); s02 = t02 ^ (~t03 & t04); s03 = t03 ^ (~t04 & t00); s04 = t04 ^ (~t00 & t01);
        s05 = t05 ^ (~t06 & t07); s06 = t06 ^ (~t07 & t08); s07 = t07 ^ (~t08 & t09); s08 = t08 ^ (~t09 & t05); s09 = t09 ^ (~t05 & t06);
        s10 = t10 ^ (~t11 & t12); s11 = t11 ^ (~t12 & t13); s12 = t12 ^ (~t13 & t14); s13 = t13 ^ (~t14 & t10); s14 = t14 ^ (~t10 & t11);
        s15 = t15 ^ (~t16 & t17); s16 = t16 ^ (~t17 & t18); s17 = t17 ^ (~t18 & t19); s18 = t18 ^ (~t19 & t15); s19 = t19 ^ (~t15 & t16);
        s20 = t20 ^ (~t21 & t22); s21 = t21 ^ (~t22 & t23); s22 = t22 ^ (~t23 & t24); s23 = t23 ^ (~t24 & t20); s24 = t24 ^ (~t20 & t21);

        // Iota
        s00 ^= RC[round];
    }

    state[0]=s00;  state[1]=s01;  state[2]=s02;  state[3]=s03;  state[4]=s04;
    state[5]=s05;  state[6]=s06;  state[7]=s07;  state[8]=s08;  state[9]=s09;
    state[10]=s10; state[11]=s11; state[12]=s12; state[13]=s13; state[14]=s14;
    state[15]=s15; state[16]=s16; state[17]=s17; state[18]=s18; state[19]=s19;
    state[20]=s20; state[21]=s21; state[22]=s22; state[23]=s23; state[24]=s24;
}

static const int RATE_BYTES = 136;
static const int RATE_WORDS = 17;

static std::string hash_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return "Error: Could not open file.";
    }

    uint64_t state[25];
    memset(state, 0, sizeof(state));
    int pos = 0;

    alignas(64) uint8_t buffer[65536];
    while (true) {
        file.read(reinterpret_cast<char*>(buffer), sizeof(buffer));
        size_t len = file.gcount();
        if (len == 0) break;

        const uint8_t* data = buffer;
        if (pos > 0) {
            size_t need = RATE_BYTES - pos;
            if (len < need) {
                for (size_t i = 0; i < len; i++) {
                    int wi = (pos + i) / 8;
                    int sh = ((pos + i) % 8) * 8;
                    state[wi] ^= ((uint64_t)data[i] << sh);
                }
                pos += len;
                continue;
            }
            for (size_t i = 0; i < need; i++) {
                int wi = (pos + i) / 8;
                int sh = ((pos + i) % 8) * 8;
                state[wi] ^= ((uint64_t)data[i] << sh);
            }
            keccak_f1600(state);
            data += need;
            len -= need;
            pos = 0;
        }

        while (len >= RATE_BYTES) {
            const uint64_t* words = reinterpret_cast<const uint64_t*>(data);
            for (int i = 0; i < RATE_WORDS; i++) {
                uint64_t w;
                memcpy(&w, data + i*8, 8);
                state[i] ^= w;
            }
            keccak_f1600(state);
            data += RATE_BYTES;
            len -= RATE_BYTES;
        }

        if (len > 0) {
            for (size_t i = 0; i < len; i++) {
                int wi = i / 8;
                int sh = (i % 8) * 8;
                state[wi] ^= ((uint64_t)data[i] << sh);
            }
            pos = len;
        }
    }

    // Padding
    {
        int wi = pos / 8;
        int sh = (pos % 8) * 8;
        state[wi] ^= ((uint64_t)0x06 << sh);
    }
    {
        int wi = 135 / 8;
        int sh = (135 % 8) * 8;
        state[wi] ^= ((uint64_t)0x80 << sh);
    }
    keccak_f1600(state);

    static const char hex_chars[] = "0123456789abcdef";
    char result[64];
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(state);
    for (int i = 0; i < 32; i++) {
        uint8_t b = bytes[i];
        result[i*2]   = hex_chars[b >> 4];
        result[i*2+1] = hex_chars[b & 0x0f];
    }
    return std::string(result, 64);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        return 1;
    }

    std::cout << hash_file(argv[1]);
    return 0;
}
// EVOLVE-BLOCK-END
