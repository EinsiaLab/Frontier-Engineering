// EVOLVE-BLOCK-START
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <unistd.h>

#ifdef __x86_64__
#include <immintrin.h>
#include <cpuid.h>
#endif

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

#ifdef __x86_64__

static bool has_sha_ni() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx >> 29) & 1; // SHA bit
    }
    return false;
}

__attribute__((target("sha,sse4.1,ssse3")))
static void sha256_transform_shani(uint32_t state[8], const uint8_t *data, size_t nblocks) {
    __m128i STATE0, STATE1;
    __m128i ABEF_SAVE, CDGH_SAVE;
    __m128i MSG, TMP;
    __m128i MSG0, MSG1, MSG2, MSG3;
    __m128i MASK;

    MASK = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);

    // Load initial state: state[0..3] = A B C D, state[4..7] = E F G H
    TMP = _mm_loadu_si128((const __m128i*)(state));     // A B C D
    STATE1 = _mm_loadu_si128((const __m128i*)(state+4)); // E F G H

    TMP = _mm_shuffle_epi32(TMP, 0xB1);       // B A D C
    STATE1 = _mm_shuffle_epi32(STATE1, 0x1B);  // H G F E
    STATE0 = _mm_alignr_epi8(TMP, STATE1, 8);  // A B F G (ABEF)
    STATE1 = _mm_blend_epi16(STATE1, TMP, 0xF0); // C D H E -> actually D C H G (CDGH)

    for (size_t b = 0; b < nblocks; b++) {
        ABEF_SAVE = STATE0;
        CDGH_SAVE = STATE1;

        // Rounds 0-3
        MSG0 = _mm_loadu_si128((const __m128i*)(data));
        MSG0 = _mm_shuffle_epi8(MSG0, MASK);
        MSG = _mm_add_epi32(MSG0, _mm_set_epi64x(0xE9B5DBA5B5C0FBCFULL, 0x7137449128A2F98ULL));
        // Fix: load K properly
        MSG = _mm_add_epi32(MSG0, _mm_loadu_si128((const __m128i*)&K[0]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

        // Rounds 4-7
        MSG1 = _mm_loadu_si128((const __m128i*)(data+16));
        MSG1 = _mm_shuffle_epi8(MSG1, MASK);
        MSG = _mm_add_epi32(MSG1, _mm_loadu_si128((const __m128i*)&K[4]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
        MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);

        // Rounds 8-11
        MSG2 = _mm_loadu_si128((const __m128i*)(data+32));
        MSG2 = _mm_shuffle_epi8(MSG2, MASK);
        MSG = _mm_add_epi32(MSG2, _mm_loadu_si128((const __m128i*)&K[8]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
        MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);

        // Rounds 12-15
        MSG3 = _mm_loadu_si128((const __m128i*)(data+48));
        MSG3 = _mm_shuffle_epi8(MSG3, MASK);
        MSG = _mm_add_epi32(MSG3, _mm_loadu_si128((const __m128i*)&K[12]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        TMP = _mm_alignr_epi8(MSG3, MSG2, 4);
        MSG0 = _mm_add_epi32(MSG0, TMP);
        MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
        MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);

        // Rounds 16-19
        MSG = _mm_add_epi32(MSG0, _mm_loadu_si128((const __m128i*)&K[16]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        TMP = _mm_alignr_epi8(MSG0, MSG3, 4);
        MSG1 = _mm_add_epi32(MSG1, TMP);
        MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
        MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);

        // Rounds 20-23
        MSG = _mm_add_epi32(MSG1, _mm_loadu_si128((const __m128i*)&K[20]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        TMP = _mm_alignr_epi8(MSG1, MSG0, 4);
        MSG2 = _mm_add_epi32(MSG2, TMP);
        MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
        MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);

        // Rounds 24-27
        MSG = _mm_add_epi32(MSG2, _mm_loadu_si128((const __m128i*)&K[24]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        TMP = _mm_alignr_epi8(MSG2, MSG1, 4);
        MSG3 = _mm_add_epi32(MSG3, TMP);
        MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
        MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);

        // Rounds 28-31
        MSG = _mm_add_epi32(MSG3, _mm_loadu_si128((const __m128i*)&K[28]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        TMP = _mm_alignr_epi8(MSG3, MSG2, 4);
        MSG0 = _mm_add_epi32(MSG0, TMP);
        MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
        MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);

        // Rounds 32-35
        MSG = _mm_add_epi32(MSG0, _mm_loadu_si128((const __m128i*)&K[32]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        TMP = _mm_alignr_epi8(MSG0, MSG3, 4);
        MSG1 = _mm_add_epi32(MSG1, TMP);
        MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
        MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);

        // Rounds 36-39
        MSG = _mm_add_epi32(MSG1, _mm_loadu_si128((const __m128i*)&K[36]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        TMP = _mm_alignr_epi8(MSG1, MSG0, 4);
        MSG2 = _mm_add_epi32(MSG2, TMP);
        MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
        MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);

        // Rounds 40-43
        MSG = _mm_add_epi32(MSG2, _mm_loadu_si128((const __m128i*)&K[40]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        TMP = _mm_alignr_epi8(MSG2, MSG1, 4);
        MSG3 = _mm_add_epi32(MSG3, TMP);
        MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
        MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);

        // Rounds 44-47
        MSG = _mm_add_epi32(MSG3, _mm_loadu_si128((const __m128i*)&K[44]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        TMP = _mm_alignr_epi8(MSG3, MSG2, 4);
        MSG0 = _mm_add_epi32(MSG0, TMP);
        MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
        MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);

        // Rounds 48-51
        MSG = _mm_add_epi32(MSG0, _mm_loadu_si128((const __m128i*)&K[48]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        TMP = _mm_alignr_epi8(MSG0, MSG3, 4);
        MSG1 = _mm_add_epi32(MSG1, TMP);
        MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
        MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);

        // Rounds 52-55
        MSG = _mm_add_epi32(MSG1, _mm_loadu_si128((const __m128i*)&K[52]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        TMP = _mm_alignr_epi8(MSG1, MSG0, 4);
        MSG2 = _mm_add_epi32(MSG2, TMP);
        MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

        // Rounds 56-59
        MSG = _mm_add_epi32(MSG2, _mm_loadu_si128((const __m128i*)&K[56]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        TMP = _mm_alignr_epi8(MSG2, MSG1, 4);
        MSG3 = _mm_add_epi32(MSG3, TMP);
        MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

        // Rounds 60-63
        MSG = _mm_add_epi32(MSG3, _mm_loadu_si128((const __m128i*)&K[60]));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        MSG = _mm_shuffle_epi32(MSG, 0x0E);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

        STATE0 = _mm_add_epi32(STATE0, ABEF_SAVE);
        STATE1 = _mm_add_epi32(STATE1, CDGH_SAVE);

        data += 64;
    }

    // Reorder back
    TMP = _mm_shuffle_epi32(STATE0, 0x1B);    // G F B A
    STATE1 = _mm_shuffle_epi32(STATE1, 0xB1);  // H G D C -> actually rearranged
    STATE0 = _mm_blend_epi16(TMP, STATE1, 0xF0);
    STATE1 = _mm_alignr_epi8(STATE1, TMP, 8);

    // Store
    _mm_storeu_si128((__m128i*)(state), STATE0);
    _mm_storeu_si128((__m128i*)(state+4), STATE1);
}

#endif // __x86_64__

static inline uint32_t load_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static void sha256_transform_sw(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    for (int i = 0; i < 16; i++)
        W[i] = load_be32(block + i * 4);
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(W[i-15], 7) ^ rotr(W[i-15], 18) ^ (W[i-15] >> 3);
        uint32_t s1 = rotr(W[i-2], 17) ^ rotr(W[i-2], 19) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K[i] + W[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

static void sha256_hash(const uint8_t *data, size_t len, char *out_hex) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    size_t nblocks = len / 64;

#ifdef __x86_64__
    static bool use_shani = has_sha_ni();
    if (use_shani && nblocks > 0) {
        sha256_transform_shani(state, data, nblocks);
    } else
#endif
    {
        for (size_t i = 0; i < nblocks; i++) {
            sha256_transform_sw(state, data + i * 64);
        }
    }

    uint8_t buf[128];
    size_t rem = len % 64;
    memcpy(buf, data + nblocks * 64, rem);
    buf[rem] = 0x80;
    size_t pad_len;
    if (rem < 56) {
        pad_len = 64;
        memset(buf + rem + 1, 0, 55 - rem);
    } else {
        pad_len = 128;
        memset(buf + rem + 1, 0, 119 - rem);
    }
    uint64_t bitlen = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++)
        buf[pad_len - 1 - i] = (uint8_t)(bitlen >> (i * 8));

#ifdef __x86_64__
    if (use_shani) {
        sha256_transform_shani(state, buf, 1);
        if (pad_len == 128)
            sha256_transform_shani(state, buf + 64, 1);
    } else
#endif
    {
        sha256_transform_sw(state, buf);
        if (pad_len == 128)
            sha256_transform_sw(state, buf + 64);
    }

    static const char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < 8; i++) {
        uint32_t v = state[i];
        out_hex[i*8+0] = hex_chars[(v >> 28) & 0xf];
        out_hex[i*8+1] = hex_chars[(v >> 24) & 0xf];
        out_hex[i*8+2] = hex_chars[(v >> 20) & 0xf];
        out_hex[i*8+3] = hex_chars[(v >> 16) & 0xf];
        out_hex[i*8+4] = hex_chars[(v >> 12) & 0xf];
        out_hex[i*8+5] = hex_chars[(v >> 8) & 0xf];
        out_hex[i*8+6] = hex_chars[(v >> 4) & 0xf];
        out_hex[i*8+7] = hex_chars[v & 0xf];
    }
    out_hex[64] = '\0';
}

int main() {
    std::vector<uint8_t> input;
    input.reserve(2 * 1024 * 1024);
    uint8_t tmp[65536];
    ssize_t n;
    while ((n = read(STDIN_FILENO, tmp, sizeof(tmp))) > 0) {
        input.insert(input.end(), tmp, tmp + n);
    }

    char hex[65];
    sha256_hash(input.data(), input.size(), hex);
    write(STDOUT_FILENO, hex, 64);
    return 0;
}
// EVOLVE-BLOCK-END
