// EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>

// AES-NI support via target attribute
#if defined(__GNUC__) || defined(__clang__)
#define HAS_AESNI_SUPPORT 1
#include <wmmintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#define AESNI_ATTR __attribute__((target("aes,sse4.1,ssse3")))
#else
#define HAS_AESNI_SUPPORT 0
#endif

// ============ S-box and constants ============
static const uint8_t sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

static const uint8_t Rcon[11] = {
    0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36
};

// ============ T-table software fallback ============
static uint32_t Te0[256], Te1[256], Te2[256], Te3[256];
static bool tables_initialized = false;

static inline uint8_t xtime_fn(uint8_t x) {
    return (x << 1) ^ (((x >> 7) & 1) * 0x1b);
}

static void init_tables() {
    if (tables_initialized) return;
    for (int i = 0; i < 256; i++) {
        uint8_t s = sbox[i];
        uint8_t s2 = xtime_fn(s);
        uint8_t s3 = s2 ^ s;
        Te0[i] = ((uint32_t)s2 << 24) | ((uint32_t)s << 16) | ((uint32_t)s << 8) | (uint32_t)s3;
        Te1[i] = ((uint32_t)s3 << 24) | ((uint32_t)s2 << 16) | ((uint32_t)s << 8) | (uint32_t)s;
        Te2[i] = ((uint32_t)s << 24) | ((uint32_t)s3 << 16) | ((uint32_t)s2 << 8) | (uint32_t)s;
        Te3[i] = ((uint32_t)s << 24) | ((uint32_t)s << 16) | ((uint32_t)s3 << 8) | (uint32_t)s2;
    }
    tables_initialized = true;
}

static void KeyExpansion_sw(const uint8_t* key, uint32_t* rk) {
    for (int i = 0; i < 4; i++) {
        rk[i] = ((uint32_t)key[4*i] << 24) | ((uint32_t)key[4*i+1] << 16) |
                 ((uint32_t)key[4*i+2] << 8) | (uint32_t)key[4*i+3];
    }
    for (int i = 4; i < 44; i++) {
        uint32_t temp = rk[i-1];
        if (i % 4 == 0) {
            temp = ((uint32_t)sbox[(temp >> 16) & 0xff] << 24) |
                   ((uint32_t)sbox[(temp >> 8) & 0xff] << 16) |
                   ((uint32_t)sbox[temp & 0xff] << 8) |
                   (uint32_t)sbox[(temp >> 24) & 0xff];
            temp ^= ((uint32_t)Rcon[i/4] << 24);
        }
        rk[i] = rk[i-4] ^ temp;
    }
}

static inline void aes_encrypt_block_sw(const uint8_t* in, uint8_t* out, const uint32_t* rk) {
    uint32_t s0, s1, s2, s3, t0, t1, t2, t3;
    s0 = ((uint32_t)in[0] << 24 | (uint32_t)in[1] << 16 | (uint32_t)in[2] << 8 | (uint32_t)in[3]) ^ rk[0];
    s1 = ((uint32_t)in[4] << 24 | (uint32_t)in[5] << 16 | (uint32_t)in[6] << 8 | (uint32_t)in[7]) ^ rk[1];
    s2 = ((uint32_t)in[8] << 24 | (uint32_t)in[9] << 16 | (uint32_t)in[10] << 8 | (uint32_t)in[11]) ^ rk[2];
    s3 = ((uint32_t)in[12] << 24 | (uint32_t)in[13] << 16 | (uint32_t)in[14] << 8 | (uint32_t)in[15]) ^ rk[3];

    #define AES_ROUND_SW(r) \
        t0 = Te0[(s0>>24)&0xff] ^ Te1[(s1>>16)&0xff] ^ Te2[(s2>>8)&0xff] ^ Te3[s3&0xff] ^ rk[r*4+0]; \
        t1 = Te0[(s1>>24)&0xff] ^ Te1[(s2>>16)&0xff] ^ Te2[(s3>>8)&0xff] ^ Te3[s0&0xff] ^ rk[r*4+1]; \
        t2 = Te0[(s2>>24)&0xff] ^ Te1[(s3>>16)&0xff] ^ Te2[(s0>>8)&0xff] ^ Te3[s1&0xff] ^ rk[r*4+2]; \
        t3 = Te0[(s3>>24)&0xff] ^ Te1[(s0>>16)&0xff] ^ Te2[(s1>>8)&0xff] ^ Te3[s2&0xff] ^ rk[r*4+3]; \
        s0=t0; s1=t1; s2=t2; s3=t3;

    AES_ROUND_SW(1) AES_ROUND_SW(2) AES_ROUND_SW(3) AES_ROUND_SW(4) AES_ROUND_SW(5)
    AES_ROUND_SW(6) AES_ROUND_SW(7) AES_ROUND_SW(8) AES_ROUND_SW(9)
    #undef AES_ROUND_SW

    t0 = ((uint32_t)sbox[(s0>>24)&0xff]<<24) | ((uint32_t)sbox[(s1>>16)&0xff]<<16) |
         ((uint32_t)sbox[(s2>>8)&0xff]<<8) | (uint32_t)sbox[s3&0xff];
    t1 = ((uint32_t)sbox[(s1>>24)&0xff]<<24) | ((uint32_t)sbox[(s2>>16)&0xff]<<16) |
         ((uint32_t)sbox[(s3>>8)&0xff]<<8) | (uint32_t)sbox[s0&0xff];
    t2 = ((uint32_t)sbox[(s2>>24)&0xff]<<24) | ((uint32_t)sbox[(s3>>16)&0xff]<<16) |
         ((uint32_t)sbox[(s0>>8)&0xff]<<8) | (uint32_t)sbox[s1&0xff];
    t3 = ((uint32_t)sbox[(s3>>24)&0xff]<<24) | ((uint32_t)sbox[(s0>>16)&0xff]<<16) |
         ((uint32_t)sbox[(s1>>8)&0xff]<<8) | (uint32_t)sbox[s2&0xff];

    s0 = t0 ^ rk[40]; s1 = t1 ^ rk[41]; s2 = t2 ^ rk[42]; s3 = t3 ^ rk[43];

    out[0]=(s0>>24); out[1]=(s0>>16); out[2]=(s0>>8); out[3]=s0;
    out[4]=(s1>>24); out[5]=(s1>>16); out[6]=(s1>>8); out[7]=s1;
    out[8]=(s2>>24); out[9]=(s2>>16); out[10]=(s2>>8); out[11]=s2;
    out[12]=(s3>>24); out[13]=(s3>>16); out[14]=(s3>>8); out[15]=s3;
}

// ============ AES-NI implementation ============
#if HAS_AESNI_SUPPORT

static __m128i aesni_rk[11];

AESNI_ATTR
static inline __m128i aes_128_key_expansion(__m128i key, __m128i keygened) {
    keygened = _mm_shuffle_epi32(keygened, _MM_SHUFFLE(3,3,3,3));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return _mm_xor_si128(key, keygened);
}

AESNI_ATTR
static void KeyExpansion_ni(const uint8_t* key) {
    aesni_rk[0] = _mm_loadu_si128((const __m128i*)key);
    aesni_rk[1] = aes_128_key_expansion(aesni_rk[0], _mm_aeskeygenassist_si128(aesni_rk[0], 0x01));
    aesni_rk[2] = aes_128_key_expansion(aesni_rk[1], _mm_aeskeygenassist_si128(aesni_rk[1], 0x02));
    aesni_rk[3] = aes_128_key_expansion(aesni_rk[2], _mm_aeskeygenassist_si128(aesni_rk[2], 0x04));
    aesni_rk[4] = aes_128_key_expansion(aesni_rk[3], _mm_aeskeygenassist_si128(aesni_rk[3], 0x08));
    aesni_rk[5] = aes_128_key_expansion(aesni_rk[4], _mm_aeskeygenassist_si128(aesni_rk[4], 0x10));
    aesni_rk[6] = aes_128_key_expansion(aesni_rk[5], _mm_aeskeygenassist_si128(aesni_rk[5], 0x20));
    aesni_rk[7] = aes_128_key_expansion(aesni_rk[6], _mm_aeskeygenassist_si128(aesni_rk[6], 0x40));
    aesni_rk[8] = aes_128_key_expansion(aesni_rk[7], _mm_aeskeygenassist_si128(aesni_rk[7], 0x80));
    aesni_rk[9] = aes_128_key_expansion(aesni_rk[8], _mm_aeskeygenassist_si128(aesni_rk[8], 0x1b));
    aesni_rk[10] = aes_128_key_expansion(aesni_rk[9], _mm_aeskeygenassist_si128(aesni_rk[9], 0x36));
}

AESNI_ATTR
static inline __m128i aes_encrypt_block_ni(__m128i pt) {
    __m128i tmp = _mm_xor_si128(pt, aesni_rk[0]);
    tmp = _mm_aesenc_si128(tmp, aesni_rk[1]);
    tmp = _mm_aesenc_si128(tmp, aesni_rk[2]);
    tmp = _mm_aesenc_si128(tmp, aesni_rk[3]);
    tmp = _mm_aesenc_si128(tmp, aesni_rk[4]);
    tmp = _mm_aesenc_si128(tmp, aesni_rk[5]);
    tmp = _mm_aesenc_si128(tmp, aesni_rk[6]);
    tmp = _mm_aesenc_si128(tmp, aesni_rk[7]);
    tmp = _mm_aesenc_si128(tmp, aesni_rk[8]);
    tmp = _mm_aesenc_si128(tmp, aesni_rk[9]);
    return _mm_aesenclast_si128(tmp, aesni_rk[10]);
}

AESNI_ATTR
static void aes_ctr_process_ni(const uint8_t* data, uint8_t* result, size_t len,
                                const uint8_t* key, const uint8_t* iv) {
    KeyExpansion_ni(key);

    const __m128i bswap_mask = _mm_set_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    __m128i ctr_be = _mm_loadu_si128((const __m128i*)iv);
    __m128i ctr_le = _mm_shuffle_epi8(ctr_be, bswap_mask);

    size_t fullBlocks = len / 16;
    size_t remaining = len % 16;

    const __m128i one = _mm_set_epi64x(0, 1);
    const __m128i carry_add = _mm_set_epi64x(1, 0);

    size_t i = 0;

    // Process 8 blocks at a time for maximum ILP
    for (; i + 7 < fullBlocks; i += 8) {
        uint64_t low = (uint64_t)_mm_extract_epi64(ctr_le, 0);

        __m128i c0 = ctr_le;
        __m128i c1 = _mm_add_epi64(ctr_le, _mm_set_epi64x(0, 1));
        __m128i c2 = _mm_add_epi64(ctr_le, _mm_set_epi64x(0, 2));
        __m128i c3 = _mm_add_epi64(ctr_le, _mm_set_epi64x(0, 3));
        __m128i c4 = _mm_add_epi64(ctr_le, _mm_set_epi64x(0, 4));
        __m128i c5 = _mm_add_epi64(ctr_le, _mm_set_epi64x(0, 5));
        __m128i c6 = _mm_add_epi64(ctr_le, _mm_set_epi64x(0, 6));
        __m128i c7 = _mm_add_epi64(ctr_le, _mm_set_epi64x(0, 7));

        // Handle carries
        if (low + 1 < low) c1 = _mm_add_epi64(c1, carry_add);
        if (low + 2 < low) c2 = _mm_add_epi64(c2, carry_add);
        if (low + 3 < low) c3 = _mm_add_epi64(c3, carry_add);
        if (low + 4 < low) c4 = _mm_add_epi64(c4, carry_add);
        if (low + 5 < low) c5 = _mm_add_epi64(c5, carry_add);
        if (low + 6 < low) c6 = _mm_add_epi64(c6, carry_add);
        if (low + 7 < low) c7 = _mm_add_epi64(c7, carry_add);

        // Convert to big-endian for AES
        __m128i b0 = _mm_shuffle_epi8(c0, bswap_mask);
        __m128i b1 = _mm_shuffle_epi8(c1, bswap_mask);
        __m128i b2 = _mm_shuffle_epi8(c2, bswap_mask);
        __m128i b3 = _mm_shuffle_epi8(c3, bswap_mask);
        __m128i b4 = _mm_shuffle_epi8(c4, bswap_mask);
        __m128i b5 = _mm_shuffle_epi8(c5, bswap_mask);
        __m128i b6 = _mm_shuffle_epi8(c6, bswap_mask);
        __m128i b7 = _mm_shuffle_epi8(c7, bswap_mask);

        // Encrypt 8 blocks pipelined
        __m128i e0 = _mm_xor_si128(b0, aesni_rk[0]);
        __m128i e1 = _mm_xor_si128(b1, aesni_rk[0]);
        __m128i e2 = _mm_xor_si128(b2, aesni_rk[0]);
        __m128i e3 = _mm_xor_si128(b3, aesni_rk[0]);
        __m128i e4 = _mm_xor_si128(b4, aesni_rk[0]);
        __m128i e5 = _mm_xor_si128(b5, aesni_rk[0]);
        __m128i e6 = _mm_xor_si128(b6, aesni_rk[0]);
        __m128i e7 = _mm_xor_si128(b7, aesni_rk[0]);

        #define ROUND8(r) \
            e0 = _mm_aesenc_si128(e0, aesni_rk[r]); \
            e1 = _mm_aesenc_si128(e1, aesni_rk[r]); \
            e2 = _mm_aesenc_si128(e2, aesni_rk[r]); \
            e3 = _mm_aesenc_si128(e3, aesni_rk[r]); \
            e4 = _mm_aesenc_si128(e4, aesni_rk[r]); \
            e5 = _mm_aesenc_si128(e5, aesni_rk[r]); \
            e6 = _mm_aesenc_si128(e6, aesni_rk[r]); \
            e7 = _mm_aesenc_si128(e7, aesni_rk[r]);

        ROUND8(1) ROUND8(2) ROUND8(3) ROUND8(4) ROUND8(5)
        ROUND8(6) ROUND8(7) ROUND8(8) ROUND8(9)
        #undef ROUND8

        e0 = _mm_aesenclast_si128(e0, aesni_rk[10]);
        e1 = _mm_aesenclast_si128(e1, aesni_rk[10]);
        e2 = _mm_aesenclast_si128(e2, aesni_rk[10]);
        e3 = _mm_aesenclast_si128(e3, aesni_rk[10]);
        e4 = _mm_aesenclast_si128(e4, aesni_rk[10]);
        e5 = _mm_aesenclast_si128(e5, aesni_rk[10]);
        e6 = _mm_aesenclast_si128(e6, aesni_rk[10]);
        e7 = _mm_aesenclast_si128(e7, aesni_rk[10]);

        // XOR with plaintext and store
        const uint8_t* dp = data + i * 16;
        uint8_t* rp = result + i * 16;
        _mm_storeu_si128((__m128i*)(rp),      _mm_xor_si128(_mm_loadu_si128((const __m128i*)(dp)),      e0));
        _mm_storeu_si128((__m128i*)(rp + 16),  _mm_xor_si128(_mm_loadu_si128((const __m128i*)(dp + 16)),  e1));
        _mm_storeu_si128((__m128i*)(rp + 32),  _mm_xor_si128(_mm_loadu_si128((const __m128i*)(dp + 32)),  e2));
        _mm_storeu_si128((__m128i*)(rp + 48),  _mm_xor_si128(_mm_loadu_si128((const __m128i*)(dp + 48)),  e3));
        _mm_storeu_si128((__m128i*)(rp + 64),  _mm_xor_si128(_mm_loadu_si128((const __m128i*)(dp + 64)),  e4));
        _mm_storeu_si128((__m128i*)(rp + 80),  _mm_xor_si128(_mm_loadu_si128((const __m128i*)(dp + 80)),  e5));
        _mm_storeu_si128((__m128i*)(rp + 96),  _mm_xor_si128(_mm_loadu_si128((const __m128i*)(dp + 96)),  e6));
        _mm_storeu_si128((__m128i*)(rp + 112), _mm_xor_si128(_mm_loadu_si128((const __m128i*)(dp + 112)), e7));

        // Advance counter
        uint64_t new_low = low + 8;
        ctr_le = _mm_add_epi64(ctr_le, _mm_set_epi64x(0, 8));
        if (new_low < low) ctr_le = _mm_add_epi64(ctr_le, carry_add);
    }

    // Process 4 blocks at a time
    for (; i + 3 < fullBlocks; i += 4) {
        uint64_t low = (uint64_t)_mm_extract_epi64(ctr_le, 0);
        __m128i c0 = ctr_le;
        __m128i c1 = _mm_add_epi64(ctr_le, _mm_set_epi64x(0, 1));
        __m128i c2 = _mm_add_epi64(ctr_le, _mm_set_epi64x(0, 2));
        __m128i c3 = _mm_add_epi64(ctr_le, _mm_set_epi64x(0, 3));
        if (low + 1 < low) c1 = _mm_add_epi64(c1, carry_add);
        if (low + 2 < low) c2 = _mm_add_epi64(c2, carry_add);
        if (low + 3 < low) c3 = _mm_add_epi64(c3, carry_add);

        __m128i b0 = _mm_shuffle_epi8(c0, bswap_mask);
        __m128i b1 = _mm_shuffle_epi8(c1, bswap_mask);
        __m128i b2 = _mm_shuffle_epi8(c2, bswap_mask);
        __m128i b3 = _mm_shuffle_epi8(c3, bswap_mask);

        __m128i e0 = _mm_xor_si128(b0, aesni_rk[0]);
        __m128i e1 = _mm_xor_si128(b1, aesni_rk[0]);
        __m128i e2 = _mm_xor_si128(b2, aesni_rk[0]);
        __m128i e3 = _mm_xor_si128(b3, aesni_rk[0]);

        #define ROUND4(r) \
            e0 = _mm_aesenc_si128(e0, aesni_rk[r]); \
            e1 = _mm_aesenc_si128(e1, aesni_rk[r]); \
            e2 = _mm_aesenc_si128(e2, aesni_rk[r]); \
            e3 = _mm_aesenc_si128(e3, aesni_rk[r]);
        ROUND4(1) ROUND4(2) ROUND4(3) ROUND4(4) ROUND4(5)
        ROUND4(6) ROUND4(7) ROUND4(8) ROUND4(9)
        #undef ROUND4

        e0 = _mm_aesenclast_si128(e0, aesni_rk[10]);
        e1 = _mm_aesenclast_si128(e1, aesni_rk[10]);
        e2 = _mm_aesenclast_si128(e2, aesni_rk[10]);
        e3 = _mm_aesenclast_si128(e3, aesni_rk[10]);

        const uint8_t* dp = data + i * 16;
        uint8_t* rp = result + i * 16;
        _mm_storeu_si128((__m128i*)(rp),      _mm_xor_si128(_mm_loadu_si128((const __m128i*)(dp)),      e0));
        _mm_storeu_si128((__m128i*)(rp + 16),  _mm_xor_si128(_mm_loadu_si128((const __m128i*)(dp + 16)),  e1));
        _mm_storeu_si128((__m128i*)(rp + 32),  _mm_xor_si128(_mm_loadu_si128((const __m128i*)(dp + 32)),  e2));
        _mm_storeu_si128((__m128i*)(rp + 48),  _mm_xor_si128(_mm_loadu_si128((const __m128i*)(dp + 48)),  e3));

        uint64_t new_low = low + 4;
        ctr_le = _mm_add_epi64(ctr_le, _mm_set_epi64x(0, 4));
        if (new_low < low) ctr_le = _mm_add_epi64(ctr_le, carry_add);
    }

    // Remaining full blocks
    for (; i < fullBlocks; i++) {
        __m128i ctr = _mm_shuffle_epi8(ctr_le, bswap_mask);
        __m128i enc = aes_encrypt_block_ni(ctr);
        __m128i d = _mm_loadu_si128((const __m128i*)(data + i*16));
        _mm_storeu_si128((__m128i*)(result + i*16), _mm_xor_si128(d, enc));
        uint64_t low = (uint64_t)_mm_extract_epi64(ctr_le, 0);
        ctr_le = _mm_add_epi64(ctr_le, one);
        if (low + 1 < low) ctr_le = _mm_add_epi64(ctr_le, carry_add);
    }

    // Remaining bytes
    if (remaining > 0) {
        __m128i ctr = _mm_shuffle_epi8(ctr_le, bswap_mask);
        __m128i enc = aes_encrypt_block_ni(ctr);
        uint8_t ks[16];
        _mm_storeu_si128((__m128i*)ks, enc);
        for (size_t j = 0; j < remaining; j++) {
            result[fullBlocks*16 + j] = data[fullBlocks*16 + j] ^ ks[j];
        }
    }
}

AESNI_ATTR
static bool check_aesni() {
#if defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(1));
    return ((ecx & (1 << 25)) != 0) && ((ecx & (1 << 19)) != 0);
#else
    return false;
#endif
}

#endif // HAS_AESNI_SUPPORT

// ============ CTR class ============
class AES128_CTR {
private:
    uint32_t rk[44];
    bool keyExpanded;
    uint8_t lastKey[16];
    bool use_aesni;
    uint8_t lastKey_ni[16];
    bool keyExpanded_ni;

public:
    AES128_CTR() : keyExpanded(false), use_aesni(false), keyExpanded_ni(false) {
        memset(lastKey, 0, 16);
        memset(lastKey_ni, 0, 16);
#if HAS_AESNI_SUPPORT
        use_aesni = check_aesni();
#endif
    }

    void process(const uint8_t* data, uint8_t* result, size_t len,
                 const uint8_t* key, const uint8_t* iv) {
        if (len == 0) return;

#if HAS_AESNI_SUPPORT
        if (use_aesni) {
            aes_ctr_process_ni(data, result, len, key, iv);
            return;
        }
#endif
        // Software fallback
        if (!keyExpanded || memcmp(key, lastKey, 16) != 0) {
            KeyExpansion_sw(key, rk);
            memcpy(lastKey, key, 16);
            keyExpanded = true;
        }

        uint8_t counterBlock[16];
        memcpy(counterBlock, iv, 16);
        uint8_t keyStream[16];

        size_t fullBlocks = len / 16;
        size_t remaining = len % 16;
        const uint8_t* dptr = data;
        uint8_t* rptr = result;

        for (size_t b = 0; b < fullBlocks; b++) {
            aes_encrypt_block_sw(counterBlock, keyStream, rk);
            for (int i = 15; i >= 0; --i) { if (++counterBlock[i] != 0) break; }
            uint64_t* kp = (uint64_t*)keyStream;
            uint64_t* dp = (uint64_t*)dptr;
            uint64_t* op = (uint64_t*)rptr;
            op[0] = dp[0] ^ kp[0];
            op[1] = dp[1] ^ kp[1];
            dptr += 16;
            rptr += 16;
        }

        if (remaining > 0) {
            aes_encrypt_block_sw(counterBlock, keyStream, rk);
            for (size_t j = 0; j < remaining; j++) {
                rptr[j] = dptr[j] ^ keyStream[j];
            }
        }
    }
};

// ============ Fast hex I/O ============
static const char hexchars[] = "0123456789abcdef";

// Lookup table for hex char -> nibble
static uint8_t hex_lut[256];
static bool hex_lut_init = false;

static void init_hex_lut() {
    if (hex_lut_init) return;
    memset(hex_lut, 0, 256);
    for (int i = 0; i < 10; i++) hex_lut['0' + i] = i;
    for (int i = 0; i < 6; i++) { hex_lut['a' + i] = 10 + i; hex_lut['A' + i] = 10 + i; }
    hex_lut_init = true;
}

static inline void hexToBytes_fast(const char* hex, size_t hexlen, uint8_t* out) {
    size_t nbytes = hexlen / 2;
    for (size_t i = 0; i < nbytes; i++) {
        out[i] = (hex_lut[(unsigned char)hex[2*i]] << 4) | hex_lut[(unsigned char)hex[2*i+1]];
    }
}

static inline void bytesToHex_fast(const uint8_t* bytes, size_t nbytes, char* out) {
    for (size_t i = 0; i < nbytes; i++) {
        out[2*i] = hexchars[bytes[i] >> 4];
        out[2*i+1] = hexchars[bytes[i] & 0x0f];
    }
}

int main() {
    init_tables();
    init_hex_lut();

    // Read entire input file at once
    FILE* fin = fopen("test_in.txt", "rb");
    if (!fin) { fprintf(stderr, "Cannot open input file\n"); return 1; }
    fseek(fin, 0, SEEK_END);
    long fsize = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    char* filedata = (char*)malloc(fsize + 1);
    fread(filedata, 1, fsize, fin);
    filedata[fsize] = '\0';
    fclose(fin);

    // Parse lines
    // Each test case: 3 lines (key, iv, plaintext)
    // Pre-allocate output buffer
    std::string allOutput;
    allOutput.reserve(fsize * 2); // generous estimate

    AES128_CTR aes_ctr;

    // Reusable buffers
    static uint8_t key_buf[16];
    static uint8_t iv_buf[16];
    static std::vector<uint8_t> plain_buf;
    static std::vector<uint8_t> cipher_buf;
    plain_buf.reserve(1 << 20);
    cipher_buf.reserve(1 << 20);

    char* p = filedata;
    char* end = filedata + fsize;

    while (p < end) {
        // Read key line
        char* line1_start = p;
        while (p < end && *p != '\n' && *p != '\r') p++;
        size_t line1_len = p - line1_start;
        while (p < end && (*p == '\n' || *p == '\r')) p++;
        if (line1_len < 32) break;

        // Read IV line
        char* line2_start = p;
        while (p < end && *p != '\n' && *p != '\r') p++;
        size_t line2_len = p - line2_start;
        while (p < end && (*p == '\n' || *p == '\r')) p++;
        if (line2_len < 32) break;

        // Read plaintext line
        char* line3_start = p;
        while (p < end && *p != '\n' && *p != '\r') p++;
        size_t line3_len = p - line3_start;
        while (p < end && (*p == '\n' || *p == '\r')) p++;

        // Decode key and IV
        hexToBytes_fast(line1_start, line1_len, key_buf);
        hexToBytes_fast(line2_start, line2_len, iv_buf);

        // Decode plaintext
        size_t plain_len = line3_len / 2;
        if (plain_buf.size() < plain_len) plain_buf.resize(plain_len);
        if (cipher_buf.size() < plain_len) cipher_buf.resize(plain_len);
        hexToBytes_fast(line3_start, line3_len, plain_buf.data());

        // Process
        aes_ctr.process(plain_buf.data(), cipher_buf.data(), plain_len, key_buf, iv_buf);

        // Encode output
        size_t oldSize = allOutput.size();
        allOutput.resize(oldSize + plain_len * 2 + 1);
        bytesToHex_fast(cipher_buf.data(), plain_len, &allOutput[oldSize]);
        allOutput[oldSize + plain_len * 2] = '\n';
    }

    free(filedata);

    // Write output
    FILE* fout = fopen("test_out_custom.txt", "wb");
    if (!fout) { fprintf(stderr, "Cannot open output file\n"); return 1; }
    fwrite(allOutput.data(), 1, allOutput.size(), fout);
    fclose(fout);

    return 0;
}
// EVOLVE-BLOCK-END