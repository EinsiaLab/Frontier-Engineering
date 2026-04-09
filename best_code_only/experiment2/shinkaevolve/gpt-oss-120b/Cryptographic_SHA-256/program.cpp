// EVOLVE-BLOCK-START
#include <iostream>
#include <string>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <immintrin.h>

class SHA256 {
public:
    SHA256() { reset(); }

    void update(const std::string &data) {
        update(reinterpret_cast<const uint8_t*>(data.c_str()), data.size());
    }

    void update(const uint8_t *data, size_t length) {
        if (m_datalen > 0) {
            size_t needed = 64 - m_datalen;
            if (length < needed) {
                memcpy(m_data + m_datalen, data, length);
                m_datalen += length;
                return;
            }
            memcpy(m_data + m_datalen, data, needed);
            transform_avx2(m_data);
            m_bitlen += 512;
            data += needed;
            length -= needed;
            m_datalen = 0;
        }

        while (length >= 64) {
            transform_avx2(data);
            m_bitlen += 512;
            data += 64;
            length -= 64;
        }

        if (length > 0) {
            memcpy(m_data, data, length);
            m_datalen = length;
        }
    }

    std::string final_hash() {
        uint8_t hash[32];
        uint32_t i = m_datalen;

        if (m_datalen < 56) {
            m_data[i] = 0x80;
            memset(m_data + i + 1, 0, 55 - i);
        } else {
            m_data[i] = 0x80;
            memset(m_data + i + 1, 0, 63 - i);
            transform_avx2(m_data);
            memset(m_data, 0, 56);
        }

        m_bitlen += m_datalen * 8;
        m_data[63] = m_bitlen;
        m_data[62] = m_bitlen >> 8;
        m_data[61] = m_bitlen >> 16;
        m_data[60] = m_bitlen >> 24;
        m_data[59] = m_bitlen >> 32;
        m_data[58] = m_bitlen >> 40;
        m_data[57] = m_bitlen >> 48;
        m_data[56] = m_bitlen >> 56;
        transform_avx2(m_data);

        for (i = 0; i < 4; ++i) {
            hash[i]      = (m_state[0] >> (24 - i * 8)) & 0xff;
            hash[i + 4]  = (m_state[1] >> (24 - i * 8)) & 0xff;
            hash[i + 8]  = (m_state[2] >> (24 - i * 8)) & 0xff;
            hash[i + 12] = (m_state[3] >> (24 - i * 8)) & 0xff;
            hash[i + 16] = (m_state[4] >> (24 - i * 8)) & 0xff;
            hash[i + 20] = (m_state[5] >> (24 - i * 8)) & 0xff;
            hash[i + 24] = (m_state[6] >> (24 - i * 8)) & 0xff;
            hash[i + 28] = (m_state[7] >> (24 - i * 8)) & 0xff;
        }

        static const char hex_digits[] = "0123456789abcdef";
        std::string result;
        result.reserve(64);
        for (int j = 0; j < 32; ++j) {
            result.push_back(hex_digits[hash[j] >> 4]);
            result.push_back(hex_digits[hash[j] & 0x0f]);
        }

        reset();
        return result;
    }

private:
    alignas(64) uint8_t m_data[64];
    uint32_t m_datalen;
    uint64_t m_bitlen;
    uint32_t m_state[8];

    static const uint32_t K[64] __attribute__((aligned(32)));

    static inline uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }
    static inline uint32_t choose(uint32_t e, uint32_t f, uint32_t g) {
        return g ^ (e & (f ^ g));
    }
    static inline uint32_t majority(uint32_t a, uint32_t b, uint32_t c) {
        return (a & b) | ((a ^ b) & c);
    }
    static inline uint32_t sig0(uint32_t x) {
        return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
    }
    static inline uint32_t sig1(uint32_t x) {
        return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
    }
    static inline uint32_t ep0(uint32_t x) {
        return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
    }
    static inline uint32_t ep1(uint32_t x) {
        return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
    }

    void reset() {
        m_datalen = 0;
        m_bitlen = 0;
        m_state[0] = 0x6a09e667;
        m_state[1] = 0xbb67ae85;
        m_state[2] = 0x3c6ef372;
        m_state[3] = 0xa54ff53a;
        m_state[4] = 0x510e527f;
        m_state[5] = 0x9b05688c;
        m_state[6] = 0x1f83d9ab;
        m_state[7] = 0x5be0cd19;
    }

    // AVX2-optimized message schedule expansion
    __attribute__((always_inline, target("avx2")))
    static inline void expand_msg_avx2(uint32_t* W, const uint8_t* data) {
        // Load 16 words using AVX2 for better throughput
        __m256i w0 = _mm256_set_epi32(
            (data[60] << 24) | (data[61] << 16) | (data[62] << 8) | data[63],
            (data[56] << 24) | (data[57] << 16) | (data[58] << 8) | data[59],
            (data[52] << 24) | (data[53] << 16) | (data[54] << 8) | data[55],
            (data[48] << 24) | (data[49] << 16) | (data[50] << 8) | data[51],
            (data[44] << 24) | (data[45] << 16) | (data[46] << 8) | data[47],
            (data[40] << 24) | (data[41] << 16) | (data[42] << 8) | data[43],
            (data[36] << 24) | (data[37] << 16) | (data[38] << 8) | data[39],
            (data[32] << 24) | (data[33] << 16) | (data[34] << 8) | data[35]
        );
        __m256i w1 = _mm256_set_epi32(
            (data[28] << 24) | (data[29] << 16) | (data[30] << 8) | data[31],
            (data[24] << 24) | (data[25] << 16) | (data[26] << 8) | data[27],
            (data[20] << 24) | (data[21] << 16) | (data[22] << 8) | data[23],
            (data[16] << 24) | (data[17] << 16) | (data[18] << 8) | data[19],
            (data[12] << 24) | (data[13] << 16) | (data[14] << 8) | data[15],
            (data[8] << 24) | (data[9] << 16) | (data[10] << 8) | data[11],
            (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7],
            (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3]
        );
        
        // Store to W array (reversed order due to set_epi32)
        _mm256_storeu_si256((__m256i*)&W[0], _mm256_permute4x64_epi64(w1, 0xD8));
        _mm256_storeu_si256((__m256i*)&W[8], _mm256_permute4x64_epi64(w0, 0xD8));
    }

    __attribute__((always_inline))
    inline void transform_avx2(const uint8_t *data) {
        alignas(64) uint32_t W[64];
        
        // Load first 16 words
        W[0] = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
        W[1] = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7];
        W[2] = (data[8] << 24) | (data[9] << 16) | (data[10] << 8) | data[11];
        W[3] = (data[12] << 24) | (data[13] << 16) | (data[14] << 8) | data[15];
        W[4] = (data[16] << 24) | (data[17] << 16) | (data[18] << 8) | data[19];
        W[5] = (data[20] << 24) | (data[21] << 16) | (data[22] << 8) | data[23];
        W[6] = (data[24] << 24) | (data[25] << 16) | (data[26] << 8) | data[27];
        W[7] = (data[28] << 24) | (data[29] << 16) | (data[30] << 8) | data[31];
        W[8] = (data[32] << 24) | (data[33] << 16) | (data[34] << 8) | data[35];
        W[9] = (data[36] << 24) | (data[37] << 16) | (data[38] << 8) | data[39];
        W[10] = (data[40] << 24) | (data[41] << 16) | (data[42] << 8) | data[43];
        W[11] = (data[44] << 24) | (data[45] << 16) | (data[46] << 8) | data[47];
        W[12] = (data[48] << 24) | (data[49] << 16) | (data[50] << 8) | data[51];
        W[13] = (data[52] << 24) | (data[53] << 16) | (data[54] << 8) | data[55];
        W[14] = (data[56] << 24) | (data[57] << 16) | (data[58] << 8) | data[59];
        W[15] = (data[60] << 24) | (data[61] << 16) | (data[62] << 8) | data[63];

        // Expand message schedule with 4-way parallelism
        for (int i = 16; i < 64; i += 4) {
            W[i]   = sig1(W[i-2]) + W[i-7] + sig0(W[i-15]) + W[i-16];
            W[i+1] = sig1(W[i-1]) + W[i-6] + sig0(W[i-14]) + W[i-15];
            W[i+2] = sig1(W[i])   + W[i-5] + sig0(W[i-13]) + W[i-14];
            W[i+3] = sig1(W[i+1]) + W[i-4] + sig0(W[i-12]) + W[i-13];
        }

        uint32_t a = m_state[0], b = m_state[1], c = m_state[2], d = m_state[3];
        uint32_t e = m_state[4], f = m_state[5], g = m_state[6], h = m_state[7];

        // Fully unrolled rounds with interleaved computation
        #define R(i) do { \
            uint32_t t1 = h + ep1(e) + choose(e, f, g) + K[i] + W[i]; \
            uint32_t t2 = ep0(a) + majority(a, b, c); \
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2; \
        } while(0)

        R(0); R(1); R(2); R(3); R(4); R(5); R(6); R(7);
        R(8); R(9); R(10); R(11); R(12); R(13); R(14); R(15);
        R(16); R(17); R(18); R(19); R(20); R(21); R(22); R(23);
        R(24); R(25); R(26); R(27); R(28); R(29); R(30); R(31);
        R(32); R(33); R(34); R(35); R(36); R(37); R(38); R(39);
        R(40); R(41); R(42); R(43); R(44); R(45); R(46); R(47);
        R(48); R(49); R(50); R(51); R(52); R(53); R(54); R(55);
        R(56); R(57); R(58); R(59); R(60); R(61); R(62); R(63);

        #undef R

        m_state[0] += a; m_state[1] += b; m_state[2] += c; m_state[3] += d;
        m_state[4] += e; m_state[5] += f; m_state[6] += g; m_state[7] += h;
    }
};

const uint32_t SHA256::K[64] __attribute__((aligned(32))) = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    // Large static buffer for direct I/O
    static char buffer[1 << 24];  // 16MB buffer
    std::cin.read(buffer, sizeof(buffer));
    std::streamsize len = std::cin.gcount();

    SHA256 sha;
    sha.update(reinterpret_cast<const uint8_t*>(buffer), len);
    std::cout << sha.final_hash();
    return 0;
}
// EVOLVE-BLOCK-END