// EVOLVE-BLOCK-START
#include <string>
#include <cstdint>
#include <cstring>
#include <unistd.h>

class SHA256 {
public:
    SHA256() {
        reset();
    }

    
    void update(const std::string &data) {
        update(reinterpret_cast<const uint8_t*>(data.c_str()), data.size());
    }

    void update(const uint8_t *data, size_t length) {
        if (m_datalen) {
            size_t needed = 64 - m_datalen;
            if (length < needed) {
                memcpy(m_data + m_datalen, data, length);
                m_datalen += static_cast<uint32_t>(length);
                return;
            }
            memcpy(m_data + m_datalen, data, needed);
            transform(m_data);
            m_bitlen += 512;
            data += needed;
            length -= needed;
            m_datalen = 0;
        }

        while (length >= 256) {
            transform(data);
            transform(data + 64);
            transform(data + 128);
            transform(data + 192);
            m_bitlen += 2048;
            data += 256;
            length -= 256;
        }
        while (length >= 64) {
            transform(data);
            m_bitlen += 512;
            data += 64;
            length -= 64;
        }

        if (length)
            memcpy(m_data, data, length);
        m_datalen = static_cast<uint32_t>(length);
    }

    
    void final_hash(char result[65]) {
        uint32_t i = m_datalen;
        const uint64_t totalBits = m_bitlen + (static_cast<uint64_t>(m_datalen) << 3);

        m_data[i++] = 0x80;
        if (i > 56) {
            memset(m_data + i, 0, 64 - i);
            transform(m_data);
            memset(m_data, 0, 56);
        } else {
            memset(m_data + i, 0, 56 - i);
        }

        const uint64_t totalBitsBE = __builtin_bswap64(totalBits);
        memcpy(m_data + 56, &totalBitsBE, sizeof(totalBitsBE));
        transform(m_data);

        static const char hex[] = "0123456789abcdef";
        int k = 0;
        for (int j = 0; j < 8; ++j) {
            uint32_t s = m_state[j];
            result[k++] = hex[(s >> 28) & 0xf];
            result[k++] = hex[(s >> 24) & 0xf];
            result[k++] = hex[(s >> 20) & 0xf];
            result[k++] = hex[(s >> 16) & 0xf];
            result[k++] = hex[(s >> 12) & 0xf];
            result[k++] = hex[(s >> 8) & 0xf];
            result[k++] = hex[(s >> 4) & 0xf];
            result[k++] = hex[s & 0xf];
        }
        result[64] = '\0';

        reset();
    }

    std::string final_hash() {
        char result[65];
        final_hash(result);
        return std::string(result, 64);
    }

private:
    uint8_t m_data[64];
    uint32_t m_datalen;
    uint64_t m_bitlen;
    uint32_t m_state[8];

    static const uint32_t K[64];

    
    static inline uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }
    static inline uint32_t choose(uint32_t e, uint32_t f, uint32_t g) {
        return g ^ (e & (f ^ g));
    }
    static inline uint32_t majority(uint32_t a, uint32_t b, uint32_t c) {
        return (a & b) | (c & (a | b));
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
    static inline uint32_t load_be(const uint8_t *p) {
        uint32_t v;
        memcpy(&v, p, sizeof(v));
        return __builtin_bswap32(v);
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

    __attribute__((always_inline)) void transform(const uint8_t *p) {
        uint32_t w[16];
        uint32_t a = m_state[0], b = m_state[1], c = m_state[2], d = m_state[3];
        uint32_t e = m_state[4], f = m_state[5], g = m_state[6], h = m_state[7];
        uint32_t t1, t2;

        for (uint32_t i = 0; i < 16; ++i, p += 4)
            w[i] = load_be(p);

        for (uint32_t i = 0; i < 16; i += 4) {
            t1 = h + ep1(e) + choose(e, f, g) + K[i] + w[i];
            t2 = ep0(a) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;

            t1 = h + ep1(e) + choose(e, f, g) + K[i + 1] + w[i + 1];
            t2 = ep0(a) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;

            t1 = h + ep1(e) + choose(e, f, g) + K[i + 2] + w[i + 2];
            t2 = ep0(a) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;

            t1 = h + ep1(e) + choose(e, f, g) + K[i + 3] + w[i + 3];
            t2 = ep0(a) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }

        for (uint32_t i = 16; i < 64; i += 4) {
            uint32_t w0 = (w[i & 15] += sig1(w[(i - 2) & 15]) + w[(i - 7) & 15] + sig0(w[(i - 15) & 15]));
            t1 = h + ep1(e) + choose(e, f, g) + K[i] + w0;
            t2 = ep0(a) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;

            uint32_t w1 = (w[(i + 1) & 15] += sig1(w[(i - 1) & 15]) + w[(i - 6) & 15] + sig0(w[(i - 14) & 15]));
            t1 = h + ep1(e) + choose(e, f, g) + K[i + 1] + w1;
            t2 = ep0(a) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;

            uint32_t w2 = (w[(i + 2) & 15] += sig1(w[i & 15]) + w[(i - 5) & 15] + sig0(w[(i - 13) & 15]));
            t1 = h + ep1(e) + choose(e, f, g) + K[i + 2] + w2;
            t2 = ep0(a) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;

            uint32_t w3 = (w[(i + 3) & 15] += sig1(w[(i + 1) & 15]) + w[(i - 4) & 15] + sig0(w[(i - 12) & 15]));
            t1 = h + ep1(e) + choose(e, f, g) + K[i + 3] + w3;
            t2 = ep0(a) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }

        m_state[0] += a; m_state[1] += b; m_state[2] += c; m_state[3] += d;
        m_state[4] += e; m_state[5] += f; m_state[6] += g; m_state[7] += h;
    }
};


const uint32_t SHA256::K[64] = {
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
    SHA256 sha;
    alignas(64) uint8_t buf[1 << 16];
    for (;;) {
        ssize_t n = ::read(0, buf, sizeof(buf));
        if (n > 0) {
            sha.update(buf, static_cast<size_t>(n));
        } else if (n == 0) {
            break;
        } else {
            return 1;
        }
    }

    char out[65];
    sha.final_hash(out);
    if (::write(1, out, 64) != 64) return 1;
    _exit(0);
}
// EVOLVE-BLOCK-END
