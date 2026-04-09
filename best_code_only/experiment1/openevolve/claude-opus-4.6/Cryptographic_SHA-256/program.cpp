// EVOLVE-BLOCK-START
#include <cstdint>
#include <cstring>
#include <unistd.h>

class SHA256 {
public:
    SHA256() {
        reset();
    }

    
    void update(const uint8_t *data, size_t length) {
        if (m_datalen > 0) {
            size_t need = 64 - m_datalen;
            if (length < need) {
                memcpy(m_data + m_datalen, data, length);
                m_datalen += length;
                return;
            }
            memcpy(m_data + m_datalen, data, need);
            transform(m_data);
            m_bitlen += 512;
            m_datalen = 0;
            data += need;
            length -= need;
        }
        while (length >= 64) {
            transform(data);
            m_bitlen += 512;
            data += 64;
            length -= 64;
        }
        if (length > 0) {
            memcpy(m_data, data, length);
            m_datalen = length;
        }
    }

    
    void final_hash(char *out) {
        uint32_t i = m_datalen;

        if (m_datalen < 56) {
            m_data[i++] = 0x80;
            memset(m_data + i, 0, 56 - i);
        } else {
            m_data[i++] = 0x80;
            memset(m_data + i, 0, 64 - i);
            transform(m_data);
            memset(m_data, 0, 56);
        }

        m_bitlen += m_datalen * 8;
        m_data[63] = (uint8_t)(m_bitlen);
        m_data[62] = (uint8_t)(m_bitlen >> 8);
        m_data[61] = (uint8_t)(m_bitlen >> 16);
        m_data[60] = (uint8_t)(m_bitlen >> 24);
        m_data[59] = (uint8_t)(m_bitlen >> 32);
        m_data[58] = (uint8_t)(m_bitlen >> 40);
        m_data[57] = (uint8_t)(m_bitlen >> 48);
        m_data[56] = (uint8_t)(m_bitlen >> 56);
        transform(m_data);

        static const char hex_chars[] = "0123456789abcdef";
        for (int s = 0; s < 8; ++s) {
            uint32_t v = m_state[s];
            out[s*8+0] = hex_chars[(v >> 28) & 0xf];
            out[s*8+1] = hex_chars[(v >> 24) & 0xf];
            out[s*8+2] = hex_chars[(v >> 20) & 0xf];
            out[s*8+3] = hex_chars[(v >> 16) & 0xf];
            out[s*8+4] = hex_chars[(v >> 12) & 0xf];
            out[s*8+5] = hex_chars[(v >>  8) & 0xf];
            out[s*8+6] = hex_chars[(v >>  4) & 0xf];
            out[s*8+7] = hex_chars[(v      ) & 0xf];
        }
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
        return (e & f) ^ (~e & g);
    }
    static inline uint32_t majority(uint32_t a, uint32_t b, uint32_t c) {
        return (a & b) ^ (a & c) ^ (b & c);
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

    __attribute__((hot)) void transform(const uint8_t *blk) {
        uint32_t a, b, c, d, e, f, g, h, t1, t2, m[64];

        for (uint32_t i = 0, j = 0; i < 16; ++i, j += 4)
            m[i] = ((uint32_t)blk[j] << 24) | ((uint32_t)blk[j+1] << 16) | ((uint32_t)blk[j+2] << 8) | (uint32_t)blk[j+3];
        for (uint32_t i = 16; i < 64; ++i)
            m[i] = sig1(m[i - 2]) + m[i - 7] + sig0(m[i - 15]) + m[i - 16];

        a = m_state[0];
        b = m_state[1];
        c = m_state[2];
        d = m_state[3];
        e = m_state[4];
        f = m_state[5];
        g = m_state[6];
        h = m_state[7];

        for (uint32_t i = 0; i < 64; ++i) {
            t1 = h + ep1(e) + choose(e, f, g) + K[i] + m[i];
            t2 = ep0(a) + majority(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        m_state[0] += a;
        m_state[1] += b;
        m_state[2] += c;
        m_state[3] += d;
        m_state[4] += e;
        m_state[5] += f;
        m_state[6] += g;
        m_state[7] += h;
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

    static char buf[2 << 20];
    size_t total = 0;
    for (;;) {
        ssize_t n = read(STDIN_FILENO, buf + total, sizeof(buf) - total);
        if (n <= 0) break;
        total += (size_t)n;
    }
    sha.update(reinterpret_cast<const uint8_t*>(buf), total);

    char out[64];
    sha.final_hash(out);
    write(STDOUT_FILENO, out, 64);
    return 0;
}
// EVOLVE-BLOCK-END
