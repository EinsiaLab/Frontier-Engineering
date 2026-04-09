// EVOLVE-BLOCK-START
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <cstring>

class SHA256 {
public:
    SHA256() { reset(); }

    void update(const std::string &data) {
        update(reinterpret_cast<const uint8_t*>(data.data()), data.size());
    }

    void update(const uint8_t *data, size_t len) {
        size_t i = 0;

        // If there is leftover data from previous update, fill the buffer first
        if (m_datalen) {
            size_t need = 64 - m_datalen;
            if (len < need) {
                memcpy(m_data + m_datalen, data, len);
                m_datalen += len;
                return;
            }
            memcpy(m_data + m_datalen, data, need);
            transform();
            m_bitlen += 512;
            m_datalen = 0;
            i = need;
        }

        // Process full 64‑byte blocks directly from the input buffer
        for (; i + 63 < len; i += 64) {
            memcpy(m_data, data + i, 64);
            transform();
            m_bitlen += 512;
        }

        // Copy any remaining bytes into the buffer
        size_t remaining = len - i;
        if (remaining) {
            memcpy(m_data, data + i, remaining);
            m_datalen = remaining;
        }
    }

    std::string final_hash() {
        uint8_t hash[32];
        uint32_t i = m_datalen;

        // Pad whatever data is left in the buffer.
        if (m_datalen < 56) {
            m_data[i++] = 0x80;
            while (i < 56) m_data[i++] = 0x00;
        } else {
            m_data[i++] = 0x80;
            while (i < 64) m_data[i++] = 0x00;
            transform();
            memset(m_data, 0, 56);
        }

        // Append the total length in bits as a 64‑bit big‑endian integer.
        m_bitlen += static_cast<uint64_t>(m_datalen) * 8;
        m_data[63] = static_cast<uint8_t>(m_bitlen);
        m_data[62] = static_cast<uint8_t>(m_bitlen >> 8);
        m_data[61] = static_cast<uint8_t>(m_bitlen >> 16);
        m_data[60] = static_cast<uint8_t>(m_bitlen >> 24);
        m_data[59] = static_cast<uint8_t>(m_bitlen >> 32);
        m_data[58] = static_cast<uint8_t>(m_bitlen >> 40);
        m_data[57] = static_cast<uint8_t>(m_bitlen >> 48);
        m_data[56] = static_cast<uint8_t>(m_bitlen >> 56);
        transform();

        // Convert state to the final hash bytes.
        for (i = 0; i < 4; ++i) {
            hash[i]      = (m_state[0] >> (24 - i * 8)) & 0xFF;
            hash[i + 4]  = (m_state[1] >> (24 - i * 8)) & 0xFF;
            hash[i + 8]  = (m_state[2] >> (24 - i * 8)) & 0xFF;
            hash[i + 12] = (m_state[3] >> (24 - i * 8)) & 0xFF;
            hash[i + 16] = (m_state[4] >> (24 - i * 8)) & 0xFF;
            hash[i + 20] = (m_state[5] >> (24 - i * 8)) & 0xFF;
            hash[i + 24] = (m_state[6] >> (24 - i * 8)) & 0xFF;
            hash[i + 28] = (m_state[7] >> (24 - i * 8)) & 0xFF;
        }

        std::stringstream ss;
        for (int j = 0; j < 32; ++j)
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[j]);

        reset(); // ready for next use
        return ss.str();
    }

private:
    uint8_t  m_data[64];
    uint32_t m_datalen;
    uint64_t m_bitlen;
    uint32_t m_state[8];

    static const uint32_t K[64];

    static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
    static inline uint32_t choose(uint32_t e, uint32_t f, uint32_t g) { return (e & f) ^ (~e & g); }
    static inline uint32_t majority(uint32_t a, uint32_t b, uint32_t c) { return (a & b) ^ (a & c) ^ (b & c); }
    static inline uint32_t sig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
    static inline uint32_t sig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }
    static inline uint32_t ep0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
    static inline uint32_t ep1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }

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

    void transform() {
        uint32_t w[64];
        uint32_t a, b, c, d, e, f, g, h;

        // Message schedule
        for (int i = 0, j = 0; i < 16; ++i, j += 4)
            w[i] = (m_data[j] << 24) | (m_data[j + 1] << 16) | (m_data[j + 2] << 8) | (m_data[j + 3]);
        for (int i = 16; i < 64; ++i)
            w[i] = sig1(w[i - 2]) + w[i - 7] + sig0(w[i - 15]) + w[i - 16];

        a = m_state[0];
        b = m_state[1];
        c = m_state[2];
        d = m_state[3];
        e = m_state[4];
        f = m_state[5];
        g = m_state[6];
        h = m_state[7];

        for (int i = 0; i < 64; ++i) {
            uint32_t t1 = h + ep1(e) + choose(e, f, g) + K[i] + w[i];
            uint32_t t2 = ep0(a) + majority(a, b, c);
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
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

int main() {
    SHA256 sha;
    std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    sha.update(reinterpret_cast<const uint8_t*>(input.data()), input.size());
    std::cout << sha.final_hash();
    return 0;
}
// EVOLVE-BLOCK-END
