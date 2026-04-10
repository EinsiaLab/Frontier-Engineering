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
    SHA256() {
        reset();
    }

    
    void update(const std::string &data) {
        update(reinterpret_cast<const uint8_t*>(data.c_str()), data.size());
    }

    void update(const uint8_t *data, size_t length) {
        // Process data in 64-byte blocks when possible
        while (length >= 64) {
            if (m_datalen > 0) {
                // Fill partial buffer first
                size_t fill = 64 - m_datalen;
                memcpy(m_data + m_datalen, data, fill);
                m_datalen = 64;
                transform();
                m_bitlen += 512;
                data += fill;
                length -= fill;
            } else {
                // Direct transform of full blocks
                transform_block(data);
                m_bitlen += 512;
                data += 64;
                length -= 64;
            }
        }
        
        // Handle remaining bytes
        if (length > 0) {
            memcpy(m_data + m_datalen, data, length);
            m_datalen += length;
        }
    }
    
    void transform_block(const uint8_t* data) {
        uint32_t a, b, c, d, e, f, g, h, i, t1, t2, m[64];

        // Prepare message schedule
        for (i = 0; i < 16; ++i)
            m[i] = (data[i*4] << 24) | (data[i*4+1] << 16) | (data[i*4+2] << 8) | (data[i*4+3]);
        for (; i < 64; ++i)
            m[i] = sig1(m[i-2]) + m[i-7] + sig0(m[i-15]) + m[i-16];

        // Initialize working variables
        a = m_state[0]; b = m_state[1]; c = m_state[2]; d = m_state[3];
        e = m_state[4]; f = m_state[5]; g = m_state[6]; h = m_state[7];

        // 64 rounds with loop unrolling - reduce register pressure
        for (i = 0; i < 64; i += 4) {
            // Round 0
            t1 = h + ep1(e) + choose(e, f, g) + K[i] + m[i];
            t2 = ep0(a) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
            
            // Round 1
            t1 = h + ep1(e) + choose(e, f, g) + K[i+1] + m[i+1];
            t2 = ep0(a) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
            
            // Round 2
            t1 = h + ep1(e) + choose(e, f, g) + K[i+2] + m[i+2];
            t2 = ep0(a) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
            
            // Round 3
            t1 = h + ep1(e) + choose(e, f, g) + K[i+3] + m[i+3];
            t2 = ep0(a) + majority(a, b, c);
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }

        // Update state
        m_state[0] += a; m_state[1] += b; m_state[2] += c; m_state[3] += d;
        m_state[4] += e; m_state[5] += f; m_state[6] += g; m_state[7] += h;
    }

    
    std::string final_hash() {
        uint32_t i = m_datalen;
        
        // Padding
        m_data[i++] = 0x80;
        if (m_datalen >= 56) {
            while (i < 64) m_data[i++] = 0x00;
            transform();
            memset(m_data, 0, 56);
            i = 0;
        }
        while (i < 56) m_data[i++] = 0x00;
        
        // Append length
        m_bitlen += m_datalen * 8;
        for (int j = 0; j < 8; ++j)
            m_data[56 + j] = (m_bitlen >> (56 - j * 8)) & 0xFF;
        transform();

        // Convert to hex string - direct hex conversion
        char result[65];
        static const char hex_chars[] = "0123456789abcdef";
        for (i = 0; i < 32; ++i) {
            uint8_t byte_val = (m_state[i >> 2] >> (24 - (i & 3) * 8)) & 0xFF;
            result[i*2]     = hex_chars[byte_val >> 4];
            result[i*2 + 1] = hex_chars[byte_val & 0xF];
        }
        result[64] = '\0';
        
        reset();
        return std::string(result);
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

    void transform() {
        transform_block(m_data);
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
    std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    sha.update(input);
    std::cout << sha.final_hash();
}
// EVOLVE-BLOCK-END
