// EVOLVE-BLOCK-START
#include <iostream>
#include <string>
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
        // Handle remaining buffer space
        if (m_datalen > 0) {
            size_t needed = 64 - m_datalen;
            if (length < needed) {
                memcpy(m_data + m_datalen, data, length);
                m_datalen += length;
                return;
            }
            memcpy(m_data + m_datalen, data, needed);
            transform();
            m_bitlen += 512;
            data += needed;
            length -= needed;
        }
        
        // Process complete 64-byte blocks directly
        while (length >= 64) {
            memcpy(m_data, data, 64);
            transform();
            m_bitlen += 512;
            data += 64;
            length -= 64;
        }
        
        // Buffer remaining bytes
        if (length > 0) {
            memcpy(m_data, data, length);
            m_datalen = length;
        }
    }

    
    std::string final_hash() {
        uint8_t hash[32];
        uint32_t i = m_datalen;

        uint64_t totalBits = m_bitlen + m_datalen * 8;
        if (m_datalen < 56) {
            m_data[i++] = 0x80;
            memset(m_data + i, 0, 56 - i);
        } else {
            m_data[i++] = 0x80;
            memset(m_data + i, 0, 64 - i);
            transform();
            memset(m_data, 0, 56);
        }
        for (int j = 0; j < 8; j++)
            m_data[56 + j] = totalBits >> (56 - j * 8);

        transform();

        
        
        for (i = 0; i < 4; ++i) {
            hash[i]      = (m_state[0] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 4]  = (m_state[1] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 8]  = (m_state[2] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 12] = (m_state[3] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 16] = (m_state[4] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 20] = (m_state[5] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 24] = (m_state[6] >> (24 - i * 8)) & 0x000000ff;
            hash[i + 28] = (m_state[7] >> (24 - i * 8)) & 0x000000ff;
        }

        // Fast hex conversion using lookup table
        static const char hex[] = "0123456789abcdef";
        char result[65];
        for (int i = 0; i < 32; ++i) {
            result[i * 2]     = hex[hash[i] >> 4];
            result[i * 2 + 1] = hex[hash[i] & 0xf];
        }
        result[64] = '\0';
        
        reset();
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
        uint32_t a, b, c, d, e, f, g, h, t1, t2, w[64];

        // Unrolled message schedule loading for first 16 words
        w[0] = (m_data[0] << 24) | (m_data[1] << 16) | (m_data[2] << 8) | m_data[3];
        w[1] = (m_data[4] << 24) | (m_data[5] << 16) | (m_data[6] << 8) | m_data[7];
        w[2] = (m_data[8] << 24) | (m_data[9] << 16) | (m_data[10] << 8) | m_data[11];
        w[3] = (m_data[12] << 24) | (m_data[13] << 16) | (m_data[14] << 8) | m_data[15];
        w[4] = (m_data[16] << 24) | (m_data[17] << 16) | (m_data[18] << 8) | m_data[19];
        w[5] = (m_data[20] << 24) | (m_data[21] << 16) | (m_data[22] << 8) | m_data[23];
        w[6] = (m_data[24] << 24) | (m_data[25] << 16) | (m_data[26] << 8) | m_data[27];
        w[7] = (m_data[28] << 24) | (m_data[29] << 16) | (m_data[30] << 8) | m_data[31];
        w[8] = (m_data[32] << 24) | (m_data[33] << 16) | (m_data[34] << 8) | m_data[35];
        w[9] = (m_data[36] << 24) | (m_data[37] << 16) | (m_data[38] << 8) | m_data[39];
        w[10] = (m_data[40] << 24) | (m_data[41] << 16) | (m_data[42] << 8) | m_data[43];
        w[11] = (m_data[44] << 24) | (m_data[45] << 16) | (m_data[46] << 8) | m_data[47];
        w[12] = (m_data[48] << 24) | (m_data[49] << 16) | (m_data[50] << 8) | m_data[51];
        w[13] = (m_data[52] << 24) | (m_data[53] << 16) | (m_data[54] << 8) | m_data[55];
        w[14] = (m_data[56] << 24) | (m_data[57] << 16) | (m_data[58] << 8) | m_data[59];
        w[15] = (m_data[60] << 24) | (m_data[61] << 16) | (m_data[62] << 8) | m_data[63];

        // Message schedule expansion
        for (uint32_t i = 16; i < 64; ++i)
            w[i] = w[i-16] + w[i-7] + (rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15]>>3))
                           + (rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2]>>10));

        a = m_state[0]; b = m_state[1]; c = m_state[2]; d = m_state[3];
        e = m_state[4]; f = m_state[5]; g = m_state[6]; h = m_state[7];

        // 4x unrolled compression rounds for better instruction-level parallelism
        for (uint32_t i = 0; i < 64; i += 4) {
            t1 = h + (rotr(e,6) ^ rotr(e,11) ^ rotr(e,25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            t2 = (rotr(a,2) ^ rotr(a,13) ^ rotr(a,22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
            t1 = h + (rotr(e,6) ^ rotr(e,11) ^ rotr(e,25)) + ((e & f) ^ (~e & g)) + K[i+1] + w[i+1];
            t2 = (rotr(a,2) ^ rotr(a,13) ^ rotr(a,22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
            t1 = h + (rotr(e,6) ^ rotr(e,11) ^ rotr(e,25)) + ((e & f) ^ (~e & g)) + K[i+2] + w[i+2];
            t2 = (rotr(a,2) ^ rotr(a,13) ^ rotr(a,22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
            t1 = h + (rotr(e,6) ^ rotr(e,11) ^ rotr(e,25)) + ((e & f) ^ (~e & g)) + K[i+3] + w[i+3];
            t2 = (rotr(a,2) ^ rotr(a,13) ^ rotr(a,22)) + ((a & b) ^ (a & c) ^ (b & c));
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
    // Fast I/O for better small message throughput
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    SHA256 sha;
    
    // Simple input reading - efficient for all message sizes
    std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    sha.update(input);
    
    std::cout << sha.final_hash();
    return 0;
}
// EVOLVE-BLOCK-END
