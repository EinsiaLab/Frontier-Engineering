// EVOLVE-BLOCK-START
#include <unistd.h>   // read / write
#include <cstdint>
#include <cstring>

class SHA256 {
public:
    SHA256() {
        reset();
    }

    
    // std::string overload removed – not needed for benchmark

    // Fast bulk update: handle any pending partial block,
    // then process complete 64‑byte blocks directly,
    // and finally store the trailing bytes.
    void update(const uint8_t * __restrict data, size_t length) {
        size_t offset = 0;

        // 1) Finish a partially‑filled block, if any.
        if (m_datalen != 0) {
            size_t need = 64 - m_datalen;
            if (length < need) {
                // Not enough data to complete the block.
                std::memcpy(m_data + m_datalen, data, length);
                m_datalen += static_cast<uint32_t>(length);
                return;
            }
            // Complete the pending block and process it.
            std::memcpy(m_data + m_datalen, data, need);
            transform();
            m_bitlen += 512;
            m_datalen = 0;
            offset = need;
            length -= need;
        }

        // 2) Process whole 64‑byte blocks directly from the input.
        while (length >= 64) {
            // Transform the block directly without an extra copy.
            transform_block(data + offset);
            m_bitlen += 512;
            offset += 64;
            length -= 64;
        }

        // 3) Store any remaining tail bytes for the final padding step.
        if (length > 0) {
            std::memcpy(m_data, data + offset, length);
            m_datalen = static_cast<uint32_t>(length);
        }
    }

    
    // Writes the 64‑character hex digest (plus NUL) into caller‑provided buffer.
    // Avoids heap allocation used by std::string version.
    void final_hash(char out[65]) {
        uint8_t hash[32];
        uint32_t i = m_datalen;

        // ---- Padding ----------------------------------------------------
        if (m_datalen < 56) {
            // Append the 0x80 byte and zero‑fill up to byte 56.
            m_data[i++] = 0x80;
            size_t pad_len = 56 - i;
            if (pad_len) {
                memset(m_data + i, 0, pad_len);
                i = 56;
            }
        } else {
            // Append the 0x80 byte, zero‑fill to the end of the block,
            // compress, then start a fresh block for the length field.
            m_data[i++] = 0x80;
            size_t pad_len = 64 - i;
            if (pad_len) {
                memset(m_data + i, 0, pad_len);
                i = 64;
            }
            transform();
            // New block: clear first 56 bytes where the length will be stored.
            memset(m_data, 0, 56);
        }

        // ---- Length in bits ---------------------------------------------
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

        // ---- Convert state to raw hash bytes ----------------------------
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

        // ---- Fast hex conversion ----------------------------------------
        static const char* hex = "0123456789abcdef";
        for (int j = 0; j < 32; ++j) {
            out[j * 2]     = hex[hash[j] >> 4];
            out[j * 2 + 1] = hex[hash[j] & 0xF];
        }
        out[64] = '\0';
        reset();
    }

private:
    alignas(64) uint8_t m_data[64];
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

    inline void transform() {
        uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

        for (i = 0, j = 0; i < 16; ++i, j += 4)
            m[i] = (m_data[j] << 24) | (m_data[j + 1] << 16) | (m_data[j + 2] << 8) | (m_data[j + 3]);
        for ( ; i < 64; ++i)
            m[i] = sig1(m[i - 2]) + m[i - 7] + sig0(m[i - 15]) + m[i - 16];

        a = m_state[0];
        b = m_state[1];
        c = m_state[2];
        d = m_state[3];
        e = m_state[4];
        f = m_state[5];
        g = m_state[6];
        h = m_state[7];

        for (i = 0; i < 64; ++i) {
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

    // Process a full 64‑byte block directly from the supplied pointer.
    // This avoids the memcpy into m_data that the original `transform()` needed.
    inline void transform_block(const uint8_t * __restrict block) {
        uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

        for (i = 0, j = 0; i < 16; ++i, j += 4)
            m[i] = (block[j] << 24) | (block[j + 1] << 16) |
                   (block[j + 2] << 8) | (block[j + 3]);
        for ( ; i < 64; ++i)
            m[i] = sig1(m[i - 2]) + m[i - 7] + sig0(m[i - 15]) + m[i - 16];

        a = m_state[0];
        b = m_state[1];
        c = m_state[2];
        d = m_state[3];
        e = m_state[4];
        f = m_state[5];
        g = m_state[6];
        h = m_state[7];

        for (i = 0; i < 64; ++i) {
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
    // Use low‑level POSIX I/O for maximum throughput.
    // Use a larger, cache‑line‑aligned buffer to reduce syscall overhead.
    // 64 KiB is a good trade‑off between cache usage and number of read() calls.
    // 32 KiB buffer – a better trade‑off between cache usage and syscall overhead
    constexpr std::size_t BUFFER_SIZE = 1 << 15; // 32 KiB buffer
    alignas(64) static uint8_t buffer[BUFFER_SIZE];

    SHA256 sha;

    ssize_t bytesRead;
    while ((bytesRead = read(STDIN_FILENO, buffer, BUFFER_SIZE)) > 0) {
        sha.update(buffer, static_cast<std::size_t>(bytesRead));
    }

    // Output the hash using a low‑overhead write().
    char out[65];
    sha.final_hash(out);
    write(STDOUT_FILENO, out, 64);

    return 0;
}
// EVOLVE-BLOCK-END
