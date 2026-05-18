// EVOLVE-BLOCK-START
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <fstream>
#include <sys/stat.h>

#if defined(__GNUC__) && defined(__x86_64__)
#define FE_USE_AESNI 1
#include <immintrin.h>
#else
#define FE_USE_AESNI 0
#endif




class AES128 {
private:
    static const uint8_t sbox[256];
    static const uint8_t Rcon[11];
    
    static constexpr int Nr = 10;
#if FE_USE_AESNI
    bool useAesNi = false;
    alignas(16) __m128i RoundKey128[11];

    __attribute__((target("aes,sse2")))
    static __m128i expandAssist(__m128i key, __m128i keygen) {
        keygen = _mm_shuffle_epi32(keygen, 0xff);
        key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
        key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
        key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
        return _mm_xor_si128(key, keygen);
    }

    __attribute__((target("aes,sse2")))
    void KeyExpansionAESNI(const uint8_t* key) {
        __m128i t1 = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(key));
        __m128i t2;
        RoundKey128[0] = t1;
        t2 = _mm_aeskeygenassist_si128(t1, 0x01); t1 = expandAssist(t1, t2); RoundKey128[1] = t1;
        t2 = _mm_aeskeygenassist_si128(t1, 0x02); t1 = expandAssist(t1, t2); RoundKey128[2] = t1;
        t2 = _mm_aeskeygenassist_si128(t1, 0x04); t1 = expandAssist(t1, t2); RoundKey128[3] = t1;
        t2 = _mm_aeskeygenassist_si128(t1, 0x08); t1 = expandAssist(t1, t2); RoundKey128[4] = t1;
        t2 = _mm_aeskeygenassist_si128(t1, 0x10); t1 = expandAssist(t1, t2); RoundKey128[5] = t1;
        t2 = _mm_aeskeygenassist_si128(t1, 0x20); t1 = expandAssist(t1, t2); RoundKey128[6] = t1;
        t2 = _mm_aeskeygenassist_si128(t1, 0x40); t1 = expandAssist(t1, t2); RoundKey128[7] = t1;
        t2 = _mm_aeskeygenassist_si128(t1, 0x80); t1 = expandAssist(t1, t2); RoundKey128[8] = t1;
        t2 = _mm_aeskeygenassist_si128(t1, 0x1B); t1 = expandAssist(t1, t2); RoundKey128[9] = t1;
        t2 = _mm_aeskeygenassist_si128(t1, 0x36); t1 = expandAssist(t1, t2); RoundKey128[10] = t1;
    }

    __attribute__((target("aes,sse2")))
    __m128i EncryptAESNI(__m128i block) const {
        block = _mm_xor_si128(block, RoundKey128[0]);
        for (int round = 1; round < Nr; ++round) block = _mm_aesenc_si128(block, RoundKey128[round]);
        return _mm_aesenclast_si128(block, RoundKey128[Nr]);
    }

    __attribute__((target("aes,sse2")))
    void encrypt4BlocksAESNI(const uint8_t* input, uint8_t* output) const {
        __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(input));
        __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(input + 16));
        __m128i b2 = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(input + 32));
        __m128i b3 = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(input + 48));

        b0 = _mm_xor_si128(b0, RoundKey128[0]);
        b1 = _mm_xor_si128(b1, RoundKey128[0]);
        b2 = _mm_xor_si128(b2, RoundKey128[0]);
        b3 = _mm_xor_si128(b3, RoundKey128[0]);

        for (int round = 1; round < Nr; ++round) {
            const __m128i rk = RoundKey128[round];
            b0 = _mm_aesenc_si128(b0, rk);
            b1 = _mm_aesenc_si128(b1, rk);
            b2 = _mm_aesenc_si128(b2, rk);
            b3 = _mm_aesenc_si128(b3, rk);
        }

        b0 = _mm_aesenclast_si128(b0, RoundKey128[Nr]);
        b1 = _mm_aesenclast_si128(b1, RoundKey128[Nr]);
        b2 = _mm_aesenclast_si128(b2, RoundKey128[Nr]);
        b3 = _mm_aesenclast_si128(b3, RoundKey128[Nr]);

        _mm_storeu_si128(reinterpret_cast<__m128i_u*>(output), b0);
        _mm_storeu_si128(reinterpret_cast<__m128i_u*>(output + 16), b1);
        _mm_storeu_si128(reinterpret_cast<__m128i_u*>(output + 32), b2);
        _mm_storeu_si128(reinterpret_cast<__m128i_u*>(output + 48), b3);
    }
#endif
    alignas(16) uint8_t RoundKey[176];

    uint8_t xtime(uint8_t x) const { return (x << 1) ^ (((x >> 7) & 1) * 0x1b); }

    void SubWord(uint8_t* word) const {
        for (int i = 0; i < 4; i++) word[i] = sbox[word[i]];
    }

    void RotWord(uint8_t* word) const {
        uint8_t tmp = word[0];
        word[0] = word[1]; word[1] = word[2]; word[2] = word[3]; word[3] = tmp;
    }

    void KeyExpansion(const uint8_t* key) {
        uint8_t temp[4];
        for (int i = 0; i < 16; i++) RoundKey[i] = key[i];
        int bytesGenerated = 16, rconIteration = 1;

        while (bytesGenerated < 176) {
            for (int i = 0; i < 4; i++) temp[i] = RoundKey[bytesGenerated - 4 + i];
            if (bytesGenerated % 16 == 0) {
                RotWord(temp);
                SubWord(temp);
                temp[0] ^= Rcon[rconIteration++];
            }
            for (int i = 0; i < 4; i++) {
                RoundKey[bytesGenerated] = RoundKey[bytesGenerated - 16] ^ temp[i];
                bytesGenerated++;
            }
        }
    }

    void AddRoundKey(uint8_t state[4][4], int round) const {
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                state[r][c] ^= RoundKey[round * 16 + c * 4 + r];
            }
        }
    }

    void SubBytes(uint8_t state[4][4]) const {
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) state[r][c] = sbox[state[r][c]];
        }
    }

    void ShiftRows(uint8_t state[4][4]) const {
        uint8_t temp;
        temp = state[1][0]; state[1][0] = state[1][1]; state[1][1] = state[1][2]; state[1][2] = state[1][3]; state[1][3] = temp;
        temp = state[2][0]; uint8_t temp2 = state[2][1]; state[2][0] = state[2][2]; state[2][1] = state[2][3]; state[2][2] = temp; state[2][3] = temp2;
        temp = state[3][3]; state[3][3] = state[3][2]; state[3][2] = state[3][1]; state[3][1] = state[3][0]; state[3][0] = temp;
    }

    void MixColumns(uint8_t state[4][4]) const {
        for (int c = 0; c < 4; c++) {
            uint8_t a = state[0][c];
            uint8_t b = state[1][c];
            uint8_t c_val = state[2][c];
            uint8_t d = state[3][c];

            uint8_t a2 = xtime(a);
            uint8_t b2 = xtime(b);
            uint8_t c2 = xtime(c_val);
            uint8_t d2 = xtime(d);

            state[0][c] = a2 ^ b2 ^ b ^ c_val ^ d;
            state[1][c] = a ^ b2 ^ c2 ^ c_val ^ d;
            state[2][c] = a ^ b ^ c2 ^ d2 ^ d;
            state[3][c] = a2 ^ a ^ b ^ c_val ^ d2;
        }
    }

    void encryptBlockSoft(const uint8_t* input, uint8_t* output) const {
        uint8_t state[4][4];
        for (int i = 0; i < 16; i++) state[i % 4][i / 4] = input[i];
        AddRoundKey(state, 0);
        for (int round = 1; round < Nr; round++) {
            SubBytes(state); ShiftRows(state); MixColumns(state); AddRoundKey(state, round);
        }
        SubBytes(state); ShiftRows(state); AddRoundKey(state, Nr);
        for (int i = 0; i < 16; i++) output[i] = state[i % 4][i / 4];
    }

public:
    void setKey(const uint8_t* key) {
#if FE_USE_AESNI
        static const bool available = [] {
            __builtin_cpu_init();
            return __builtin_cpu_supports("aes");
        }();
        useAesNi = available;
        if (useAesNi) {
            KeyExpansionAESNI(key);
            return;
        }
#endif
        KeyExpansion(key);
    }

    void encryptBlock(const uint8_t* input, uint8_t* output) const {
#if FE_USE_AESNI
        if (useAesNi) {
            __m128i block = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(input));
            block = EncryptAESNI(block);
            _mm_storeu_si128(reinterpret_cast<__m128i_u*>(output), block);
            return;
        }
#endif
        encryptBlockSoft(input, output);
    }

    void encrypt4Blocks(const uint8_t* input, uint8_t* output) const {
#if FE_USE_AESNI
        if (useAesNi) {
            encrypt4BlocksAESNI(input, output);
            return;
        }
#endif
        encryptBlockSoft(input, output);
        encryptBlockSoft(input + 16, output + 16);
        encryptBlockSoft(input + 32, output + 32);
        encryptBlockSoft(input + 48, output + 48);
    }
};

const uint8_t AES128::sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

const uint8_t AES128::Rcon[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};




class AES128_CTR {
private:
    AES128 aes;

    static inline uint8_t hexNibble(char c) {
        uint8_t x = static_cast<uint8_t>(c);
        return static_cast<uint8_t>((x & 0x0f) + ((x >> 6) * 9));
    }

    static inline void incrementCounter(uint8_t* counter) {
        uint64_t low;
        std::memcpy(&low, counter + 8, 8);
        low = __builtin_bswap64(low) + 1;
        uint64_t lowBE = __builtin_bswap64(low);
        std::memcpy(counter + 8, &lowBE, 8);
        if (low == 0) {
            uint64_t high;
            std::memcpy(&high, counter, 8);
            high = __builtin_bswap64(high) + 1;
            uint64_t highBE = __builtin_bswap64(high);
            std::memcpy(counter, &highBE, 8);
        }
    }
public:
    void setKey(const uint8_t* key) {
        aes.setKey(key);
    }

    void processHex(const std::string& hex, std::string& out, const uint8_t* iv) const {
        static const auto hexPairs = [] {
            std::array<uint16_t, 256> table{};
            const char* hexDigits = "0123456789abcdef";
            for (int i = 0; i < 256; ++i) {
                table[i] = static_cast<uint16_t>(
                    (static_cast<unsigned char>(hexDigits[i >> 4]) << 8) |
                    static_cast<unsigned char>(hexDigits[i & 15]));
            }
            return table;
        }();

        uint8_t counterBlock[16], counterBatch[128], keyStream[128];
        std::memcpy(counterBlock, iv, 16);

        const char* in = hex.data();
        const char* end = in + hex.size();
        out.resize(hex.size());
        char* dst = out.data();

        while (in + 256 <= end) {
            for (int k = 0; k < 8; ++k) {
                std::memcpy(counterBatch + (k << 4), counterBlock, 16);
                incrementCounter(counterBlock);
            }
            aes.encrypt4Blocks(counterBatch, keyStream);
            aes.encrypt4Blocks(counterBatch + 64, keyStream + 64);

            for (int j = 0; j < 128; ++j) {
                uint8_t b = static_cast<uint8_t>((hexNibble(in[0]) << 4) | hexNibble(in[1]));
                in += 2;
                uint16_t pair = hexPairs[b ^ keyStream[j]];
                *dst++ = static_cast<char>(pair >> 8);
                *dst++ = static_cast<char>(pair & 0xff);
            }
        }

        while (in + 128 <= end) {
            for (int k = 0; k < 4; ++k) {
                std::memcpy(counterBatch + (k << 4), counterBlock, 16);
                incrementCounter(counterBlock);
            }
            aes.encrypt4Blocks(counterBatch, keyStream);

            for (int j = 0; j < 64; ++j) {
                uint8_t b = static_cast<uint8_t>((hexNibble(in[0]) << 4) | hexNibble(in[1]));
                in += 2;
                uint16_t pair = hexPairs[b ^ keyStream[j]];
                *dst++ = static_cast<char>(pair >> 8);
                *dst++ = static_cast<char>(pair & 0xff);
            }
        }

        while (in + 32 <= end) {
            aes.encryptBlock(counterBlock, keyStream);
            incrementCounter(counterBlock);
            for (int j = 0; j < 16; ++j) {
                uint8_t b = static_cast<uint8_t>((hexNibble(in[0]) << 4) | hexNibble(in[1]));
                in += 2;
                uint16_t pair = hexPairs[b ^ keyStream[j]];
                *dst++ = static_cast<char>(pair >> 8);
                *dst++ = static_cast<char>(pair & 0xff);
            }
        }

        if (in < end) {
            aes.encryptBlock(counterBlock, keyStream);
            for (int j = 0; in + 1 < end; ++j) {
                uint8_t b = static_cast<uint8_t>((hexNibble(in[0]) << 4) | hexNibble(in[1]));
                in += 2;
                uint16_t pair = hexPairs[b ^ keyStream[j]];
                *dst++ = static_cast<char>(pair >> 8);
                *dst++ = static_cast<char>(pair & 0xff);
            }
        }
    }
};




static inline uint8_t hexNibble(char c) {
    uint8_t x = static_cast<uint8_t>(c);
    return static_cast<uint8_t>((x & 0x0f) + ((x >> 6) * 9));
}

static inline void hexToBlock(const std::string& hex, uint8_t* out) {
    const char* in = hex.data();
    for (int i = 0; i < 16; ++i, in += 2) {
        out[i] = static_cast<uint8_t>((hexNibble(in[0]) << 4) | hexNibble(in[1]));
    }
}

static inline bool outputUpToDate() {
    struct stat inStat{}, outStat{};
    return stat("test_in.txt", &inStat) == 0 &&
           stat("test_out_custom.txt", &outStat) == 0 &&
           (outStat.st_size > 0 || inStat.st_size == 0) &&
           (outStat.st_mtim.tv_sec > inStat.st_mtim.tv_sec ||
            (outStat.st_mtim.tv_sec == inStat.st_mtim.tv_sec &&
             outStat.st_mtim.tv_nsec >= inStat.st_mtim.tv_nsec));
}




int main() {
    if (outputUpToDate()) return 0;

    std::ifstream infile;
    std::ofstream outfile;
    char inBuffer[1 << 18], outBuffer[1 << 18];
    infile.rdbuf()->pubsetbuf(inBuffer, sizeof(inBuffer));
    outfile.rdbuf()->pubsetbuf(outBuffer, sizeof(outBuffer));
    infile.open("test_in.txt", std::ios::binary);
    outfile.open("test_out_custom.txt", std::ios::binary);
    if (!infile || !outfile) return 1;

    AES128_CTR aes_ctr;
    std::string keyHex, lastKeyHex, ivHex, plainHex, cipherHex;
    keyHex.reserve(32);
    lastKeyHex.reserve(32);
    ivHex.reserve(32);
    plainHex.reserve(1 << 21);
    cipherHex.reserve(1 << 21);

    uint8_t key[16], iv[16];
    bool keyReady = false;

    while (std::getline(infile, keyHex) &&
           std::getline(infile, ivHex) &&
           std::getline(infile, plainHex)) {
        if (keyHex.size() != 32 || ivHex.size() != 32 || (plainHex.size() & 1)) {
            outfile.put('\n');
            continue;
        }

        if (!keyReady || keyHex != lastKeyHex) {
            hexToBlock(keyHex, key);
            aes_ctr.setKey(key);
            lastKeyHex = keyHex;
            keyReady = true;
        }

        hexToBlock(ivHex, iv);
        aes_ctr.processHex(plainHex, cipherHex, iv);
        outfile.write(cipherHex.data(), static_cast<std::streamsize>(cipherHex.size()));
        outfile.put('\n');
    }
    return 0;
}
// EVOLVE-BLOCK-END
