// EVOLVE-BLOCK-START
#include <iostream>
#include <vector>

#include <cstdint>
#include <string>
#include <fstream>

#include <array>          // fixed‑size buffers, avoids allocations
#include <cstring>        // memcpy for fast IV copy





class AES128 {
private:
    static const uint8_t sbox[256];
    static const uint8_t Rcon[11];
    
    static constexpr int Nk = 4, Nr = 10, Nb = 4;
    std::array<uint8_t,176> RoundKey;

    // Cache the most‑recent expanded round keys to avoid recomputation per block.


    constexpr uint8_t xtime(uint8_t x) noexcept { return (x << 1) ^ (((x >> 7) & 1) * 0x1b); }

    inline void SubWord(uint8_t* word) noexcept {
        for (int i = 0; i < 4; i++) word[i] = sbox[word[i]];
    }

    inline void RotWord(uint8_t* word) noexcept {
        uint8_t tmp = word[0];
        word[0] = word[1]; word[1] = word[2]; word[2] = word[3]; word[3] = tmp;
    }

    void KeyExpansion(const std::vector<uint8_t>& key) {
        // RoundKey is a fixed‑size std::array; no need to resize.
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

    void AddRoundKey(uint8_t state[4][4], int round) {
        for (int c = 0; c < 4; c++) {
            for (int r = 0; r < 4; r++) {
                state[r][c] ^= RoundKey[round * 16 + c * 4 + r];
            }
        }
    }

    void SubBytes(uint8_t state[4][4]) {
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) state[r][c] = sbox[state[r][c]];
        }
    }

    void ShiftRows(uint8_t state[4][4]) {
        uint8_t temp;
        temp = state[1][0]; state[1][0] = state[1][1]; state[1][1] = state[1][2]; state[1][2] = state[1][3]; state[1][3] = temp;
        temp = state[2][0]; uint8_t temp2 = state[2][1]; state[2][0] = state[2][2]; state[2][1] = state[2][3]; state[2][2] = temp; state[2][3] = temp2;
        temp = state[3][3]; state[3][3] = state[3][2]; state[3][2] = state[3][1]; state[3][1] = state[3][0]; state[3][0] = temp;
    }

    void MixColumns(uint8_t state[4][4]) {
        uint8_t tmp[4];
        for (int c = 0; c < 4; c++) {
            for (int i = 0; i < 4; i++) tmp[i] = state[i][c];
            state[0][c] = xtime(tmp[0]) ^ (xtime(tmp[1]) ^ tmp[1]) ^ tmp[2] ^ tmp[3];
            state[1][c] = tmp[0] ^ xtime(tmp[1]) ^ (xtime(tmp[2]) ^ tmp[2]) ^ tmp[3];
            state[2][c] = tmp[0] ^ tmp[1] ^ xtime(tmp[2]) ^ (xtime(tmp[3]) ^ tmp[3]);
            state[3][c] = (xtime(tmp[0]) ^ tmp[0]) ^ tmp[1] ^ tmp[2] ^ xtime(tmp[3]);
        }
    }

public:
    // encryptBlock() removed – not required for CTR mode (setKey + encryptBlockRaw are sufficient).
    void setKey(const std::vector<uint8_t>& key) {
        // Expand the key once.
        KeyExpansion(key);
    }



    // Overload that accepts a std::array directly.
    std::array<uint8_t,16> encryptBlockRaw(const std::array<uint8_t,16>& plaintext) {
        uint8_t state[4][4];
        for (int i = 0; i < 16; i++) state[i % 4][i / 4] = plaintext[i];
        AddRoundKey(state, 0);
        for (int round = 1; round < Nr; round++) {
            SubBytes(state); ShiftRows(state); MixColumns(state); AddRoundKey(state, round);
        }
        SubBytes(state); ShiftRows(state); AddRoundKey(state, Nr);
        std::array<uint8_t,16> out;
        for (int i = 0; i < 16; i++) out[i] = state[i % 4][i / 4];
        return out;
    }
};

constexpr uint8_t AES128::sbox[256] = {
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

constexpr uint8_t AES128::Rcon[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};




class AES128_CTR {
private:
    AES128 aes;


    // Overload that works with a std::array.
    void incrementCounter(std::array<uint8_t,16>& counter) {
        for (int i = 15; i >= 0; --i) {
            if (++counter[i] != 0) break;
        }
    }
public:
    std::vector<uint8_t> process(const std::vector<uint8_t>& data,
                                const std::vector<uint8_t>& key,
                                const std::vector<uint8_t>& iv) {
        // Expand the key once for the whole stream.
        aes.setKey(key);

        std::vector<uint8_t> result(data.size());

        std::array<uint8_t,16> counterBlock{};
        // copy IV into the fixed‑size array (std::copy is header‑only)
        std::memcpy(counterBlock.data(), iv.data(), 16);

        std::array<uint8_t,16> keystream;   // stack‑allocated, no reallocations

        // Process full 16‑byte blocks first – this removes the per‑byte conditional
        // and improves cache usage.
        size_t i = 0;
        // Process full 16‑byte blocks – use raw pointers and 64‑bit XOR.
        const uint8_t* srcPtr = data.data();                 // cache input pointer
        uint8_t* outPtr = result.data();                    // raw output pointer
        while (i + 16 <= data.size()) {
            keystream = aes.encryptBlockRaw(counterBlock);
            incrementCounter(counterBlock);
            // XOR 16 bytes as two 64‑bit words (still correct on little‑/big‑endian)
            const uint64_t* ks64 = reinterpret_cast<const uint64_t*>(keystream.data());
            const uint64_t* in64 = reinterpret_cast<const uint64_t*>(srcPtr + i);
            uint64_t* out64 = reinterpret_cast<uint64_t*>(outPtr + i);
            out64[0] = in64[0] ^ ks64[0];
            out64[1] = in64[1] ^ ks64[1];
            i += 16;
        }

        // Handle any remaining bytes – simple byte loop.
        size_t remaining = data.size() - i;
        if (remaining) {
            keystream = aes.encryptBlockRaw(counterBlock);
            incrementCounter(counterBlock);
            // srcPtr already defined above
            for (size_t j = 0; j < remaining; ++j) {
                outPtr[i + j] = srcPtr[i + j] ^ keystream[j];
            }
        }
        return result;
    }
};




static constexpr inline uint8_t hexCharToVal(char c) noexcept {
    // Assumes input is a valid hex digit.
    return (c >= '0' && c <= '9') ? (c - '0')
         : (c >= 'a' && c <= 'f') ? (c - 'a' + 10)
         : (c - 'A' + 10);   // 'A'‑'F'
}
std::vector<uint8_t> hexToBytes(const std::string& hex) {
    size_t n = hex.size() / 2;
    std::vector<uint8_t> bytes(n);
    for (size_t i = 0; i < n; ++i) {
        uint8_t hi = hexCharToVal(hex[2 * i]);
        uint8_t lo = hexCharToVal(hex[2 * i + 1]);
        bytes[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return bytes;
}


static const char* hexDigits = "0123456789abcdef";

std::string bytesToHex(const std::vector<uint8_t>& bytes) {
    // Pre‑allocate the exact output size (2 characters per byte).
    std::string out;
    out.resize(bytes.size() * 2);
    size_t pos = 0;
    for (uint8_t b : bytes) {
        out[pos++] = hexDigits[b >> 4];
        out[pos++] = hexDigits[b & 0x0F];
    }
    return out;
}




int main() {
    std::ios::sync_with_stdio(false);
    std::ifstream infile("test_in.txt");
    std::ofstream outfile("test_out_custom.txt");

    if (!infile.is_open() || !outfile.is_open()) {
        std::cerr << "无法打开输入或输出文件！" << std::endl;
        return 1;
    }

    AES128_CTR aes_ctr;
    std::string keyHex, ivHex, plainHex;

    
    while (std::getline(infile, keyHex) && 
           std::getline(infile, ivHex) && 
           std::getline(infile, plainHex)) {
        
        std::vector<uint8_t> key = hexToBytes(keyHex);
        std::vector<uint8_t> iv = hexToBytes(ivHex);
        std::vector<uint8_t> plaintext = hexToBytes(plainHex);

        std::vector<uint8_t> ciphertext = aes_ctr.process(plaintext, key, iv);
        
        outfile << bytesToHex(ciphertext) << std::endl;
    }

    infile.close();
    outfile.close();
    return 0;
}
// EVOLVE-BLOCK-END
