// EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>




class AES128 {
private:
    static const uint8_t sbox[256];
    static const uint8_t Rcon[11];
    
    uint8_t RoundKey[176];
    bool keyExpanded = false;
    uint8_t lastKey[16];

    static inline uint8_t xtime(uint8_t x) { return (x << 1) ^ (((x >> 7) & 1) * 0x1b); }

    void KeyExpansion(const uint8_t* key) {
        uint8_t temp[4];
        for (int i = 0; i < 16; i++) { RoundKey[i] = key[i]; lastKey[i] = key[i]; }
        int bytesGenerated = 16, rconIteration = 1;
        while (bytesGenerated < 176) {
            temp[0] = RoundKey[bytesGenerated-4]; temp[1] = RoundKey[bytesGenerated-3];
            temp[2] = RoundKey[bytesGenerated-2]; temp[3] = RoundKey[bytesGenerated-1];
            if (bytesGenerated % 16 == 0) {
                uint8_t t = temp[0]; temp[0] = sbox[temp[1]]; temp[1] = sbox[temp[2]];
                temp[2] = sbox[temp[3]]; temp[3] = sbox[t];
                temp[0] ^= Rcon[rconIteration++];
            }
            for (int i = 0; i < 4; i++) {
                RoundKey[bytesGenerated] = RoundKey[bytesGenerated - 16] ^ temp[i];
                bytesGenerated++;
            }
        }
        keyExpanded = true;
    }

    inline void AddRoundKey(uint8_t* s, int round) {
        const uint8_t* rk = &RoundKey[round * 16];
        for (int i = 0; i < 16; i++) s[i] ^= rk[i];
    }

    inline void SubBytes(uint8_t* s) {
        for (int i = 0; i < 16; i++) s[i] = sbox[s[i]];
    }

    inline void ShiftRows(uint8_t* s) {
        uint8_t t;
        // Row 1: shift left 1
        t = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = t;
        // Row 2: shift left 2
        t = s[2]; s[2] = s[10]; s[10] = t; t = s[6]; s[6] = s[14]; s[14] = t;
        // Row 3: shift left 3 (= right 1)
        t = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = t;
    }

    inline void MixColumns(uint8_t* s) {
        for (int c = 0; c < 4; c++) {
            uint8_t* col = s + c * 4;
            uint8_t a0 = col[0], a1 = col[1], a2 = col[2], a3 = col[3];
            uint8_t x0 = xtime(a0), x1 = xtime(a1), x2 = xtime(a2), x3 = xtime(a3);
            col[0] = x0 ^ x1 ^ a1 ^ a2 ^ a3;
            col[1] = a0 ^ x1 ^ x2 ^ a2 ^ a3;
            col[2] = a0 ^ a1 ^ x2 ^ x3 ^ a3;
            col[3] = x0 ^ a0 ^ a1 ^ a2 ^ x3;
        }
    }

public:
    void setKey(const uint8_t* key) {
        if (!keyExpanded || __builtin_expect(memcmp(lastKey, key, 16) != 0, 0))
            KeyExpansion(key);
    }

    void encryptBlock(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
        uint8_t s[16];
        memcpy(s, in, 16);
        AddRoundKey(s, 0);
        for (int round = 1; round < 10; round++) {
            SubBytes(s); ShiftRows(s); MixColumns(s); AddRoundKey(s, round);
        }
        SubBytes(s); ShiftRows(s); AddRoundKey(s, 10);
        memcpy(out, s, 16);
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
    static inline void incrementCounter(uint8_t* counter) {
        for (int i = 15; i >= 0; --i) {
            if (++counter[i] != 0) break;
        }
    }
public:
    void process(const uint8_t* data, size_t len, const uint8_t* key, const uint8_t* iv, uint8_t* result) {
        aes.setKey(key);
        uint8_t counterBlock[16];
        uint8_t keyStream[16];
        memcpy(counterBlock, iv, 16);
        size_t fullBlocks = len / 16;
        size_t i = 0;
        for (size_t b = 0; b < fullBlocks; b++) {
            aes.encryptBlock(counterBlock, keyStream);
            incrementCounter(counterBlock);
            const uint8_t* dp = data + i;
            uint8_t* rp = result + i;
            for (int j = 0; j < 16; j++) rp[j] = dp[j] ^ keyStream[j];
            i += 16;
        }
        if (i < len) {
            aes.encryptBlock(counterBlock, keyStream);
            for (size_t j = 0; i < len; i++, j++) {
                result[i] = data[i] ^ keyStream[j];
            }
        }
    }
};




static inline uint8_t hexCharVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

std::vector<uint8_t> hexToBytes(const std::string& hex) {
    size_t len = hex.length() / 2;
    std::vector<uint8_t> bytes(len);
    for (size_t i = 0; i < len; i++) {
        bytes[i] = (hexCharVal(hex[2*i]) << 4) | hexCharVal(hex[2*i+1]);
    }
    return bytes;
}


std::string bytesToHex(const std::vector<uint8_t>& bytes) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string result(bytes.size() * 2, '0');
    for (size_t i = 0; i < bytes.size(); i++) {
        result[2*i] = hex_chars[bytes[i] >> 4];
        result[2*i+1] = hex_chars[bytes[i] & 0x0F];
    }
    return result;
}




int main() {
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

        std::vector<uint8_t> ciphertext(plaintext.size());
        aes_ctr.process(plaintext.data(), plaintext.size(), key.data(), iv.data(), ciphertext.data());
        
        outfile << bytesToHex(ciphertext) << std::endl;
    }

    infile.close();
    outfile.close();
    return 0;
}
// EVOLVE-BLOCK-END
