// EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>




class AES128 {
private:
    static const uint8_t sbox[256];
    static const uint8_t Rcon[11];
    
    int Nk = 4, Nr = 10, Nb = 4;
    uint8_t RoundKey[176];

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

public:
    void setKey(const uint8_t* key) {
        KeyExpansion(key);
    }

    void encryptBlock(const uint8_t* input, uint8_t* output) const {
        uint8_t state[4][4];
        for (int i = 0; i < 16; i++) state[i % 4][i / 4] = input[i];
        AddRoundKey(state, 0);
        for (int round = 1; round < Nr; round++) {
            SubBytes(state); ShiftRows(state); MixColumns(state); AddRoundKey(state, round);
        }
        SubBytes(state); ShiftRows(state); AddRoundKey(state, Nr);
        for (int i = 0; i < 16; i++) output[i] = state[i % 4][i / 4];
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
    uint8_t keyStream[16];
    inline void incrementCounter(uint8_t* counter) {
        for (int i = 15; i >= 0; --i) {
            if (++counter[i] != 0) break;
        }
    }
public:
    void setKey(const uint8_t* key) {
        aes.setKey(key);
    }

    void process(const uint8_t* data, uint8_t* result, size_t dataSize, const uint8_t* iv) {
        uint8_t counterBlock[16];
        std::copy(iv, iv + 16, counterBlock);
        size_t i = 0;
        // Process full blocks first
        while (i + 16 <= dataSize) {
            aes.encryptBlock(counterBlock, keyStream);
            // XOR 16 bytes at once
            for (int j = 0; j < 16; ++j) {
                result[i + j] = data[i + j] ^ keyStream[j];
            }
            incrementCounter(counterBlock);
            i += 16;
        }
        // Process remaining bytes
        if (i < dataSize) {
            aes.encryptBlock(counterBlock, keyStream);
            for (size_t j = 0; j < dataSize - i; ++j) {
                result[i + j] = data[i + j] ^ keyStream[j];
            }
        }
    }
};




std::vector<uint8_t> hexToBytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    bytes.reserve(hex.size() / 2);
    static const int8_t hexLookup[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,1,2,3,4,5,6,7,8,9,-1,-1,-1,-1,-1,-1,
        -1,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
    };
    for (size_t i = 0; i < hex.length(); i += 2) {
        int high = hexLookup[static_cast<uint8_t>(hex[i])];
        int low = hexLookup[static_cast<uint8_t>(hex[i+1])];
        if (high == -1 || low == -1) break; // Invalid hex
        bytes.push_back((high << 4) | low);
    }
    return bytes;
}


std::string bytesToHex(const std::vector<uint8_t>& bytes) {
    static const char hexDigits[] = "0123456789abcdef";
    std::string result;
    result.reserve(bytes.size() * 2);
    for (uint8_t b : bytes) {
        result.push_back(hexDigits[b >> 4]);
        result.push_back(hexDigits[b & 0x0F]);
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
        
        std::vector<uint8_t> keyVec = hexToBytes(keyHex);
        std::vector<uint8_t> ivVec = hexToBytes(ivHex);
        std::vector<uint8_t> plaintext = hexToBytes(plainHex);
        
        // Set key once per test case
        aes_ctr.setKey(keyVec.data());
        
        std::vector<uint8_t> ciphertext(plaintext.size());
        aes_ctr.process(plaintext.data(), ciphertext.data(), plaintext.size(), ivVec.data());
        
        outfile << bytesToHex(ciphertext) << std::endl;
    }

    infile.close();
    outfile.close();
    return 0;
}
// EVOLVE-BLOCK-END
