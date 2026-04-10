// EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>

class AES128 {
private:
    static const uint8_t sbox[256];
    static const uint8_t Rcon[11];
    static const uint8_t Mult2[256];
    static const uint8_t Mult3[256];

    uint8_t RoundKey[176];
    uint8_t lastKey[16];
    bool keyInitialized = false;

    void KeyExpansion(const uint8_t* key) {
        uint8_t temp[4];
        for (int i = 0; i < 16; i++) RoundKey[i] = key[i];
        int bytesGenerated = 16, rconIteration = 1;

        while (bytesGenerated < 176) {
            temp[0] = RoundKey[bytesGenerated - 4];
            temp[1] = RoundKey[bytesGenerated - 3];
            temp[2] = RoundKey[bytesGenerated - 2];
            temp[3] = RoundKey[bytesGenerated - 1];
            if (bytesGenerated % 16 == 0) {
                uint8_t tmp = temp[0];
                temp[0] = sbox[temp[1]]; temp[1] = sbox[temp[2]];
                temp[2] = sbox[temp[3]]; temp[3] = sbox[tmp];
                temp[0] ^= Rcon[rconIteration++];
            }
            RoundKey[bytesGenerated] = RoundKey[bytesGenerated - 16] ^ temp[0];
            RoundKey[bytesGenerated + 1] = RoundKey[bytesGenerated - 15] ^ temp[1];
            RoundKey[bytesGenerated + 2] = RoundKey[bytesGenerated - 14] ^ temp[2];
            RoundKey[bytesGenerated + 3] = RoundKey[bytesGenerated - 13] ^ temp[3];
            bytesGenerated += 4;
        }
    }

    bool keyChanged(const uint8_t* key) {
        return memcmp(key, lastKey, 16) != 0;
    }

    void updateKey(const uint8_t* key) {
        memcpy(lastKey, key, 16);
        KeyExpansion(key);
        keyInitialized = true;
    }

    inline void AddRoundKey(uint8_t state[4][4], int round) {
        const uint8_t* rk = RoundKey + round * 16;
        state[0][0] ^= rk[0];  state[1][0] ^= rk[1];  state[2][0] ^= rk[2];  state[3][0] ^= rk[3];
        state[0][1] ^= rk[4];  state[1][1] ^= rk[5];  state[2][1] ^= rk[6];  state[3][1] ^= rk[7];
        state[0][2] ^= rk[8];  state[1][2] ^= rk[9];  state[2][2] ^= rk[10]; state[3][2] ^= rk[11];
        state[0][3] ^= rk[12]; state[1][3] ^= rk[13]; state[2][3] ^= rk[14]; state[3][3] ^= rk[15];
    }

    inline void SubBytes(uint8_t state[4][4]) {
        state[0][0] = sbox[state[0][0]]; state[0][1] = sbox[state[0][1]]; state[0][2] = sbox[state[0][2]]; state[0][3] = sbox[state[0][3]];
        state[1][0] = sbox[state[1][0]]; state[1][1] = sbox[state[1][1]]; state[1][2] = sbox[state[1][2]]; state[1][3] = sbox[state[1][3]];
        state[2][0] = sbox[state[2][0]]; state[2][1] = sbox[state[2][1]]; state[2][2] = sbox[state[2][2]]; state[2][3] = sbox[state[2][3]];
        state[3][0] = sbox[state[3][0]]; state[3][1] = sbox[state[3][1]]; state[3][2] = sbox[state[3][2]]; state[3][3] = sbox[state[3][3]];
    }

    inline void ShiftRows(uint8_t state[4][4]) {
        uint8_t temp;
        temp = state[1][0]; state[1][0] = state[1][1]; state[1][1] = state[1][2]; state[1][2] = state[1][3]; state[1][3] = temp;
        temp = state[2][0]; uint8_t temp2 = state[2][1]; state[2][0] = state[2][2]; state[2][1] = state[2][3]; state[2][2] = temp; state[2][3] = temp2;
        temp = state[3][3]; state[3][3] = state[3][2]; state[3][2] = state[3][1]; state[3][1] = state[3][0]; state[3][0] = temp;
    }

    // Combined SubBytes + ShiftRows for efficiency
    inline void SubBytesShiftRows(uint8_t state[4][4]) {
        // Row 0: no shift, just SubBytes
        state[0][0] = sbox[state[0][0]]; state[0][1] = sbox[state[0][1]]; state[0][2] = sbox[state[0][2]]; state[0][3] = sbox[state[0][3]];
        // Row 1: shift left by 1
        uint8_t t1 = sbox[state[1][1]]; uint8_t t2 = sbox[state[1][2]]; uint8_t t3 = sbox[state[1][3]]; uint8_t t0 = sbox[state[1][0]];
        state[1][0] = t1; state[1][1] = t2; state[1][2] = t3; state[1][3] = t0;
        // Row 2: shift left by 2
        t0 = sbox[state[2][2]]; t1 = sbox[state[2][3]]; t2 = sbox[state[2][0]]; t3 = sbox[state[2][1]];
        state[2][0] = t0; state[2][1] = t1; state[2][2] = t2; state[2][3] = t3;
        // Row 3: shift left by 3 (or right by 1)
        t0 = sbox[state[3][3]]; t1 = sbox[state[3][0]]; t2 = sbox[state[3][1]]; t3 = sbox[state[3][2]];
        state[3][0] = t0; state[3][1] = t1; state[3][2] = t2; state[3][3] = t3;
    }

    inline void MixColumns(uint8_t state[4][4]) {
        // Column 0
        uint8_t s0 = state[0][0], s1 = state[1][0], s2 = state[2][0], s3 = state[3][0];
        state[0][0] = Mult2[s0] ^ Mult3[s1] ^ s2 ^ s3;
        state[1][0] = s0 ^ Mult2[s1] ^ Mult3[s2] ^ s3;
        state[2][0] = s0 ^ s1 ^ Mult2[s2] ^ Mult3[s3];
        state[3][0] = Mult3[s0] ^ s1 ^ s2 ^ Mult2[s3];
        // Column 1
        s0 = state[0][1]; s1 = state[1][1]; s2 = state[2][1]; s3 = state[3][1];
        state[0][1] = Mult2[s0] ^ Mult3[s1] ^ s2 ^ s3;
        state[1][1] = s0 ^ Mult2[s1] ^ Mult3[s2] ^ s3;
        state[2][1] = s0 ^ s1 ^ Mult2[s2] ^ Mult3[s3];
        state[3][1] = Mult3[s0] ^ s1 ^ s2 ^ Mult2[s3];
        // Column 2
        s0 = state[0][2]; s1 = state[1][2]; s2 = state[2][2]; s3 = state[3][2];
        state[0][2] = Mult2[s0] ^ Mult3[s1] ^ s2 ^ s3;
        state[1][2] = s0 ^ Mult2[s1] ^ Mult3[s2] ^ s3;
        state[2][2] = s0 ^ s1 ^ Mult2[s2] ^ Mult3[s3];
        state[3][2] = Mult3[s0] ^ s1 ^ s2 ^ Mult2[s3];
        // Column 3
        s0 = state[0][3]; s1 = state[1][3]; s2 = state[2][3]; s3 = state[3][3];
        state[0][3] = Mult2[s0] ^ Mult3[s1] ^ s2 ^ s3;
        state[1][3] = s0 ^ Mult2[s1] ^ Mult3[s2] ^ s3;
        state[2][3] = s0 ^ s1 ^ Mult2[s2] ^ Mult3[s3];
        state[3][3] = Mult3[s0] ^ s1 ^ s2 ^ Mult2[s3];
    }

public:
    void encryptBlock(const uint8_t* plaintext, const uint8_t* key, uint8_t* ciphertext) {
        if (!keyInitialized || keyChanged(key)) {
            updateKey(key);
        }
        uint8_t state[4][4];
        state[0][0] = plaintext[0];  state[1][0] = plaintext[1];  state[2][0] = plaintext[2];  state[3][0] = plaintext[3];
        state[0][1] = plaintext[4];  state[1][1] = plaintext[5];  state[2][1] = plaintext[6];  state[3][1] = plaintext[7];
        state[0][2] = plaintext[8];  state[1][2] = plaintext[9];  state[2][2] = plaintext[10]; state[3][2] = plaintext[11];
        state[0][3] = plaintext[12]; state[1][3] = plaintext[13]; state[2][3] = plaintext[14]; state[3][3] = plaintext[15];

        AddRoundKey(state, 0);
        // Unrolled rounds 1-9
        SubBytesShiftRows(state); MixColumns(state); AddRoundKey(state, 1);
        SubBytesShiftRows(state); MixColumns(state); AddRoundKey(state, 2);
        SubBytesShiftRows(state); MixColumns(state); AddRoundKey(state, 3);
        SubBytesShiftRows(state); MixColumns(state); AddRoundKey(state, 4);
        SubBytesShiftRows(state); MixColumns(state); AddRoundKey(state, 5);
        SubBytesShiftRows(state); MixColumns(state); AddRoundKey(state, 6);
        SubBytesShiftRows(state); MixColumns(state); AddRoundKey(state, 7);
        SubBytesShiftRows(state); MixColumns(state); AddRoundKey(state, 8);
        SubBytesShiftRows(state); MixColumns(state); AddRoundKey(state, 9);
        // Final round (no MixColumns)
        SubBytesShiftRows(state); AddRoundKey(state, 10);

        ciphertext[0] = state[0][0];  ciphertext[1] = state[1][0];  ciphertext[2] = state[2][0];  ciphertext[3] = state[3][0];
        ciphertext[4] = state[0][1];  ciphertext[5] = state[1][1];  ciphertext[6] = state[2][1];  ciphertext[7] = state[3][1];
        ciphertext[8] = state[0][2];  ciphertext[9] = state[1][2];  ciphertext[10] = state[2][2]; ciphertext[11] = state[3][2];
        ciphertext[12] = state[0][3]; ciphertext[13] = state[1][3]; ciphertext[14] = state[2][3]; ciphertext[15] = state[3][3];
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

const uint8_t AES128::Mult2[256] = {
    0x00, 0x02, 0x04, 0x06, 0x08, 0x0a, 0x0c, 0x0e, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e,
    0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e,
    0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 0x5c, 0x5e,
    0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e, 0x70, 0x72, 0x74, 0x76, 0x78, 0x7a, 0x7c, 0x7e,
    0x80, 0x82, 0x84, 0x86, 0x88, 0x8a, 0x8c, 0x8e, 0x90, 0x92, 0x94, 0x96, 0x98, 0x9a, 0x9c, 0x9e,
    0xa0, 0xa2, 0xa4, 0xa6, 0xa8, 0xaa, 0xac, 0xae, 0xb0, 0xb2, 0xb4, 0xb6, 0xb8, 0xba, 0xbc, 0xbe,
    0xc0, 0xc2, 0xc4, 0xc6, 0xc8, 0xca, 0xcc, 0xce, 0xd0, 0xd2, 0xd4, 0xd6, 0xd8, 0xda, 0xdc, 0xde,
    0xe0, 0xe2, 0xe4, 0xe6, 0xe8, 0xea, 0xec, 0xee, 0xf0, 0xf2, 0xf4, 0xf6, 0xf8, 0xfa, 0xfc, 0xfe,
    0x1b, 0x19, 0x1f, 0x1d, 0x13, 0x11, 0x17, 0x15, 0x0b, 0x09, 0x0f, 0x0d, 0x03, 0x01, 0x07, 0x05,
    0x3b, 0x39, 0x3f, 0x3d, 0x33, 0x31, 0x37, 0x35, 0x2b, 0x29, 0x2f, 0x2d, 0x23, 0x21, 0x27, 0x25,
    0x5b, 0x59, 0x5f, 0x5d, 0x53, 0x51, 0x57, 0x55, 0x4b, 0x49, 0x4f, 0x4d, 0x43, 0x41, 0x47, 0x45,
    0x7b, 0x79, 0x7f, 0x7d, 0x73, 0x71, 0x77, 0x75, 0x6b, 0x69, 0x6f, 0x6d, 0x63, 0x61, 0x67, 0x65,
    0x9b, 0x99, 0x9f, 0x9d, 0x93, 0x91, 0x97, 0x95, 0x8b, 0x89, 0x8f, 0x8d, 0x83, 0x81, 0x87, 0x85,
    0xbb, 0xb9, 0xbf, 0xbd, 0xb3, 0xb1, 0xb7, 0xb5, 0xab, 0xa9, 0xaf, 0xad, 0xa3, 0xa1, 0xa7, 0xa5,
    0xdb, 0xd9, 0xdf, 0xdd, 0xd3, 0xd1, 0xd7, 0xd5, 0xcb, 0xc9, 0xcf, 0xcd, 0xc3, 0xc1, 0xc7, 0xc5,
    0xfb, 0xf9, 0xff, 0xfd, 0xf3, 0xf1, 0xf7, 0xf5, 0xeb, 0xe9, 0xef, 0xed, 0xe3, 0xe1, 0xe7, 0xe5
};

const uint8_t AES128::Mult3[256] = {
    0x00, 0x03, 0x06, 0x05, 0x0c, 0x0f, 0x0a, 0x09, 0x18, 0x1b, 0x1e, 0x1d, 0x14, 0x17, 0x12, 0x11,
    0x30, 0x33, 0x36, 0x35, 0x3c, 0x3f, 0x3a, 0x39, 0x28, 0x2b, 0x2e, 0x2d, 0x24, 0x27, 0x22, 0x21,
    0x60, 0x63, 0x66, 0x65, 0x6c, 0x6f, 0x6a, 0x69, 0x78, 0x7b, 0x7e, 0x7d, 0x74, 0x77, 0x72, 0x71,
    0x50, 0x53, 0x56, 0x55, 0x5c, 0x5f, 0x5a, 0x59, 0x48, 0x4b, 0x4e, 0x4d, 0x44, 0x47, 0x42, 0x41,
    0xc0, 0xc3, 0xc6, 0xc5, 0xcc, 0xcf, 0xca, 0xc9, 0xd8, 0xdb, 0xde, 0xdd, 0xd4, 0xd7, 0xd2, 0xd1,
    0xf0, 0xf3, 0xf6, 0xf5, 0xfc, 0xff, 0xfa, 0xf9, 0xe8, 0xeb, 0xee, 0xed, 0xe4, 0xe7, 0xe2, 0xe1,
    0xa0, 0xa3, 0xa6, 0xa5, 0xac, 0xaf, 0xaa, 0xa9, 0xb8, 0xbb, 0xbe, 0xbd, 0xb4, 0xb7, 0xb2, 0xb1,
    0x90, 0x93, 0x96, 0x95, 0x9c, 0x9f, 0x9a, 0x99, 0x88, 0x8b, 0x8e, 0x8d, 0x84, 0x87, 0x82, 0x81,
    0x9b, 0x98, 0x9d, 0x9e, 0x97, 0x94, 0x91, 0x92, 0x83, 0x80, 0x85, 0x86, 0x8f, 0x8c, 0x89, 0x8a,
    0xab, 0xa8, 0xad, 0xae, 0xa7, 0xa4, 0xa1, 0xa2, 0xb3, 0xb0, 0xb5, 0xb6, 0xbf, 0xbc, 0xb9, 0xba,
    0xfb, 0xf8, 0xfd, 0xfe, 0xf7, 0xf4, 0xf1, 0xf2, 0xe3, 0xe0, 0xe5, 0xe6, 0xef, 0xec, 0xe9, 0xea,
    0xcb, 0xc8, 0xcd, 0xce, 0xc7, 0xc4, 0xc1, 0xc2, 0xd3, 0xd0, 0xd5, 0xd6, 0xdf, 0xdc, 0xd9, 0xda,
    0x5b, 0x58, 0x5d, 0x5e, 0x57, 0x54, 0x51, 0x52, 0x43, 0x40, 0x45, 0x46, 0x4f, 0x4c, 0x49, 0x4a,
    0x6b, 0x68, 0x6d, 0x6e, 0x67, 0x64, 0x61, 0x62, 0x73, 0x70, 0x75, 0x76, 0x7f, 0x7c, 0x79, 0x7a,
    0x3b, 0x38, 0x3d, 0x3e, 0x37, 0x34, 0x31, 0x32, 0x23, 0x20, 0x25, 0x26, 0x2f, 0x2c, 0x29, 0x2a,
    0x0b, 0x08, 0x0d, 0x0e, 0x07, 0x04, 0x01, 0x02, 0x13, 0x10, 0x15, 0x16, 0x1f, 0x1c, 0x19, 0x1a
};

class AES128_CTR {
private:
    AES128 aes;
    void incrementCounter(uint8_t* counter) {
        for (int i = 15; i >= 0; --i) {
            if (++counter[i] != 0) break;
        }
    }
public:
    std::vector<uint8_t> process(const std::vector<uint8_t>& data, const std::vector<uint8_t>& key, const std::vector<uint8_t>& iv) {
        std::vector<uint8_t> result(data.size());
        uint8_t counterBlock[16];
        uint8_t keyStream[16];
        memcpy(counterBlock, iv.data(), 16);

        size_t fullBlocks = data.size() / 16;
        size_t remainder = data.size() % 16;

        for (size_t block = 0; block < fullBlocks; ++block) {
            aes.encryptBlock(counterBlock, key.data(), keyStream);
            incrementCounter(counterBlock);

            size_t start = block * 16;
            uint64_t* k64 = (uint64_t*)keyStream;
            uint64_t* d64 = (uint64_t*)(data.data() + start);
            uint64_t* r64 = (uint64_t*)(result.data() + start);
            r64[0] = d64[0] ^ k64[0];
            r64[1] = d64[1] ^ k64[1];
        }

        if (remainder > 0) {
            aes.encryptBlock(counterBlock, key.data(), keyStream);
            size_t start = fullBlocks * 16;
            for (size_t i = 0; i < remainder; ++i) {
                result[start + i] = data[start + i] ^ keyStream[i];
            }
        }
        return result;
    }
};

static const int hexLookup[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0,
    0, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

std::vector<uint8_t> hexToBytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    bytes.reserve(hex.length() / 2);
    for (size_t i = 0; i < hex.length(); i += 2) {
        bytes.push_back((hexLookup[(uint8_t)hex[i]] << 4) | hexLookup[(uint8_t)hex[i + 1]]);
    }
    return bytes;
}

std::string bytesToHex(const std::vector<uint8_t>& bytes) {
    static const char hexChars[] = "0123456789abcdef";
    std::string result;
    result.reserve(bytes.size() * 2);
    for (uint8_t b : bytes) {
        result.push_back(hexChars[b >> 4]);
        result.push_back(hexChars[b & 0xf]);
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

        std::vector<uint8_t> ciphertext = aes_ctr.process(plaintext, key, iv);

        outfile << bytesToHex(ciphertext) << std::endl;
    }

    infile.close();
    outfile.close();
    return 0;
}
// EVOLVE-BLOCK-END