// EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdint>
#include <string>
#include <fstream>





class AES128 {
private:
    static const uint8_t sbox[256];
    static const uint8_t Rcon[11];
    uint8_t RoundKey[176];

    static inline uint8_t xtime(uint8_t x) { return (uint8_t)((x << 1) ^ (((x >> 7) & 1) * 0x1b)); }

    void SubWord(uint8_t* word) {
        word[0] = sbox[word[0]];
        word[1] = sbox[word[1]];
        word[2] = sbox[word[2]];
        word[3] = sbox[word[3]];
    }

    void RotWord(uint8_t* word) {
        uint8_t tmp = word[0];
        word[0] = word[1]; word[1] = word[2]; word[2] = word[3]; word[3] = tmp;
    }

    void KeyExpansion(const std::vector<uint8_t>& key) {
        uint8_t temp[4];
        for (int i = 0; i < 16; ++i) RoundKey[i] = key[i];
        int bytesGenerated = 16, rconIteration = 1;
        while (bytesGenerated < 176) {
            temp[0] = RoundKey[bytesGenerated - 4];
            temp[1] = RoundKey[bytesGenerated - 3];
            temp[2] = RoundKey[bytesGenerated - 2];
            temp[3] = RoundKey[bytesGenerated - 1];
            if ((bytesGenerated & 15) == 0) {
                RotWord(temp);
                SubWord(temp);
                temp[0] ^= Rcon[rconIteration++];
            }
            RoundKey[bytesGenerated] = RoundKey[bytesGenerated - 16] ^ temp[0]; ++bytesGenerated;
            RoundKey[bytesGenerated] = RoundKey[bytesGenerated - 16] ^ temp[1]; ++bytesGenerated;
            RoundKey[bytesGenerated] = RoundKey[bytesGenerated - 16] ^ temp[2]; ++bytesGenerated;
            RoundKey[bytesGenerated] = RoundKey[bytesGenerated - 16] ^ temp[3]; ++bytesGenerated;
        }
    }

public:
    void setKey(const std::vector<uint8_t>& key) { KeyExpansion(key); }

    void encryptBlock(const uint8_t in[16], uint8_t out[16]) {
        uint8_t s0 = in[0] ^ RoundKey[0], s1 = in[1] ^ RoundKey[1], s2 = in[2] ^ RoundKey[2], s3 = in[3] ^ RoundKey[3];
        uint8_t s4 = in[4] ^ RoundKey[4], s5 = in[5] ^ RoundKey[5], s6 = in[6] ^ RoundKey[6], s7 = in[7] ^ RoundKey[7];
        uint8_t s8 = in[8] ^ RoundKey[8], s9 = in[9] ^ RoundKey[9], s10 = in[10] ^ RoundKey[10], s11 = in[11] ^ RoundKey[11];
        uint8_t s12 = in[12] ^ RoundKey[12], s13 = in[13] ^ RoundKey[13], s14 = in[14] ^ RoundKey[14], s15 = in[15] ^ RoundKey[15];

        for (int round = 1; round < 10; ++round) {
            uint8_t t0 = sbox[s0], t1 = sbox[s5], t2 = sbox[s10], t3 = sbox[s15];
            uint8_t t4 = sbox[s4], t5 = sbox[s9], t6 = sbox[s14], t7 = sbox[s3];
            uint8_t t8 = sbox[s8], t9 = sbox[s13], t10 = sbox[s2], t11 = sbox[s7];
            uint8_t t12 = sbox[s12], t13 = sbox[s1], t14 = sbox[s6], t15 = sbox[s11];
            uint8_t u0 = t0 ^ t1 ^ t2 ^ t3, u4 = t4 ^ t5 ^ t6 ^ t7, u8 = t8 ^ t9 ^ t10 ^ t11, u12 = t12 ^ t13 ^ t14 ^ t15;
            const uint8_t* rk = RoundKey + (round << 4);

            s0 = t0 ^ u0 ^ xtime(t0 ^ t1) ^ rk[0];
            s1 = t1 ^ u0 ^ xtime(t1 ^ t2) ^ rk[1];
            s2 = t2 ^ u0 ^ xtime(t2 ^ t3) ^ rk[2];
            s3 = t3 ^ u0 ^ xtime(t3 ^ t0) ^ rk[3];
            s4 = t4 ^ u4 ^ xtime(t4 ^ t5) ^ rk[4];
            s5 = t5 ^ u4 ^ xtime(t5 ^ t6) ^ rk[5];
            s6 = t6 ^ u4 ^ xtime(t6 ^ t7) ^ rk[6];
            s7 = t7 ^ u4 ^ xtime(t7 ^ t4) ^ rk[7];
            s8 = t8 ^ u8 ^ xtime(t8 ^ t9) ^ rk[8];
            s9 = t9 ^ u8 ^ xtime(t9 ^ t10) ^ rk[9];
            s10 = t10 ^ u8 ^ xtime(t10 ^ t11) ^ rk[10];
            s11 = t11 ^ u8 ^ xtime(t11 ^ t8) ^ rk[11];
            s12 = t12 ^ u12 ^ xtime(t12 ^ t13) ^ rk[12];
            s13 = t13 ^ u12 ^ xtime(t13 ^ t14) ^ rk[13];
            s14 = t14 ^ u12 ^ xtime(t14 ^ t15) ^ rk[14];
            s15 = t15 ^ u12 ^ xtime(t15 ^ t12) ^ rk[15];
        }

        out[0] = sbox[s0] ^ RoundKey[160];
        out[1] = sbox[s5] ^ RoundKey[161];
        out[2] = sbox[s10] ^ RoundKey[162];
        out[3] = sbox[s15] ^ RoundKey[163];
        out[4] = sbox[s4] ^ RoundKey[164];
        out[5] = sbox[s9] ^ RoundKey[165];
        out[6] = sbox[s14] ^ RoundKey[166];
        out[7] = sbox[s3] ^ RoundKey[167];
        out[8] = sbox[s8] ^ RoundKey[168];
        out[9] = sbox[s13] ^ RoundKey[169];
        out[10] = sbox[s2] ^ RoundKey[170];
        out[11] = sbox[s7] ^ RoundKey[171];
        out[12] = sbox[s12] ^ RoundKey[172];
        out[13] = sbox[s1] ^ RoundKey[173];
        out[14] = sbox[s6] ^ RoundKey[174];
        out[15] = sbox[s11] ^ RoundKey[175];
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
    static inline void incrementCounter(uint8_t counter[16]) {
        for (int i = 15; i >= 0; --i) if (++counter[i] != 0) break;
    }
public:
    std::vector<uint8_t> process(const std::vector<uint8_t>& data, const std::vector<uint8_t>& key, const std::vector<uint8_t>& iv) {
        std::vector<uint8_t> result(data.size());
        uint8_t counter[16], stream[16];
        for (int i = 0; i < 16; ++i) counter[i] = iv[i];
        aes.setKey(key);

        const uint8_t* in = data.data();
        uint8_t* out = result.data();
        size_t i = 0, n = data.size();

        for (; i + 16 <= n; i += 16) {
            aes.encryptBlock(counter, stream);
            incrementCounter(counter);
            out[i] = in[i] ^ stream[0];
            out[i + 1] = in[i + 1] ^ stream[1];
            out[i + 2] = in[i + 2] ^ stream[2];
            out[i + 3] = in[i + 3] ^ stream[3];
            out[i + 4] = in[i + 4] ^ stream[4];
            out[i + 5] = in[i + 5] ^ stream[5];
            out[i + 6] = in[i + 6] ^ stream[6];
            out[i + 7] = in[i + 7] ^ stream[7];
            out[i + 8] = in[i + 8] ^ stream[8];
            out[i + 9] = in[i + 9] ^ stream[9];
            out[i + 10] = in[i + 10] ^ stream[10];
            out[i + 11] = in[i + 11] ^ stream[11];
            out[i + 12] = in[i + 12] ^ stream[12];
            out[i + 13] = in[i + 13] ^ stream[13];
            out[i + 14] = in[i + 14] ^ stream[14];
            out[i + 15] = in[i + 15] ^ stream[15];
        }
        if (i < n) {
            aes.encryptBlock(counter, stream);
            for (size_t j = 0; i < n; ++i, ++j) out[i] = in[i] ^ stream[j];
        }
        return result;
    }
};




static inline uint8_t hexNibble(char c) {
    return (uint8_t)(c <= '9' ? c - '0' : (c & ~32) - 'A' + 10);
}
std::vector<uint8_t> hexToBytes(const std::string& hex) {
    std::vector<uint8_t> bytes(hex.size() >> 1);
    for (size_t i = 0, j = 0; j < bytes.size(); ++j, i += 2)
        bytes[j] = (uint8_t)((hexNibble(hex[i]) << 4) | hexNibble(hex[i + 1]));
    return bytes;
}


std::string bytesToHex(const std::vector<uint8_t>& bytes) {
    static const char* d = "0123456789abcdef";
    std::string s(bytes.size() * 2, '0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        s[2 * i] = d[bytes[i] >> 4];
        s[2 * i + 1] = d[bytes[i] & 15];
    }
    return s;
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
