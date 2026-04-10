// EVOLVE-BLOCK-START
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>

class SHA3_256 {
private:
    uint64_t state[25]; 
    int pos;            

    
    static inline uint64_t rotl(uint64_t a, unsigned offset) __attribute__((always_inline)) {
        return (a << offset) | (a >> ((-offset) & 63));
    }

    
    void __attribute__((hot)) keccak_f1600() {
        
        static const uint64_t RC[24] = {
            0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
            0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
            0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
            0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
            0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
            0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
            0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
            0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
        };

        for (int round = 0; round < 24; ++round) {
            uint64_t C[5];
            C[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
            C[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
            C[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
            C[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
            C[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];
            uint64_t D0 = C[4] ^ rotl(C[1], 1);
            uint64_t D1 = C[0] ^ rotl(C[2], 1);
            uint64_t D2 = C[1] ^ rotl(C[3], 1);
            uint64_t D3 = C[2] ^ rotl(C[4], 1);
            uint64_t D4 = C[3] ^ rotl(C[0], 1);
            state[0]^=D0; state[5]^=D0; state[10]^=D0; state[15]^=D0; state[20]^=D0;
            state[1]^=D1; state[6]^=D1; state[11]^=D1; state[16]^=D1; state[21]^=D1;
            state[2]^=D2; state[7]^=D2; state[12]^=D2; state[17]^=D2; state[22]^=D2;
            state[3]^=D3; state[8]^=D3; state[13]^=D3; state[18]^=D3; state[23]^=D3;
            state[4]^=D4; state[9]^=D4; state[14]^=D4; state[19]^=D4; state[24]^=D4;
            uint64_t B[25];
            B[0]=state[0]; B[10]=rotl(state[1],1); B[20]=rotl(state[2],62); B[5]=rotl(state[3],28); B[15]=rotl(state[4],27);
            B[16]=rotl(state[5],36); B[1]=rotl(state[6],44); B[11]=rotl(state[7],6); B[21]=rotl(state[8],55); B[6]=rotl(state[9],20);
            B[7]=rotl(state[10],3); B[17]=rotl(state[11],10); B[2]=rotl(state[12],43); B[12]=rotl(state[13],25); B[22]=rotl(state[14],39);
            B[23]=rotl(state[15],41); B[8]=rotl(state[16],45); B[18]=rotl(state[17],15); B[3]=rotl(state[18],21); B[13]=rotl(state[19],8);
            B[14]=rotl(state[20],18); B[24]=rotl(state[21],2); B[9]=rotl(state[22],61); B[19]=rotl(state[23],56); B[4]=rotl(state[24],14);
            for (int y = 0; y < 25; y += 5) {
                uint64_t t0=B[y], t1=B[y+1], t2=B[y+2], t3=B[y+3], t4=B[y+4];
                state[y]=t0^(~t1&t2); state[y+1]=t1^(~t2&t3); state[y+2]=t2^(~t3&t4); state[y+3]=t3^(~t4&t0); state[y+4]=t4^(~t0&t1);
            }
            state[0] ^= RC[round];
        }
    }

    
    void xor_byte(int byte_index, uint8_t byte_val) {
        int word_index = byte_index / 8;
        int shift = (byte_index % 8) * 8;
        state[word_index] ^= ((uint64_t)byte_val << shift);
    }

public:
    SHA3_256() {
        reset();
    }

    
    void reset() {
        for (int i = 0; i < 25; ++i) {
            state[i] = 0;
        }
        pos = 0;
    }

    
    void update(const uint8_t* data, size_t len) {
        const int RATE_BYTES = 136;
        while (len > 0 && pos != 0) {
            xor_byte(pos, *data); data++; len--; pos++;
            if (pos == RATE_BYTES) { keccak_f1600(); pos = 0; }
        }
        while (len >= RATE_BYTES) {
            for (int i = 0; i < 17; ++i) { uint64_t w; __builtin_memcpy(&w, data + i*8, 8); state[i] ^= w; }
            keccak_f1600(); data += RATE_BYTES; len -= RATE_BYTES;
        }
        while (len > 0) {
            if ((pos & 7) == 0 && len >= 8) { uint64_t w; __builtin_memcpy(&w, data, 8); state[pos/8] ^= w; pos += 8; data += 8; len -= 8; }
            else { xor_byte(pos, *data); pos++; data++; len--; }
        }
    }

    void finalize_and_hex(char* out) {
        xor_byte(pos, 0x06); xor_byte(135, 0x80); keccak_f1600();
        static const char hx[] = "0123456789abcdef";
        for (int i = 0; i < 4; ++i) { uint64_t w = state[i]; for (int j = 0; j < 8; ++j) { uint8_t b = (uint8_t)(w & 0xFF); w >>= 8; *out++ = hx[b >> 4]; *out++ = hx[b & 0xF]; } }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) return 1;
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) return 1;
    struct stat st; fstat(fd, &st); size_t filesize = st.st_size;
    SHA3_256 sha3;
    if (filesize > 0) {
        if (filesize <= 65536) {
            uint8_t buf[65536];
            ssize_t n = read(fd, buf, filesize);
            if (n > 0) sha3.update(buf, (size_t)n);
        } else {
            void* mapped = mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
            if (mapped != MAP_FAILED) { sha3.update(reinterpret_cast<const uint8_t*>(mapped), filesize); munmap(mapped, filesize); }
            else { uint8_t buffer[65536]; ssize_t n; while ((n = read(fd, buffer, sizeof(buffer))) > 0) sha3.update(buffer, (size_t)n); }
        }
    }
    close(fd);
    char hex[64]; sha3.finalize_and_hex(hex); write(STDOUT_FILENO, hex, 64);
    return 0;
}
// EVOLVE-BLOCK-END
