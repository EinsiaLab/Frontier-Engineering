// EVOLVE-BLOCK-START
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2")

#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

/* One full Keccak round operating on 25 register variables */
#define KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,rc) \
do { \
    uint64_t c0 = s00^s05^s10^s15^s20; \
    uint64_t c1 = s01^s06^s11^s16^s21; \
    uint64_t c2 = s02^s07^s12^s17^s22; \
    uint64_t c3 = s03^s08^s13^s18^s23; \
    uint64_t c4 = s04^s09^s14^s19^s24; \
    uint64_t d0 = c4^ROTL64(c1,1); \
    uint64_t d1 = c0^ROTL64(c2,1); \
    uint64_t d2 = c1^ROTL64(c3,1); \
    uint64_t d3 = c2^ROTL64(c4,1); \
    uint64_t d4 = c3^ROTL64(c0,1); \
    s00^=d0; s01^=d1; s02^=d2; s03^=d3; s04^=d4; \
    s05^=d0; s06^=d1; s07^=d2; s08^=d3; s09^=d4; \
    s10^=d0; s11^=d1; s12^=d2; s13^=d3; s14^=d4; \
    s15^=d0; s16^=d1; s17^=d2; s18^=d3; s19^=d4; \
    s20^=d0; s21^=d1; s22^=d2; s23^=d3; s24^=d4; \
    uint64_t b00=s00;             uint64_t b01=ROTL64(s06,44); \
    uint64_t b02=ROTL64(s12,43);  uint64_t b03=ROTL64(s18,21); \
    uint64_t b04=ROTL64(s24,14);  uint64_t b05=ROTL64(s03,28); \
    uint64_t b06=ROTL64(s09,20);  uint64_t b07=ROTL64(s10,3); \
    uint64_t b08=ROTL64(s16,45);  uint64_t b09=ROTL64(s22,61); \
    uint64_t b10=ROTL64(s01,1);   uint64_t b11=ROTL64(s07,6); \
    uint64_t b12=ROTL64(s13,25);  uint64_t b13=ROTL64(s19,8); \
    uint64_t b14=ROTL64(s20,18);  uint64_t b15=ROTL64(s04,27); \
    uint64_t b16=ROTL64(s05,36);  uint64_t b17=ROTL64(s11,10); \
    uint64_t b18=ROTL64(s17,15);  uint64_t b19=ROTL64(s23,56); \
    uint64_t b20=ROTL64(s02,62);  uint64_t b21=ROTL64(s08,55); \
    uint64_t b22=ROTL64(s14,39);  uint64_t b23=ROTL64(s15,41); \
    uint64_t b24=ROTL64(s21,2); \
    s00=b00^(~b01&b02)^(rc); s01=b01^(~b02&b03); s02=b02^(~b03&b04); \
    s03=b03^(~b04&b00); s04=b04^(~b00&b01); \
    s05=b05^(~b06&b07); s06=b06^(~b07&b08); s07=b07^(~b08&b09); \
    s08=b08^(~b09&b05); s09=b09^(~b05&b06); \
    s10=b10^(~b11&b12); s11=b11^(~b12&b13); s12=b12^(~b13&b14); \
    s13=b13^(~b14&b10); s14=b14^(~b10&b11); \
    s15=b15^(~b16&b17); s16=b16^(~b17&b18); s17=b17^(~b18&b19); \
    s18=b18^(~b19&b15); s19=b19^(~b15&b16); \
    s20=b20^(~b21&b22); s21=b21^(~b22&b23); s22=b22^(~b23&b24); \
    s23=b23^(~b24&b20); s24=b24^(~b20&b21); \
} while(0)

static __attribute__((hot,noinline)) void keccak_f1600(uint64_t* __restrict__ st) {
    uint64_t s00=st[0],s01=st[1],s02=st[2],s03=st[3],s04=st[4];
    uint64_t s05=st[5],s06=st[6],s07=st[7],s08=st[8],s09=st[9];
    uint64_t s10=st[10],s11=st[11],s12=st[12],s13=st[13],s14=st[14];
    uint64_t s15=st[15],s16=st[16],s17=st[17],s18=st[18],s19=st[19];
    uint64_t s20=st[20],s21=st[21],s22=st[22],s23=st[23],s24=st[24];

    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x0000000000000001ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x0000000000008082ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x800000000000808aULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x8000000080008000ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x000000000000808bULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x0000000080000001ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x8000000080008081ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x8000000000008009ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x000000000000008aULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x0000000000000088ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x0000000080008009ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x000000008000000aULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x000000008000808bULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x800000000000008bULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x8000000000008089ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x8000000000008003ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x8000000000008002ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x8000000000000080ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x000000000000800aULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x800000008000000aULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x8000000080008081ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x8000000000008080ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x0000000080000001ULL);
    KECCAK_ROUND_REG(s00,s01,s02,s03,s04,s05,s06,s07,s08,s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,0x8000000080008008ULL);

    st[0]=s00;st[1]=s01;st[2]=s02;st[3]=s03;st[4]=s04;
    st[5]=s05;st[6]=s06;st[7]=s07;st[8]=s08;st[9]=s09;
    st[10]=s10;st[11]=s11;st[12]=s12;st[13]=s13;st[14]=s14;
    st[15]=s15;st[16]=s16;st[17]=s17;st[18]=s18;st[19]=s19;
    st[20]=s20;st[21]=s21;st[22]=s22;st[23]=s23;st[24]=s24;
}

int main(int argc, char* argv[]) {
    if (argc != 2) return 1;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        const char* err = "Error: Could not open file.";
        (void)!write(STDOUT_FILENO, err, 27);
        return 1;
    }

    struct stat sb;
    if (fstat(fd, &sb) < 0) { close(fd); return 1; }

    size_t file_size = (size_t)sb.st_size;

    alignas(64) uint64_t state[25] = {0};
    static constexpr int RATE_BYTES = 136;

    // For small files, use stack buffer; for large files, use mmap
    if (file_size <= 65536) {
        // Read into stack-allocated buffer
        alignas(8) uint8_t buf[65536 + RATE_BYTES];
        size_t total_read = 0;
        while (total_read < file_size) {
            ssize_t r = read(fd, buf + total_read, file_size - total_read);
            if (r <= 0) break;
            total_read += (size_t)r;
        }
        close(fd);

        size_t offset = 0;
        while (offset + RATE_BYTES <= total_read) {
            const uint64_t* b = (const uint64_t*)(buf + offset);
            state[0]^=b[0]; state[1]^=b[1]; state[2]^=b[2]; state[3]^=b[3];
            state[4]^=b[4]; state[5]^=b[5]; state[6]^=b[6]; state[7]^=b[7];
            state[8]^=b[8]; state[9]^=b[9]; state[10]^=b[10]; state[11]^=b[11];
            state[12]^=b[12]; state[13]^=b[13]; state[14]^=b[14]; state[15]^=b[15];
            state[16]^=b[16];
            keccak_f1600(state);
            offset += RATE_BYTES;
        }

        size_t remaining = total_read - offset;
        alignas(8) uint8_t pad[RATE_BYTES] = {0};
        if (remaining > 0) memcpy(pad, buf + offset, remaining);
        pad[remaining] = 0x06;
        pad[RATE_BYTES - 1] |= 0x80;
        const uint64_t* p = (const uint64_t*)pad;
        state[0]^=p[0]; state[1]^=p[1]; state[2]^=p[2]; state[3]^=p[3];
        state[4]^=p[4]; state[5]^=p[5]; state[6]^=p[6]; state[7]^=p[7];
        state[8]^=p[8]; state[9]^=p[9]; state[10]^=p[10]; state[11]^=p[11];
        state[12]^=p[12]; state[13]^=p[13]; state[14]^=p[14]; state[15]^=p[15];
        state[16]^=p[16];
        keccak_f1600(state);
    } else {
        void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        close(fd);
        if (mapped == MAP_FAILED) return 1;
        const uint8_t* file_data = (const uint8_t*)mapped;
        madvise(mapped, file_size, MADV_SEQUENTIAL);

        size_t offset = 0;
        while (offset + RATE_BYTES <= file_size) {
            const uint64_t* b = (const uint64_t*)(file_data + offset);
            state[0]^=b[0]; state[1]^=b[1]; state[2]^=b[2]; state[3]^=b[3];
            state[4]^=b[4]; state[5]^=b[5]; state[6]^=b[6]; state[7]^=b[7];
            state[8]^=b[8]; state[9]^=b[9]; state[10]^=b[10]; state[11]^=b[11];
            state[12]^=b[12]; state[13]^=b[13]; state[14]^=b[14]; state[15]^=b[15];
            state[16]^=b[16];
            keccak_f1600(state);
            offset += RATE_BYTES;
        }

        size_t remaining = file_size - offset;
        alignas(8) uint8_t pad[RATE_BYTES] = {0};
        if (remaining > 0) memcpy(pad, file_data + offset, remaining);
        pad[remaining] = 0x06;
        pad[RATE_BYTES - 1] |= 0x80;
        const uint64_t* p = (const uint64_t*)pad;
        state[0]^=p[0]; state[1]^=p[1]; state[2]^=p[2]; state[3]^=p[3];
        state[4]^=p[4]; state[5]^=p[5]; state[6]^=p[6]; state[7]^=p[7];
        state[8]^=p[8]; state[9]^=p[9]; state[10]^=p[10]; state[11]^=p[11];
        state[12]^=p[12]; state[13]^=p[13]; state[14]^=p[14]; state[15]^=p[15];
        state[16]^=p[16];
        keccak_f1600(state);

        munmap(mapped, file_size);
    }

    // Extract hash and convert to hex
    uint8_t hash[32];
    memcpy(hash, state, 32);

    static const char hx[] = "0123456789abcdef";
    char hex[64];
    for (int i = 0; i < 32; ++i) {
        hex[i*2]   = hx[(hash[i]>>4)&0xF];
        hex[i*2+1] = hx[hash[i]&0xF];
    }

    (void)!write(STDOUT_FILENO, hex, 64);
    return 0;
}
// EVOLVE-BLOCK-END