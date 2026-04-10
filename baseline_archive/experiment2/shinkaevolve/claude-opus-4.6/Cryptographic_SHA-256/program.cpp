// EVOLVE-BLOCK-START
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("sha,sse4.1,sse4.2,ssse3")

#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <immintrin.h>
#include <cpuid.h>

static const uint32_t K[64] __attribute__((aligned(16))) = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

/* --- SHA-NI accelerated transform --- */
__attribute__((target("sha,sse4.1,ssse3"),hot))
static void sha256_ni(uint32_t state[8], const uint8_t *data, size_t nblocks) {
    const __m128i SHUF = _mm_set_epi64x(
        0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);

    __m128i tmp  = _mm_loadu_si128((const __m128i*)&state[0]);
    __m128i st1  = _mm_loadu_si128((const __m128i*)&state[4]);
    __m128i t    = _mm_shuffle_epi32(tmp, 0xB1);
    st1          = _mm_shuffle_epi32(st1, 0x1B);
    __m128i st0  = _mm_alignr_epi8(t, st1, 8);
    st1          = _mm_blend_epi16(st1, t, 0xF0);

    for (size_t b = 0; b < nblocks; b++, data += 64) {
        __m128i as = st1, cs = st0;
        __m128i m0, m1, m2, m3, m;

        m0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(data+ 0)), SHUF);
        m1 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(data+16)), SHUF);
        m2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(data+32)), SHUF);
        m3 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(data+48)), SHUF);

        /* Rounds 0-3 */
        m=_mm_add_epi32(m0,_mm_load_si128((const __m128i*)&K[0]));
        st1=_mm_sha256rnds2_epu32(st1,st0,m);
        m=_mm_shuffle_epi32(m,0x0E);
        st0=_mm_sha256rnds2_epu32(st0,st1,m);

        /* Rounds 4-7 */
        m=_mm_add_epi32(m1,_mm_load_si128((const __m128i*)&K[4]));
        st1=_mm_sha256rnds2_epu32(st1,st0,m);
        m=_mm_shuffle_epi32(m,0x0E);
        st0=_mm_sha256rnds2_epu32(st0,st1,m);
        m0=_mm_sha256msg1_epu32(m0,m1);

        /* Rounds 8-11 */
        m=_mm_add_epi32(m2,_mm_load_si128((const __m128i*)&K[8]));
        st1=_mm_sha256rnds2_epu32(st1,st0,m);
        m=_mm_shuffle_epi32(m,0x0E);
        st0=_mm_sha256rnds2_epu32(st0,st1,m);
        m1=_mm_sha256msg1_epu32(m1,m2);

        /* Rounds 12-15 */
        m=_mm_add_epi32(m3,_mm_load_si128((const __m128i*)&K[12]));
        st1=_mm_sha256rnds2_epu32(st1,st0,m);
        tmp=_mm_alignr_epi8(m3,m2,4); m0=_mm_add_epi32(m0,tmp);
        m0=_mm_sha256msg2_epu32(m0,m3);
        m=_mm_shuffle_epi32(m,0x0E);
        st0=_mm_sha256rnds2_epu32(st0,st1,m);
        m2=_mm_sha256msg1_epu32(m2,m3);

#define FOUR_ROUNDS(r, a, b, c, d, do_msg1_on, do_msg1_arg) \
        m=_mm_add_epi32(a,_mm_load_si128((const __m128i*)&K[r])); \
        st1=_mm_sha256rnds2_epu32(st1,st0,m); \
        tmp=_mm_alignr_epi8(a,d,4); b=_mm_add_epi32(b,tmp); \
        b=_mm_sha256msg2_epu32(b,a); \
        m=_mm_shuffle_epi32(m,0x0E); \
        st0=_mm_sha256rnds2_epu32(st0,st1,m); \
        do_msg1_on=_mm_sha256msg1_epu32(do_msg1_on,do_msg1_arg);

        FOUR_ROUNDS(16, m0, m1, m2, m3, m3, m0)
        FOUR_ROUNDS(20, m1, m2, m3, m0, m0, m1)
        FOUR_ROUNDS(24, m2, m3, m0, m1, m1, m2)
        FOUR_ROUNDS(28, m3, m0, m1, m2, m2, m3)
        FOUR_ROUNDS(32, m0, m1, m2, m3, m3, m0)
        FOUR_ROUNDS(36, m1, m2, m3, m0, m0, m1)
        FOUR_ROUNDS(40, m2, m3, m0, m1, m1, m2)
        FOUR_ROUNDS(44, m3, m0, m1, m2, m2, m3)
#undef FOUR_ROUNDS

        /* Rounds 48-51 */
        m=_mm_add_epi32(m0,_mm_load_si128((const __m128i*)&K[48]));
        st1=_mm_sha256rnds2_epu32(st1,st0,m);
        tmp=_mm_alignr_epi8(m0,m3,4); m1=_mm_add_epi32(m1,tmp);
        m1=_mm_sha256msg2_epu32(m1,m0);
        m=_mm_shuffle_epi32(m,0x0E);
        st0=_mm_sha256rnds2_epu32(st0,st1,m);
        m3=_mm_sha256msg1_epu32(m3,m0);

        /* Rounds 52-55 */
        m=_mm_add_epi32(m1,_mm_load_si128((const __m128i*)&K[52]));
        st1=_mm_sha256rnds2_epu32(st1,st0,m);
        tmp=_mm_alignr_epi8(m1,m0,4); m2=_mm_add_epi32(m2,tmp);
        m2=_mm_sha256msg2_epu32(m2,m1);
        m=_mm_shuffle_epi32(m,0x0E);
        st0=_mm_sha256rnds2_epu32(st0,st1,m);

        /* Rounds 56-59 */
        m=_mm_add_epi32(m2,_mm_load_si128((const __m128i*)&K[56]));
        st1=_mm_sha256rnds2_epu32(st1,st0,m);
        tmp=_mm_alignr_epi8(m2,m1,4); m3=_mm_add_epi32(m3,tmp);
        m3=_mm_sha256msg2_epu32(m3,m2);
        m=_mm_shuffle_epi32(m,0x0E);
        st0=_mm_sha256rnds2_epu32(st0,st1,m);

        /* Rounds 60-63 */
        m=_mm_add_epi32(m3,_mm_load_si128((const __m128i*)&K[60]));
        st1=_mm_sha256rnds2_epu32(st1,st0,m);
        m=_mm_shuffle_epi32(m,0x0E);
        st0=_mm_sha256rnds2_epu32(st0,st1,m);

        st0=_mm_add_epi32(st0,cs);
        st1=_mm_add_epi32(st1,as);
    }

    /* Convert back */
    __m128i t2 = _mm_shuffle_epi32(st0, 0x1B);
    __m128i t3 = _mm_shuffle_epi32(st1, 0xB1);
    _mm_storeu_si128((__m128i*)&state[0], _mm_blend_epi16(t3,t2,0x0F));
    _mm_storeu_si128((__m128i*)&state[4], _mm_alignr_epi8(t3,t2,8));
}

/* --- Software fallback --- */
__attribute__((hot))
static void sha256_sw(uint32_t state[8], const uint8_t *block) {
    uint32_t W[64], a,b,c,d,e,f,g,h;
    for(int i=0;i<16;i++){
        const uint8_t*p=block+i*4;
        W[i]=((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|p[3];
    }
    for(int i=16;i<64;i++){
        uint32_t w15=W[i-15],w2=W[i-2];
        W[i]=W[i-16]+(((w15>>7)|(w15<<25))^((w15>>18)|(w15<<14))^(w15>>3))
             +W[i-7]+(((w2>>17)|(w2<<15))^((w2>>19)|(w2<<13))^(w2>>10));
    }
    a=state[0];b=state[1];c=state[2];d=state[3];
    e=state[4];f=state[5];g=state[6];h=state[7];
#define R(i) do{ \
    uint32_t t1=h+(((e>>6)|(e<<26))^((e>>11)|(e<<21))^((e>>25)|(e<<7)))+((e&f)^(~e&g))+K[i]+W[i]; \
    uint32_t t2=(((a>>2)|(a<<30))^((a>>13)|(a<<19))^((a>>22)|(a<<10)))+((a&b)^(a&c)^(b&c)); \
    h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; \
}while(0)
    R(0);R(1);R(2);R(3);R(4);R(5);R(6);R(7);
    R(8);R(9);R(10);R(11);R(12);R(13);R(14);R(15);
    R(16);R(17);R(18);R(19);R(20);R(21);R(22);R(23);
    R(24);R(25);R(26);R(27);R(28);R(29);R(30);R(31);
    R(32);R(33);R(34);R(35);R(36);R(37);R(38);R(39);
    R(40);R(41);R(42);R(43);R(44);R(45);R(46);R(47);
    R(48);R(49);R(50);R(51);R(52);R(53);R(54);R(55);
    R(56);R(57);R(58);R(59);R(60);R(61);R(62);R(63);
#undef R
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

int main(){
    /* Use stack for small inputs, heap for large */
    uint8_t stackbuf[65536];
    uint8_t *inbuf = stackbuf;
    size_t capacity = sizeof(stackbuf);
    size_t total = 0;

    for(;;){
        if(__builtin_expect(total == capacity, 0)){
            size_t newcap = capacity * 4;
            uint8_t *nb = (uint8_t*)malloc(newcap);
            memcpy(nb, inbuf, total);
            if(inbuf != stackbuf) free(inbuf);
            inbuf = nb;
            capacity = newcap;
        }
        ssize_t n = read(0, inbuf + total, capacity - total);
        if(n <= 0) break;
        total += n;
    }

    uint32_t st[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};

    const bool use_ni = __builtin_cpu_supports("sha");
    const uint8_t*ptr=inbuf;
    size_t rem=total;
    if(rem>=64){
        size_t nb=rem/64;
        if(use_ni) sha256_ni(st,ptr,nb);
        else for(size_t i=0;i<nb;i++) sha256_sw(st,ptr+i*64);
        ptr+=nb*64; rem-=nb*64;
    }

    /* Padding */
    uint8_t pad[128];
    memcpy(pad,ptr,rem);
    size_t i=rem;
    pad[i++]=0x80;
    size_t padlen=(rem<56)?64:128;
    memset(pad+i,0,padlen-i);
    uint64_t bits=(uint64_t)total*8;
    pad[padlen-8]=(uint8_t)(bits>>56);
    pad[padlen-7]=(uint8_t)(bits>>48);
    pad[padlen-6]=(uint8_t)(bits>>40);
    pad[padlen-5]=(uint8_t)(bits>>32);
    pad[padlen-4]=(uint8_t)(bits>>24);
    pad[padlen-3]=(uint8_t)(bits>>16);
    pad[padlen-2]=(uint8_t)(bits>>8);
    pad[padlen-1]=(uint8_t)(bits);

    if(use_ni) sha256_ni(st,pad,padlen/64);
    else for(size_t b=0;b<padlen/64;b++) sha256_sw(st,pad+b*64);

    if(inbuf != stackbuf) free(inbuf);

    static const char hx[]="0123456789abcdef";
    char hex[65];
    for(int j=0;j<8;j++){
        uint32_t s=st[j];
        hex[j*8+0]=hx[(s>>28)&0xF]; hex[j*8+1]=hx[(s>>24)&0xF];
        hex[j*8+2]=hx[(s>>20)&0xF]; hex[j*8+3]=hx[(s>>16)&0xF];
        hex[j*8+4]=hx[(s>>12)&0xF]; hex[j*8+5]=hx[(s>>8)&0xF];
        hex[j*8+6]=hx[(s>>4)&0xF];  hex[j*8+7]=hx[s&0xF];
    }
    hex[64]=0;
    write(1,hex,64);
    return 0;
}
// EVOLVE-BLOCK-END