#ifndef VECTOR_TOOLS_H
#define VECTOR_TOOLS_H
#include <stdint.h>
#include <immintrin.h>

#define CAT_CONST(a, b, w) (a & ((1 << w) - 1)) | ((b & ((1 << w) - 1)) << w)

#define CAT_CONST_8B(x0,x1,x2,x3,x4,x5,x6,x7)\
    (uint64_t) (x0 & 0xff) |\
    ((uint64_t) (x1 &0xff) << 8) |\
    ((uint64_t) (x2 &0xff) << 16) |\
    ((uint64_t) (x3 &0xff) << 24) |\
    ((uint64_t) (x4 &0xff) << 32) |\
    ((uint64_t) (x5 &0xff) << 40) |\
    ((uint64_t) (x6 &0xff) << 48) |\
    ((uint64_t) (x7 &0xff) << 56)

#define DOUBLE_XMM_TO_YMM(a,b) a, b, a, b
#define XMM_SHUFB_INNER(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15) CAT_CONST_8B(x0,x1,x2,x3,x4,x5,x6,x7), CAT_CONST_8B(x8,x9,x10,x11,x12,x13,x14,x15)
#define XMM_SHUFB(...) (const __m128i) {XMM_SHUFB_INNER(__VA_ARGS__)}
#define XMM2_SHUFB(...) (const __m256i) {XMM_SHUFB_INNER(__VA_ARGS__), XMM_SHUFB_INNER(__VA_ARGS__)}

typedef struct ymm_pair_s {
    __m256i ymm0;
    __m256i ymm1;
} ymm_pair_t;

extern const uint64_t low_halves_mask64;
extern const __m128i low_halves_mask128;
extern const __m256i low_halves_mask256;

extern const uint64_t high_halves_mask64;
extern const __m128i high_halves_mask128;
extern const __m256i high_halves_mask256;

extern const uint64_t low_bytes_mask64;
extern const __m128i low_bytes_mask128;
extern const __m256i low_bytes_mask256;

extern const uint64_t high_bytes_mask64;
extern const __m128i high_bytes_mask128;
extern const __m256i high_bytes_mask256;

extern const __m128i uint_max128;
extern const __m256i uint_max256;

extern const uint64_t low_15_bytes_mask64;
extern const __m128i low_15_bytes_mask128;
extern const __m256i low_15_bytes_mask256;

extern const uint64_t identity_permutation_big_endian64;
extern const uint64_t identity_permutation_packed64;
extern const uint64_t identity_permutation_little_endian64;

extern const __m128i reverse_permutation128;
extern const __m256i reverse_permutation256;

extern const __m128i identity_permutation128;
extern const __m256i identity_permutation256;

extern const __m256i split_test_mask256;

extern const __m256i SHUFB_REV_16x2x8;
extern const __m256i SHUFB_REV_8x4x8;
extern const __m256i SHUFB_REV_4x8x8;
extern const __m256i SHUFB_REV_2x16x8;
extern const __m256i SHUFB_REV_8x2x16;
extern const __m256i SHUFB_REV_4x4x16;
extern const __m256i SHUFB_REV_2x8x16;

#define SHUFD_REV_4x2x32 0b10110001
#define SHUFD_REV_2x4x32 0b00011011
#define SHUFD_REV_2x2x64 0b01001110

#define PERMQ_REV_1x4x64 SHUFD_REV_2x4x32
#define PERMQ_REV_1x2x128 SHUFD_REV_2x2x64

__m128i unpack_uint_to_xmm(uint64_t uint);
uint64_t pack_xmm_to_uint(__m128i xmm);

__m128i little_endian_uint_to_xmm(uint64_t uint);
uint64_t little_endian_xmm_to_uint(__m128i xmm);

__m128i big_endian_uint_to_xmm(uint64_t uint);
uint64_t big_endian_xmm_to_uint(__m128i xmm);

ymm_pair_t quad_unpack_map256(__m256i packed);
__m256i quad_pack_map256(ymm_pair_t unpacked);

uint64_t apply_mapping_packed64(uint64_t first, uint64_t second);

int get_group64(uint64_t x);

#endif
