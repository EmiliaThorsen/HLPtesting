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
#define XMM_SHUFB(...) ((const __m128i) {XMM_SHUFB_INNER(__VA_ARGS__)})
#define XMM2_SHUFB(...) ((const __m256i) {XMM_SHUFB_INNER(__VA_ARGS__), XMM_SHUFB_INNER(__VA_ARGS__)})

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

#define uint_max128 ((__m128i) {-1, -1})
#define uint_max256 ((__m256i) {-1, -1, -1, -1})

extern const uint64_t low_15_bytes_mask64;
#define low_15_bytes_mask128 _mm_srli_si128(uint_max128, 1)
#define low_15_bytes_mask256 _mm256_srli_si256(uint_max256, 1)

extern const uint64_t identity_permutation_big_endian64;
extern const uint64_t identity_permutation_packed64;
extern const uint64_t identity_permutation_little_endian64;

extern const __m128i reverse_permutation128;
extern const __m256i reverse_permutation256;

extern const __m128i identity_permutation128;
extern const __m256i identity_permutation256;

#define split_test_mask256 _mm256_castsi128_si256(uint_max128)

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


static __m128i unpack_uint_to_xmm(uint64_t uint) {
    __m128i input = _mm_cvtsi64_si128(uint);
    return _mm_and_si128(_mm_or_si128(_mm_slli_si128(input, 8), _mm_srli_epi64(input, 4)), low_halves_mask128);
}

static uint64_t pack_xmm_to_uint(__m128i xmm) {
    return _mm_cvtsi128_si64(_mm_or_si128(_mm_slli_epi32(xmm, 4), _mm_srli_si128(xmm, 8)));
}

static __m128i little_endian_uint_to_xmm(uint64_t uint) {
    __m128i input = _mm_cvtsi64_si128(uint);
    return _mm_unpacklo_epi8(
            _mm_and_si128(input, low_halves_mask128),
            _mm_and_si128(_mm_srli_epi64(input, 4), low_halves_mask128)
            );
}

static uint64_t little_endian_xmm_to_uint(__m128i xmm) {
    __m128i merged = _mm_or_si128(_mm_and_si128(xmm, low_bytes_mask128), _mm_srli_epi16(xmm, 4));
    return _mm_cvtsi128_si64(_mm_packus_epi16(merged, _mm_setzero_si128()));
}

static __m128i big_endian_uint_to_xmm(uint64_t uint) {
    return _mm_shuffle_epi8(little_endian_uint_to_xmm(uint), reverse_permutation128);
}

static uint64_t big_endian_xmm_to_uint(__m128i xmm) {
    return little_endian_xmm_to_uint(_mm_shuffle_epi8(xmm, reverse_permutation128));
}

static ymm_pair_t quad_unpack_map256(__m256i packed) {
    const __m256i upackShifts = {4, 0, 4, 0};
    ymm_pair_t pair = {
        _mm256_and_si256(_mm256_srlv_epi64(_mm256_shuffle_epi32(packed, 0x44), upackShifts), low_halves_mask256),
        _mm256_and_si256(_mm256_srlv_epi64(_mm256_shuffle_epi32(packed, 0xee), upackShifts), low_halves_mask256)};
    return pair;
}

static __m256i quad_pack_map256(ymm_pair_t unpacked) {
    return _mm256_blend_epi32(
            _mm256_or_si256(_mm256_srli_si256(unpacked.ymm0, 8), _mm256_slli_epi64(unpacked.ymm0, 4)),
            _mm256_or_si256(unpacked.ymm1, _mm256_slli_epi64(_mm256_slli_si256(unpacked.ymm1, 8), 4)),
            0b11001100
            );
}

static uint64_t apply_mapping_packed64(uint64_t first, uint64_t second) {
    return pack_xmm_to_uint(_mm_shuffle_epi8(unpack_uint_to_xmm(second), unpack_uint_to_xmm(first)));
}

//counts uniqe values, usefull for generalizable prune of layers that reduce too much from the get go
static int get_group64(uint64_t x) {
    uint16_t bit_feild = 0;
    for(int i = 0; i < 16; i++) {
        int ss = (x >> (i * 4)) & 15;
        bit_feild |= 1 << ss;
    }
    return _popcnt32(bit_feild);
}

#endif
