#include "vector_tools.h"

const uint64_t low_halves_mask64 = 0x0f0f0f0f0f0f0f0f;
const __m128i low_halves_mask128 = {low_halves_mask64, low_halves_mask64};
const __m256i low_halves_mask256 = {low_halves_mask64, low_halves_mask64, low_halves_mask64, low_halves_mask64};

const uint64_t high_halves_mask64 = 0xf0f0f0f0f0f0f0f0;
const __m128i high_halves_mask128 = {high_halves_mask64, high_halves_mask64};
const __m256i high_halves_mask256 = {high_halves_mask64, high_halves_mask64, high_halves_mask64, high_halves_mask64};

const uint64_t low_bytes_mask64 = 0x00ff00ff00ff00ff;
const __m128i low_bytes_mask128 = {low_bytes_mask64, low_bytes_mask64};
const __m256i low_bytes_mask256 = {low_bytes_mask64, low_bytes_mask64, low_bytes_mask64, low_bytes_mask64};

const uint64_t high_bytes_mask64 = 0xff00ff00ff00ff00;
const __m128i high_bytes_mask128 = {high_bytes_mask64, high_bytes_mask64};
const __m256i high_bytes_mask256 = {high_bytes_mask64, high_bytes_mask64, high_bytes_mask64, high_bytes_mask64};

const __m128i uint_max128 = {-1, -1};
const __m256i uint_max256 = {-1, -1, -1, -1};

const uint64_t low_15_bytes_mask64 = UINT64_MAX >> 8;
const __m128i low_15_bytes_mask128 = { low_15_bytes_mask64, low_15_bytes_mask64 };
const __m256i low_15_bytes_mask256 = { low_15_bytes_mask64, low_15_bytes_mask64, low_15_bytes_mask64, low_15_bytes_mask64 };

const uint64_t identity_permutation_big_endian64 = 0x0123456789abcdef;
const uint64_t identity_permutation_packed64 = 0x7f6e5d4c3b2a1908;
const uint64_t identity_permutation_little_endian64 = 0xfedcba9876543210;

const uint64_t identity_perm_lo = 0x0706050403020100;
const uint64_t identity_perm_hi =  0x0f0e0d0c0b0a0908;
const __m128i identity_permutation128 = {identity_perm_lo, identity_perm_hi};
const __m256i identity_permutation256 = {identity_perm_lo, identity_perm_hi, identity_perm_lo, identity_perm_hi};

const uint64_t reverse_perm_lo = 0x08090a0b0c0d0e0f;
const uint64_t reverse_perm_hi =  0x0001020304050607;
const __m128i reverse_permutation128 = {reverse_perm_lo, reverse_perm_hi};
const __m256i reverse_permutation256 = {reverse_perm_lo, reverse_perm_hi, reverse_perm_lo, reverse_perm_hi};

const __m256i split_test_mask256 = {-1, -1, 0, 0};

__m128i unpack_uint_to_xmm(uint64_t uint) {
    __m128i input = _mm_cvtsi64_si128(uint);
    return _mm_and_si128(_mm_or_si128(_mm_slli_si128(input, 8), _mm_srli_epi64(input, 4)), low_halves_mask128);
}

uint64_t pack_xmm_to_uint(__m128i xmm) {
    return _mm_cvtsi128_si64(_mm_or_si128(_mm_slli_epi32(xmm, 4), _mm_srli_si128(xmm, 8)));
}

__m128i little_endian_uint_to_xmm(uint64_t uint) {
    __m128i input = _mm_cvtsi64_si128(uint);
    return _mm_unpacklo_epi8(
            _mm_and_si128(input, low_halves_mask128),
            _mm_and_si128(_mm_srli_epi64(input, 4), low_halves_mask128)
            );
}

uint64_t little_endian_xmm_to_uint(__m128i xmm) {
    __m128i merged = _mm_or_si128(_mm_and_si128(xmm, low_bytes_mask128), _mm_srli_epi16(xmm, 4));
    return _mm_cvtsi128_si64(_mm_packus_epi16(merged, _mm_setzero_si128()));
}

__m128i big_endian_uint_to_xmm(uint64_t uint) {
    return _mm_shuffle_epi8(little_endian_uint_to_xmm(uint), reverse_permutation128);
}

uint64_t big_endian_xmm_to_uint(__m128i xmm) {
    return little_endian_xmm_to_uint(_mm_shuffle_epi8(xmm, reverse_permutation128));
}

ymm_pair_t quad_unpack_map256(__m256i packed) {
    const __m256i upackShifts = {4, 0, 4, 0};
    ymm_pair_t pair = {
        _mm256_and_si256(_mm256_srlv_epi64(_mm256_shuffle_epi32(packed, 0x44), upackShifts), low_halves_mask256),
        _mm256_and_si256(_mm256_srlv_epi64(_mm256_shuffle_epi32(packed, 0xee), upackShifts), low_halves_mask256)};
    return pair;
}

__m256i quad_pack_map256(ymm_pair_t unpacked) {
    return _mm256_blend_epi32(
            _mm256_or_si256(_mm256_srli_si256(unpacked.ymm0, 8), _mm256_slli_epi64(unpacked.ymm0, 4)),
            _mm256_or_si256(unpacked.ymm1, _mm256_slli_epi64(_mm256_slli_si256(unpacked.ymm1, 8), 4)),
            0b11001100
            );
}

uint64_t apply_mapping_packed64(uint64_t first, uint64_t second) {
    return pack_xmm_to_uint(_mm_shuffle_epi8(unpack_uint_to_xmm(second), unpack_uint_to_xmm(first)));
}

//counts uniqe values, usefull for generalizable prune of layers that reduce too much from the get go
int get_group64(uint64_t x) {
    uint16_t bit_feild = 0;
    for(int i = 0; i < 16; i++) {
        int ss = (x >> (i * 4)) & 15;
        bit_feild |= 1 << ss;
    }
    return _popcnt32(bit_feild);
}


