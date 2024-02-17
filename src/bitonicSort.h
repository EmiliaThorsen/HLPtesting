#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H
#include <stdint.h>
#include <immintrin.h>

typedef struct ymm_pair_s {
    __m256i ymm0;
    __m256i ymm1;
} ymm_pair_t;


extern const __m256i bitonicSortByteSwap, bitonicSortWordReverse1x8, bitonicSortWordReverse2x4, bitonicSortWordReverse4x2, bitonicSortInitialZip, bitonicSortInitialZip2, lowBytesMask256;

inline ymm_pair_t bitonic_sort4x16x8_blend_w(ymm_pair_t pair) {
    ymm_pair_t newpair = {_mm256_blend_epi16(pair.ymm0, pair.ymm1, 0b10101010),
        _mm256_blend_epi16(pair.ymm1, pair.ymm0, 0b10101010)};
    return newpair;
}

inline ymm_pair_t bitonic_sort4x16x8_blend(ymm_pair_t pair, const uint8_t blendConstant) {
    if (!blendConstant) return pair;
    const uint8_t actualBlendConstant = (blendConstant & 0xf) | ((blendConstant & 0xf) << 4);

    ymm_pair_t newpair = {_mm256_blend_epi32(pair.ymm0, pair.ymm1, actualBlendConstant),
        _mm256_blend_epi32(pair.ymm1, pair.ymm0, actualBlendConstant)};
    return newpair;
}

inline ymm_pair_t bitonic_sort4x16x8_minmax(ymm_pair_t pair) {
    ymm_pair_t newpair = {_mm256_min_epu8(pair.ymm0, pair.ymm1),
        _mm256_max_epu8(pair.ymm1, pair.ymm0)};
    return newpair;
}

inline ymm_pair_t bitonic_sort4x16x8_step(ymm_pair_t pair, const uint8_t blendConstant, const uint8_t shufConstant) {
    pair = bitonic_sort4x16x8_blend(pair, blendConstant);
    pair.ymm1 = _mm256_shuffle_epi32(pair.ymm1, shufConstant);
    return bitonic_sort4x16x8_minmax(pair);
}

inline ymm_pair_t bitonic_sort4x16x8_inner(ymm_pair_t pair) {
    const uint8_t shufd_identity = 0b11100100;
    const uint8_t shufd_qswap = 0b01001110;
    const uint8_t shufd_dswap = 0b10110001;
    const uint8_t shufd_dreverse = 0b00011011;

    // zip together
    __m256i tmp = _mm256_unpacklo_epi8(pair.ymm1, pair.ymm0);
    pair.ymm1 = _mm256_unpackhi_epi8(pair.ymm1, pair.ymm0);
    pair.ymm0 = tmp;

    // 2
    pair = bitonic_sort4x16x8_minmax(pair);

    // 4
    pair = bitonic_sort4x16x8_step(pair, 0, shufd_dswap);
    pair = bitonic_sort4x16x8_step(pair, 0b1010, shufd_dswap);

    // 8
    pair = bitonic_sort4x16x8_step(pair, 0, shufd_dreverse);
    pair = bitonic_sort4x16x8_step(pair, 0b1010, shufd_dswap);
    pair = bitonic_sort4x16x8_step(pair, 0b1100, shufd_qswap);

    // 16
    pair.ymm1 = _mm256_shuffle_epi8(pair.ymm1, bitonicSortWordReverse1x8);
    pair = bitonic_sort4x16x8_minmax(pair);
    pair = bitonic_sort4x16x8_blend_w(pair);
    pair.ymm1 = _mm256_shuffle_epi8(pair.ymm1, bitonicSortWordReverse2x4);

    pair = bitonic_sort4x16x8_step(pair, 0b1010, shufd_dswap);
    pair = bitonic_sort4x16x8_step(pair, 0b1100, shufd_qswap);
    pair = bitonic_sort4x16x8_step(pair, 0b1010, shufd_dswap);

    // pack back together
    // interlace words
    pair.ymm1 = _mm256_shuffle_epi8(pair.ymm1, bitonicSortWordReverse4x2);
    pair = bitonic_sort4x16x8_blend_w(pair);
    pair.ymm1 = _mm256_shuffle_epi8(pair.ymm1, bitonicSortWordReverse4x2);

    // pack bytes
    ymm_pair_t newpair = {_mm256_packus_epi16(_mm256_srli_epi16(pair.ymm0, 8), _mm256_srli_epi16(pair.ymm1, 8)),
        _mm256_packus_epi16(_mm256_and_si256(pair.ymm0, lowBytesMask256), _mm256_and_si256(pair.ymm1, lowBytesMask256))};
    return newpair;
}


extern void bitonic_sort4x16x8(uint8_t* arrays);

#endif
