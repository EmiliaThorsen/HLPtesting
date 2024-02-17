#include "bitonicSort.h"

const uint64_t ymmByteConstants[] = {
    0x0607040502030001, 0x0e0f0c0d0a0b0809, // bitonicSortByteSwap
    0x09080b0a0d0c0f0e, 0x0100030205040706, // bitonicSortWordReverse1x8
    0x0100030205040706, 0x09080b0a0d0c0f0e, // bitonicSortWordReverse4x2
    0x0504070601000302, 0x0d0c0f0e09080b0a, // bitonicSortWordReverse2x4
    0x0b030a0209010800, 0x0f070e060d050c04, // bitonicSortInitialZip
    0x030b020a01090008, 0x070f060e050d040c  // bitonicSortInitialZip2
};

const __m256i bitonicSortByteSwap = {ymmByteConstants[0], ymmByteConstants[1], ymmByteConstants[0], ymmByteConstants[1]};
const __m256i bitonicSortWordReverse1x8 = {ymmByteConstants[2], ymmByteConstants[3], ymmByteConstants[2], ymmByteConstants[3]};
const __m256i bitonicSortWordReverse2x4 = {ymmByteConstants[4], ymmByteConstants[5], ymmByteConstants[4], ymmByteConstants[5]};
const __m256i bitonicSortWordReverse4x2 = {ymmByteConstants[6], ymmByteConstants[7], ymmByteConstants[6], ymmByteConstants[7]};
const __m256i bitonicSortInitialZip = {ymmByteConstants[8], ymmByteConstants[9], ymmByteConstants[8], ymmByteConstants[9]};
const __m256i bitonicSortInitialZip2 = {ymmByteConstants[10], ymmByteConstants[11], ymmByteConstants[10], ymmByteConstants[11]};

const uint64_t lowBytesMask64 = 0x00ff00ff00ff00ff;
const __m128i lowBytesMask128 = {lowBytesMask64, lowBytesMask64};
const __m256i lowBytesMask256 = {lowBytesMask64, lowBytesMask64, lowBytesMask64, lowBytesMask64};

void bitonic_sort4x16x8(uint8_t* arrays) {
    ymm_pair_t pair = {
        _mm256_loadu_si256((__m256i*) arrays),
        _mm256_loadu_si256(((__m256i*) arrays) + 1)
    };
    pair = bitonic_sort4x16x8_inner(pair);
    _mm256_storeu_si256((__m256i*) arrays, pair.ymm0);
    _mm256_storeu_si256(((__m256i*) arrays) + 1, pair.ymm1);
}

