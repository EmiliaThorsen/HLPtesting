#ifndef VECTOR_TOOLS_H
#define VECTOR_TOOLS_H
#include <stdint.h>
#include <immintrin.h>

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

extern const uint64_t identity_permutation_big_endian64;
extern const uint64_t identity_permutation_packed64;
extern const uint64_t identity_permutation_little_endian64;

extern const __m128i reverse_permutation128;
extern const __m256i reverse_permutation256;

extern const __m128i identity_permutation128;
extern const __m256i identity_permutation256;

extern const __m256i split_test_mask256;

__m128i unpack_uint_to_xmm(uint64_t uint);
uint64_t pack_xmm_to_uint(__m128i xmm);

__m128i little_endian_uint_to_xmm(uint64_t uint);
uint64_t little_endian_xmm_to_uint(__m128i xmm);

__m128i big_endian_uint_to_xmm(uint64_t uint);
uint64_t big_endian_xmm_to_uint(__m128i xmm);

ymm_pair_t quad_unpack_map256(__m256i packed);
__m256i quad_pack_map256(ymm_pair_t unpacked);

uint64_t apply_mapping_packed64(uint64_t first, uint64_t second);

#endif
