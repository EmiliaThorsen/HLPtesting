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

const __m256i SHUFB_REV_16x2x8  = XMM2_SHUFB(  1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 10, 13, 12, 15, 14);
const __m256i SHUFB_REV_8x4x8   = XMM2_SHUFB(  3,  2,  1,  0,  7,  6,  5,  4, 11, 10,  9,  8, 15, 14, 13, 12);
const __m256i SHUFB_REV_4x8x8   = XMM2_SHUFB(  7,  6,  5,  4,  3,  2,  1,  0, 15, 14, 13, 12, 11, 10,  9,  8);
const __m256i SHUFB_REV_2x16x8  = XMM2_SHUFB( 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0);
const __m256i SHUFB_REV_8x2x16  = XMM2_SHUFB(  2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9, 14, 15, 12, 13);
const __m256i SHUFB_REV_4x4x16  = XMM2_SHUFB(  6,  7,  4,  5,  2,  3,  0,  1, 14, 15, 12, 13, 10, 11,  8,  9);
const __m256i SHUFB_REV_2x8x16  = XMM2_SHUFB( 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  1);

