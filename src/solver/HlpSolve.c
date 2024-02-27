#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <immintrin.h>
#include "../aa_tree.h"
#include "HlpSolve.h"
#include <stdbool.h>
#include "../bitonicSort.h"
#include "../vector_tools.h"
#include "../redstone.h"

struct cache_entry {
    uint64_t map;
    uint32_t trial;
    uint8_t depth;
};

struct cache {
    struct cache_entry* array;
    uint64_t mask;
    uint32_t global_trial;
};

struct hlp_solve_globals {
    struct __config__ {
        __m256i goal_min, goal_max, dont_care_mask, dont_care_post_sort_perm;
        int solve_type, dont_care_count, current_bfs_depth, group, accuracy;
    } config;

    struct __output__ {
        uint16_t* chain;
        int chain_length;
        int solutions_found;
    } output;

    struct __stats__ {
        long total_iterations;
        long cache_same_depth_hits, cache_dif_layer_hits, cache_misses, cache_bucket_util, cache_total_checks;
        clock_t start_time;
    } stats;

    struct cache cache;
};

int cacheSize = 22;

int hlpSolveVerbosity = 1;

int globalMaxDepth;
int globalAccuracy;

int isHex(char c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
}

int toHex(char c) {
    return c - (c <= '9' ? '0' : c <= 'F' ? 'A' - 10 : 'a' - 10);
}

int mapPairContainsRanges(uint64_t mins, uint64_t maxs) {
    while (mins && maxs) {
        int minVal = mins & 15;
        int maxVal = maxs & 15;
        if (minVal != minVal && !(minVal == 0 && maxVal == 15)) return 1;
        mins >>= 4;
        maxs >>= 4;
    }
    return 0;
}

hlp_request_t parseHlpRequestStr(char* str) {
    hlp_request_t result = {0};
    if (!str){
        result.error = HLP_ERROR_NULL;
        return result;
    };
    if (!*str){
        result.error = HLP_ERROR_BLANK;
        return result;
    };
    int length = 0;
    char* c = str;
    while (*c) {
        if (*(c + 1) == '-') {
            if (!isHex(*(c + 2))) {
                result.error = HLP_ERROR_MALFORMED;
                return result;
            }
            result.mins = (result.mins << 4) | toHex(*c);
            result.maxs = (result.maxs << 4) | toHex(*(c + 2));
            length++;
            c += 3;
            continue;
        }
        if (*c == '.' || *c == 'x' || *c == 'X') {
            result.mins = (result.mins << 4) | 0;
            result.maxs = (result.maxs << 4) | 15;
            length++;
            c++;
            continue;
        }
        if (*c == '[' || *c == ']') continue;
        if (isHex(*c)) {
            result.mins = (result.mins << 4) | toHex(*c);
            result.maxs = (result.maxs << 4) | toHex(*c);
            length++;
            c++;
            continue;
        }
        result.error = HLP_ERROR_MALFORMED;
        return result;
    }
    int remainingLength = 16 - length;
 
    if (remainingLength < 0) {
        result.error = HLP_ERROR_TOO_LONG;
        return result;
    }
    result.mins <<= remainingLength * 4;
    result.maxs <<= remainingLength * 4;
    result.maxs |= ((uint64_t) 1 << (remainingLength * 4)) - 1;

    if (result.mins == result.maxs)
        result.solveType = HLP_SOLVE_TYPE_EXACT;
    else if (mapPairContainsRanges(result.mins, result.maxs))
        result.solveType = HLP_SOLVE_TYPE_RANGED;
    else
        result.solveType = HLP_SOLVE_TYPE_PARTIAL;

    return result;
}

#define COMBINE_RANGES_INNER(shift, s1, s2)\
    mask = _mm256_cmpeq_epi8(equalityReference, _mm256_s##s1##li_si256(equalityReference, shift));\
    minsAndMaxs.ymm0 = _mm256_max_epu8(minsAndMaxs.ymm0, _mm256_s##s2##li_si256(_mm256_and_si256(minsAndMaxs.ymm0, mask), shift));\
    minsAndMaxs.ymm1 = _mm256_max_epu8(minsAndMaxs.ymm1, _mm256_s##s2##li_si256(_mm256_and_si256(minsAndMaxs.ymm1, mask), shift));\

ymm_pair_t combineRanges(__m256i equalityReference, ymm_pair_t minsAndMaxs) {
    // we invert the max values so that after shifting things, any zeros
    // shifted in will not affect anything, as we only combine things with max
    // function
    minsAndMaxs.ymm1 = _mm256_xor_si256(minsAndMaxs.ymm1, UINT256_MAX);
    __m256i mask;

    COMBINE_RANGES_INNER(1, r, l);
    COMBINE_RANGES_INNER(2, r, l);
    COMBINE_RANGES_INNER(4, r, l);
    COMBINE_RANGES_INNER(8, r, l);

    COMBINE_RANGES_INNER(1, l, r);
    COMBINE_RANGES_INNER(2, l, r);
    COMBINE_RANGES_INNER(4, l, r);
    COMBINE_RANGES_INNER(8, l, r);

    minsAndMaxs.ymm1 = _mm256_xor_si256(minsAndMaxs.ymm1, UINT256_MAX);
}

int getLegalDistCheckMaskRanged(struct hlp_solve_globals* globals, __m256i sortedYmm, int threshhold) {
    __m256i finalIndices = _mm256_and_si256(sortedYmm, LO_HALVES_4_256);
    __m256i current = _mm256_and_si256(_mm256_srli_epi64(sortedYmm, 4), LO_HALVES_4_256);
    ymm_pair_t final = {_mm256_shuffle_epi8(globals->config.goal_min, finalIndices), _mm256_shuffle_epi8(globals->config.goal_max, finalIndices)};
    final = combineRanges(current, final);

    // if the min is higher than the max, that's all we need to know
    __m256i illegals = _mm256_cmpgt_epi8(final.ymm0, final.ymm1);
    int mask = -1;
    mask &= _mm256_testz_si256(LO_HALVES_128_256, illegals) | (_mm256_testc_si256(LO_HALVES_128_256, illegals) << 2);

    __m256i finalDelta = _mm256_max_epi8(
            _mm256_sub_epi8(final.ymm0, _mm256_srli_si256(final.ymm1, 1)),
            _mm256_sub_epi8(_mm256_srli_si256(final.ymm0, 1), final.ymm1)
            );
    __m256i currentDelta = _mm256_abs_epi8(_mm256_sub_epi8(_mm256_srli_si256(current, 1), current));

    uint32_t separationsMask = _mm256_movemask_epi8(_mm256_cmpgt_epi8(finalDelta, currentDelta)) & 0x7fff7fff;
    mask &= (_popcnt32(separationsMask & 0xffff) <= threshhold) | ((_popcnt32(separationsMask >> 16) <= threshhold) << 2);
    return mask;
}

int getLegalDistCheckMaskPartial(struct hlp_solve_globals* globals, __m256i sortedYmm, int threshhold) {
    __m256i final = _mm256_and_si256(sortedYmm, LO_HALVES_4_256);
    __m256i current = _mm256_and_si256(_mm256_srli_epi64(sortedYmm, 4), LO_HALVES_4_256);

    __m256i finalDelta = _mm256_abs_epi8(_mm256_sub_epi8(_mm256_srli_si256(final, 1), final));
    __m256i currentDelta = _mm256_abs_epi8(_mm256_sub_epi8(_mm256_srli_si256(current, 1), current));

    __m256i illegals = _mm256_and_si256(_mm256_cmpeq_epi8(currentDelta, _mm256_setzero_si256()), finalDelta);
    illegals = _mm256_and_si256(illegals, LOW_15_BYTES_256);
    int mask = -1;
    mask &= _mm256_testz_si256(LO_HALVES_128_256, illegals) | (_mm256_testc_si256(LO_HALVES_128_256, illegals) << 2);

    uint32_t separationsMask = _mm256_movemask_epi8(_mm256_cmpgt_epi8(finalDelta, currentDelta)) & 0x7fff7fff;
    mask &= (_popcnt32(separationsMask & 0xffff) <= threshhold) | ((_popcnt32(separationsMask >> 16) <= threshhold) << 2);
    return mask;
}

int batchApplyAndCheckExact(
        struct hlp_solve_globals* globals,
        struct precomputed_hex_layer* layer,
        uint16_t* outputs,
        uint64_t input,
        int threshhold) {
    __m256i doubledInput = _mm256_permute4x64_epi64(_mm256_castsi128_si256(unpack_uint_to_xmm(input)), 0x44);

    // this contains extra bits to overwrite the current value on dont care entries
    __m256i doubledGoal;
    if (globals->config.solve_type == HLP_SOLVE_TYPE_RANGED) 
        doubledGoal = SHUFB_IDENTITY_256;
    else
        doubledGoal = _mm256_or_si256(globals->config.goal_min, globals->config.dont_care_mask);

    uint16_t* currentOutput = outputs;

    for (int i = (layer->next_layer_count - 1) / 4; i >= 0; i--) {
        ymm_pair_t quad = quad_unpack_map256(_mm256_loadu_si256(((__m256i*) layer->next_layer_luts) + i));
        quad.ymm0 = _mm256_shuffle_epi8(quad.ymm0, doubledInput);
        quad.ymm1 = _mm256_shuffle_epi8(quad.ymm1, doubledInput);

        ymm_pair_t sortedQuad = { _mm256_or_si256(doubledGoal, _mm256_slli_epi64(quad.ymm0, 4)),
            _mm256_or_si256(doubledGoal, _mm256_slli_epi64(quad.ymm1, 4)) };
        sortedQuad = bitonic_sort4x16x8_inner(sortedQuad);

        sortedQuad.ymm0 = _mm256_shuffle_epi8(sortedQuad.ymm0, globals->config.dont_care_post_sort_perm);
        sortedQuad.ymm1 = _mm256_shuffle_epi8(sortedQuad.ymm1, globals->config.dont_care_post_sort_perm);
        int mask;
        if (globals->config.solve_type == HLP_SOLVE_TYPE_RANGED)
            mask = getLegalDistCheckMaskRanged(globals, sortedQuad.ymm0, threshhold) | (getLegalDistCheckMaskRanged(globals, sortedQuad.ymm1, threshhold) << 1);
        else
            mask = getLegalDistCheckMaskPartial(globals, sortedQuad.ymm0, threshhold) | (getLegalDistCheckMaskPartial(globals, sortedQuad.ymm1, threshhold) << 1);
        if (i & (mask == 0)) continue;
        __m256i packed = quad_pack_map256(quad);

        for (int j = 3; j >= 0; j--) {
            *currentOutput = i * 4 + j;
            currentOutput += (mask >> j) & 1;
        }
    }

    return currentOutput - outputs;
}

int getMinGroup(uint64_t mins, uint64_t maxs) {
    // not great but works for now
    uint16_t bitFeild = 0;
    for(int i = 16; i; i--) {
        if ((mins & 15) == (maxs & 15))
            bitFeild |= 1 << (mins & 15);
        mins >>= 4;
        maxs >>= 4;
    }
    int result = _popcnt32(bitFeild);
    if (!result) return 1;
    return result;
}

//faster implementation of searching over the last layer while checking if you found the goal, unexpectedly big optimization
int fastLastLayerSearch(struct hlp_solve_globals* globals, uint64_t input, struct precomputed_hex_layer* layer) {
    __m256i doubledInput = _mm256_permute4x64_epi64(_mm256_castsi128_si256(unpack_uint_to_xmm(input)), 0x44);

    __m256i* quadMaps = (__m256i*) (layer->next_layer_luts);

    globals->stats.total_iterations += layer->next_layer_count;
    for (int i = (layer->next_layer_count - 1) / 4; i >= 0; i--) {
        ymm_pair_t quad = quad_unpack_map256(_mm256_loadu_si256(quadMaps + i));

        // determine if there are any spots that do not match up
        // if all zeros, that means they match
        // instead of split tests based on variant, the ranged test is only 2
        // cycles longer than the exact mode, so we just use ranged for
        // everything to avoid branches
        quad.ymm0 = _mm256_shuffle_epi8(quad.ymm0, doubledInput);
        quad.ymm1 = _mm256_shuffle_epi8(quad.ymm1, doubledInput);
        quad.ymm0 = _mm256_or_si256(_mm256_cmpgt_epi8(globals->config.goal_min, quad.ymm0), _mm256_cmpgt_epi8(quad.ymm0, globals->config.goal_max));
        quad.ymm1 = _mm256_or_si256(_mm256_cmpgt_epi8(globals->config.goal_min, quad.ymm1), _mm256_cmpgt_epi8(quad.ymm1, globals->config.goal_max));
        // no need to apply globals->config.dont_care_mask, they already will always succeed anyways
        if (i && _mm256_testnzc_si256(LO_HALVES_128_256, quad.ymm0) && _mm256_testnzc_si256(LO_HALVES_128_256, quad.ymm1)) continue;

        bool successes[] = {
            _mm256_testz_si256(LO_HALVES_128_256, quad.ymm0),
            _mm256_testz_si256(LO_HALVES_128_256, quad.ymm1),
            _mm256_testc_si256(LO_HALVES_128_256, quad.ymm0),
            _mm256_testc_si256(LO_HALVES_128_256, quad.ymm1)};

        for (int j=0; j<4; j++) {
            if (!successes[j]) continue;
            int index = i * 4 + j;
            globals->stats.total_iterations -= index;
            uint16_t config = layer->next_layers[index]->config;
            if (globals->output.solutions_found != -1) {
                globals->output.solutions_found++;
                continue;
            }
            globals->output.chain_length = globals->config.current_bfs_depth;
            if (globals->output.chain != 0) globals->output.chain[globals->config.current_bfs_depth - 1] = config;
            return 1;
        }
    }
    return 0;
}

int cacheCheck(struct hlp_solve_globals* globals, uint64_t output, int depth) {
    uint32_t pos = _mm_crc32_u32(_mm_crc32_u32(0, output & UINT32_MAX), output >> 32) & globals->cache.mask;
    struct cache_entry* entry = globals->cache.array + pos;
    globals->stats.cache_total_checks++;
    if (entry->map == output && entry->depth <= depth && entry->trial == globals->cache.global_trial) {
        if (entry->depth == depth) globals->stats.cache_same_depth_hits++;
        else globals->stats.cache_dif_layer_hits++;
        return 1;
    }

    if (entry->trial == globals->cache.global_trial && entry->map != output) globals->stats.cache_misses++;
    else globals->stats.cache_bucket_util++;

    entry->map = output;
    entry->depth = depth;
    entry->trial = globals->cache.global_trial;

    return 0;
}

void invalidateCache(struct cache* cache) {
    cache->global_trial++;
    // clear the cache if we somehow hit overflow
    if (!cache->global_trial) {
        for (int i = 0; i <= cache->mask; i++) {
            cache->array[i].map = 0;
            cache->array[i].depth = 0;
            cache->array[i].trial = 0;
        }
        // trial 0 should always mean blank
        cache->global_trial++;
    }
}

// the most number of separations that can be found in the distance check before it prunes
int getDistThreshold(struct hlp_solve_globals* globals, int remainingLayers) {
    if (globals->config.accuracy == ACCURACY_REDUCED) return remainingLayers - (remainingLayers > 2);
    // n is always sufficient anyways for 15-16 outputs
    if (globals->config.accuracy == ACCURACY_NORMAL || globals->config.group > 14) return remainingLayers;
    // n+1 is always sufficient for 14 outputs
    if (globals->config.accuracy == ACCURACY_INCREASED || globals->config.group > 13) return remainingLayers + 1;

    // currently the best known general threshhold
    // +/-1 is for round up division
    return ((remainingLayers * 3 - 1) >> 1) + 1;
}

/* test to see if this map falls under a solution
 */
int testMap(struct hlp_solve_globals* globals, uint64_t map) {
    __m128i xmm = unpack_uint_to_xmm(map);
    return _mm_testz_si128(_mm_or_si128(
                _mm_cmpgt_epi8(_mm256_castsi256_si128(globals->config.goal_min), xmm),
                _mm_cmpgt_epi8(xmm, _mm256_castsi256_si128(globals->config.goal_max))
                ), UINT128_MAX);
}


//main dfs recursive search function
int dfs(struct hlp_solve_globals* globals, uint64_t input, int depth, struct precomputed_hex_layer* layer, uint16_t* staged_branches) {
    // test to see if we found a solution, even if we're not at the end. this
    // can happen even though it seems like it shouldn't
    if (testMap(globals, input)) {
        globals->output.chain_length = depth + 1;
        if (globals->output.chain != 0) globals->output.chain[depth] = layer->config;
        return 1;
    }

    if(depth == globals->config.current_bfs_depth - 1) return fastLastLayerSearch(globals, input, layer);
    globals->stats.total_iterations += layer->next_layer_count;
    int totalNextLayersIdentified = batchApplyAndCheckExact(
            globals,
            layer,
            staged_branches,
            input,
            getDistThreshold(globals, globals->config.current_bfs_depth - depth - 1));

    for(int i = totalNextLayersIdentified - 1; i >= 0; i--) {
        struct precomputed_hex_layer* next_layer = layer->next_layers[staged_branches[i]];
        uint64_t output = apply_mapping_packed64(input, next_layer->map);

        //cache check
        if(cacheCheck(globals, output, depth)) continue;

        //call next layers
        if(dfs(globals, output, depth + 1, next_layer, staged_branches + layer->next_layer_count)) {
            if (globals->output.chain != 0) globals->output.chain[depth] = next_layer->config;
            return 1;
        }
        if (hlpSolveVerbosity < 3) continue;
        if(depth == 0 && globals->config.current_bfs_depth > 8) printf("done:%d/%d\n", totalNextLayersIdentified - i, totalNextLayersIdentified);
    }
    return 0;
}

static struct cache global_cache = {0};

int init(struct hlp_solve_globals* globals, hlp_request_t request) {
    globals->stats.start_time = clock();
    if (!globals->cache.array) {
        if (!global_cache.array) global_cache.array = calloc((1 << cacheSize), sizeof(struct cache_entry));
        globals->cache.array = global_cache.array;
        globals->cache.global_trial = global_cache.global_trial;
        globals->cache.mask = (1 << cacheSize) - 1;
    }
    globals->config.solve_type = request.solveType;
    globals->stats.total_iterations = 0;

    switch (globals->config.solve_type) {
        case HLP_SOLVE_TYPE_EXACT:
            globals->config.group = get_group64(request.mins);
            break;
        case HLP_SOLVE_TYPE_PARTIAL:
            globals->config.group = getMinGroup(request.mins, request.maxs);
            break;
        default:
            printf("you found a search mode that isn't implemented\n");
            return 1;
    }
    
    globals->config.goal_min = _mm256_permute4x64_epi64(_mm256_castsi128_si256(big_endian_uint_to_xmm(request.mins)), 0x44);
    globals->config.goal_max = _mm256_permute4x64_epi64(_mm256_castsi128_si256(big_endian_uint_to_xmm(request.maxs)), 0x44);

    globals->config.dont_care_mask = _mm256_cmpeq_epi8(_mm256_sub_epi8(globals->config.goal_max, globals->config.goal_min), LO_HALVES_4_256);
    globals->config.dont_care_count = _popcnt32(_mm_movemask_epi8(_mm256_castsi256_si128(globals->config.dont_care_mask)));
    globals->config.dont_care_post_sort_perm = _mm256_min_epi8(SHUFB_IDENTITY_256, _mm256_set1_epi8(15 - globals->config.dont_care_count));

    return 0;
}

//main search loop
int singleSearchInner(struct hlp_solve_globals* globals, struct precomputed_hex_layer* base_layer, int maxDepth) {
    globals->config.current_bfs_depth = 1;

    while (globals->config.current_bfs_depth <= maxDepth) {
        uint16_t* staged_branches = malloc(base_layer->next_layer_count * globals->config.current_bfs_depth * sizeof(uint16_t));
        int success = dfs(globals, IDENTITY_PERM_PK64, 0, base_layer, staged_branches);
        free(staged_branches);
        if (success) {
            if (hlpSolveVerbosity >= 3) {
                printf("solution found at %.2fms\n", (double)(clock() - globals->stats.start_time) / CLOCKS_PER_SEC * 1000);
                printf("total iter over all: %'ld\n", globals->stats.total_iterations);
                printf("cache checks: %'ld; same depth hits: %'ld; dif layer hits: %'ld; misses: %'ld; bucket utilization: %'ld\n",
                        globals->stats.cache_total_checks,
                        globals->stats.cache_same_depth_hits,
                        globals->stats.cache_dif_layer_hits,
                        globals->stats.cache_misses,
                        globals->stats.cache_bucket_util);
            }
            return globals->output.chain_length;
        }
        invalidateCache(&globals->cache);
        globals->config.current_bfs_depth++;

        if (hlpSolveVerbosity < 2) continue;
        printf("search over layer %d done\n",globals->config.current_bfs_depth - 1);

        if (hlpSolveVerbosity < 3) continue;
        printf("layer search done after %.2fms; %'ld iterations\n", (double)(clock() - globals->stats.start_time) / CLOCKS_PER_SEC * 1000, globals->stats.total_iterations);
    }
    if (hlpSolveVerbosity >= 2) {
        printf("failed to beat depth\n");
        printf("cache checks: %'ld; same depth hits: %'ld; dif layer hits: %'ld; misses: %'ld; bucket utilization: %'ld\n",
                globals->stats.cache_total_checks,
                globals->stats.cache_same_depth_hits,
                globals->stats.cache_dif_layer_hits,
                globals->stats.cache_misses,
                globals->stats.cache_bucket_util);
    }
    return maxDepth + 1;
}

/*
int singleSearch(hlp_request_t request, uint16_t* outputChain, int maxDepth, enum SearchAccuracy accuracy) {
    if (maxDepth < 0 || maxDepth > 31) maxDepth = 31;
    if (request.mins == 0) {
        if (outputChain) outputChain[0] = 0x2f0;
        return 1;
    }

    if (init(request)) return -1;

    globals->output.chain = outputChain;
    globals->output.solutions_found = -1;
    globals->config.accuracy = accuracy;
    struct precomputed_hex_layer* identity_layer = precompute_hex_layers(globals->config.group);
    return singleSearchInner(identity_layer, maxDepth);
}
*/

int solve(hlp_request_t request, uint16_t* outputChain, int maxDepth, enum SearchAccuracy accuracy) {
    struct hlp_solve_globals globals = {0};
    int requestedMaxDepth = maxDepth;
    if (maxDepth < 0 || maxDepth > 31) maxDepth = 31;

    if (init(&globals, request)) {
        printf("an error occurred\n");
        return requestedMaxDepth + 1;
    }

    struct precomputed_hex_layer* identity_layer = precompute_hex_layers(globals.config.group);
    /* return requestedMaxDepth + 1; */

    if (request.mins == 0) {
        if (outputChain) outputChain[0] = 0x2f0;
        return 1;
    }

    globals.output.chain = outputChain;
    globals.output.solutions_found = -1;
    int solutionLength = maxDepth;

    if (hlpSolveVerbosity >= 2) {
        if (accuracy > ACCURACY_REDUCED) printf("starting presearch\n");
        else printf("starting search\n");
    }

    // reduced accuracy search is sometimes faster than the others but
    // still often gets an optimal solution, so we start with that so the
    // "real" search can cut short if it doesn't find a better solution.
    // when it's not faster, the solution is found pretty fast anyways.
    globals.config.accuracy = ACCURACY_REDUCED;
    solutionLength = singleSearchInner(&globals, identity_layer, solutionLength);

    if (solutionLength == maxDepth) solutionLength = maxDepth;
    if (accuracy == ACCURACY_REDUCED) return solutionLength;
    long totalIter = globals.stats.total_iterations;
    globals.stats.total_iterations = 0;

    if (hlpSolveVerbosity >= 2) printf("starting main search\n");

    globals.config.accuracy = accuracy;
    int result = singleSearchInner(&globals, identity_layer, solutionLength - 1);
    if (hlpSolveVerbosity >= 2) printf("total iter across searches: %'ld\n", totalIter + globals.stats.total_iterations);
    if (result > maxDepth) return requestedMaxDepth + 1;
    return result;
}

void hlpSetCacheSize(int size) {
    cacheSize = size;
}

uint64_t applyChain(uint64_t start, uint16_t* chain, int length) {
    for (int i = 0; i < length; i++) {
        start = hex_layer64(start, chain[i]);
    }
    return start;
}

void printChain(uint16_t* chain, int length) {
    const char layerStrings[][16] = {
        "%X, %X",
        "%X, *%X",
        "*%X, %X",
        "*%X, *%X",
        "^%X, *%X",
        "^*%X, %X"
    };
    for (int i = 0; i < length; i++) {
        uint16_t conf = chain[i];
        printf(layerStrings[conf >> 8], (conf >> 4) & 15, conf & 15);
        if (i < length - 1) printf(";  ");
    }
}

void printHlpMap(uint64_t map) {
    hlp_request_t request = {map, map};
    printHlpRequest(request);
}

void printHlpRequest(hlp_request_t request) {
    for (int i = 15; i >= 0; i--) {
        if (i % 4 == 3 && i != 15) printf(" ");
        int minVal = (request.mins >> i * 4) & 15;
        int maxVal = (request.maxs >> i * 4) & 15;
        if (minVal == maxVal) {
            printf("%X", minVal);
            continue;
        }
        if (minVal == 0 && maxVal == 15) {
            printf("X");
            continue;
        }
        printf("[%X-%X]", minVal, maxVal);
    }
}

void hlpPrintSearch(char* map) {
    uint16_t result[32];
    hlp_request_t request = parseHlpRequestStr(map);
    switch (request.error) {
        case HLP_ERROR_NULL:
        case HLP_ERROR_BLANK:
            printf("Error: must provide a function to solve for\n");
            return;
        case HLP_ERROR_TOO_LONG:
            printf("Error: too many values are provided\n");
            return;
        case HLP_ERROR_MALFORMED:
            printf("Error: malformed expression\n");
            return;
    }

    if (hlpSolveVerbosity > 0) {
        printf("searching for ");
        printHlpRequest(request);
        printf("\n");
    }

    int length = solve(request, result, globalMaxDepth, globalAccuracy);

    if (length > globalMaxDepth) {
        if (hlpSolveVerbosity > 0)
            printf("no result found\n");
    } else {
        if (hlpSolveVerbosity > 0) {
            printf("result found, length %d", length);
            if (hlpSolveVerbosity > 2 || request.solveType != HLP_SOLVE_TYPE_EXACT) {
                printf(" (");
                printHlpMap(applyChain(IDENTITY_PERM_BE64, result, length));
                printf(")");
            }
            printf(":  ");
        }
        printChain(result, length);
        printf("\n");
    }
}

enum LONG_OPTIONS {
    LONG_OPTION_MAX_DEPTH = 1000,
    LONG_OPTION_ACCURACY,
    LONG_OPTION_CACHE_SIZE
};

static const struct argp_option options[] = {
    { "fast", 'f', 0, 0, "Equivilant to --accuracy -1" },
    { "perfect", 'p', 0, 0, "Equivilant to --accuracy 2" },
    { "max-length", LONG_OPTION_MAX_DEPTH, "N", 0, "Limit results to chains up to N layers long" },
    { "accuracy", LONG_OPTION_ACCURACY, "LEVEL", 0, "Set search accuracy from -1 to 2, 0 being normal, 2 being perfect" },
    { "cache", LONG_OPTION_CACHE_SIZE, "N", 0, "Set the cache size to 2**N bytes. default: 26 (64MB)" },
    { 0 }
};

static error_t parse_opt(int key, char* arg, struct argp_state *state) {
    struct arg_settings_solver_hex* settings = state->input;
    switch (key) {
        case LONG_OPTION_ACCURACY:
            int level = atoi(arg);
            if (level < -1 || level > 2)
                argp_error(state, "%s is not a valid accuracy", arg);
            else
                globalAccuracy = level;
            break;
        case 'f':
            return parse_opt(LONG_OPTION_ACCURACY, "-1", state);
        case 'p':
            return parse_opt(LONG_OPTION_ACCURACY, "2", state);
        case LONG_OPTION_MAX_DEPTH:
            globalMaxDepth = atoi(arg);
            break;
        case LONG_OPTION_CACHE_SIZE:
            hlpSetCacheSize(atoi(arg) - 4);
            break;
        case ARGP_KEY_INIT:
            globalAccuracy = ACCURACY_NORMAL;
            globalMaxDepth = 31;
            settings->settings_redstone.global = settings->global;
            state->child_inputs[0] = &settings->settings_redstone;
            break;
        case ARGP_KEY_SUCCESS:
            hlpSolveVerbosity = settings->global->verbosity;
            break;
    }
    return 0;
}

static struct argp_child argp_children[] = {
    {&argp_redstone},
    { 0 }
};

struct argp argp_solver_hex = {
    options,
    parse_opt,
    0,
    0,
    argp_children
};


//#pragma GCC pop_options
