#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <immintrin.h>
#include "aa_tree.h"
#include "HlpSolve.h"
#include <stdbool.h>

// not sure what it causing this to be needed, but something breaks
#pragma GCC push_options
#pragma GCC optimize ("O0")


extern uint64_t apply_mapping(uint64_t input, uint64_t map);
extern uint64_t apply_and_check(uint64_t input, uint64_t map, int threshhold);

extern void uint_to_array(uint64_t uint, uint8_t* array);

// takes a uint from a pretty form to one that works nicely for vectorizing
extern uint64_t fix_uint(uint64_t uint);

extern uint64_t layer(uint64_t map, uint16_t config);

extern int batch_apply_and_check(
        uint64_t start,
        uint64_t* input_maps,
        uint64_t* output_ids_and_maps,
        int quantity,
        int threshhold
        );

extern int search_last_layer(
        uint64_t input,
        uint64_t* maps,
        int quantity,
        uint64_t goal);

const uint64_t hlpStartPos = 0x0123456789abcdef;
const uint64_t startPos = 0x7f6e5d4c3b2a1908;
const uint64_t broadcastH16 = 0x1111111111111111;
int cacheSize = 22;

uint64_t wanted;

uint8_t goal[16]; //goal in array form instead of packed nibbles in uint

//precomputed layer lookup tables
uint16_t* layerConf;
uint16_t* nextValidLayers;
uint64_t* nextValidLayerLuts;
int* nextValidLayersSize;


uint16_t* layerConfAll[16] = {0};
uint16_t* nextValidLayersAll[16] = {0};
uint64_t* nextValidLayerLutsAll[16] = {0};
int* nextValidLayersSizeAll[16] = {0};
uint16_t layerPrecomputesFinished = 0;

uint16_t getLayerConf(int group, int layerId) { return layerConfAll[group - 1][layerId];}
uint16_t getNextValidLayerId(int group, int prevLayerId, int index) { return nextValidLayersAll[group - 1][800 * prevLayerId + index]; }
uint16_t getNextValidLayerSize(int group, int layerId) { return nextValidLayersSizeAll[group - 1][layerId]; }

int iter;
int currLayer;
uint16_t* _outputChain;

//counts uniqe values, usefull for generalizable prune of layers that reduce too much from the get go
int getGroup(uint64_t x) {
    int group = 0;
    uint16_t bitFeild = 0;
    for(int i = 0; i < 16; i++) {
        int ss = (x >> (i * 4)) & 15;
        if(bitFeild & (1 << ss)) continue;
        bitFeild |= 1 << ss;
        group++;
    }
    return group;
}

bool inList(uint64_t item, uint64_t* list, int maxIndex) {
        for (int i=0; i<maxIndex; i++)
            if (list[i] == item)
                return 1;
        return 0;
}
//precompute of layers into lut, proceding layers deduplicated for lower branching, and distance estimate table
void precomputeLayers(int group) {
    if ((layerPrecomputesFinished << (group - 1)) & 1) {
        layerConf = layerConfAll[group];
        nextValidLayers = nextValidLayersAll[group];
        nextValidLayerLuts = nextValidLayerLutsAll[group];
        nextValidLayersSize = nextValidLayersSizeAll[group];
        return;
    }

    int layerCount = 0;
    layerConf = malloc(800*sizeof(uint16_t));
    nextValidLayers = malloc(800*800*sizeof(uint16_t));
    nextValidLayerLuts = calloc(800*800, sizeof(uint64_t));
    nextValidLayersSize = malloc(800*sizeof(int));

    /* printf("starting layer precompute\n"); */
    int64_t flag;
    aatree_node* uniqueNextLayersTree = aa_tree_insert(startPos, NULL, &flag);
    /* uint64_t* uniqueNextLayersList = calloc(800*800, sizeof(uint64_t)); */
    /* for (int i=0; i<800*800; i++) uniqueNextLayersList[i] = 0; */

    for(int conf = 0; conf < 1536; conf++) {
        uint64_t output = layer(startPos, conf);
        if(getGroup(output) < group) continue;
        if (output == startPos) continue;
        if (aa_tree_search(uniqueNextLayersTree, output)) continue;
        uniqueNextLayersTree = aa_tree_insert(output, uniqueNextLayersTree, &flag);
        /* if (inList(output, uniqueNextLayersList, layerCount)) continue; */
        /* uniqueNextLayersList[layerCount] = output; */

        layerConf[layerCount] = conf;
        nextValidLayers[799 * 800 + layerCount] = layerCount;
        nextValidLayerLuts[799 * 800 + layerCount] = output;
        layerCount++;
    }
    nextValidLayersSize[799] = layerCount;
    long totalNext = layerCount;
    /* printf("starting next layer compute\n"); */
    for(int conf = 0; conf < layerCount; conf++) {
        uint64_t layerOut = nextValidLayerLuts[799*800 + conf];
        int nextLayerSize = 0;
        for (int conf2 = 0; conf2 < layerCount; conf2++) {
            uint64_t nextLayerOut = apply_mapping(layerOut, nextValidLayerLuts[799*800 + conf2]);
            if(getGroup(nextLayerOut) < group) continue;
            if (nextLayerOut == startPos) continue;

            if (aa_tree_search(uniqueNextLayersTree, nextLayerOut)) continue;
            uniqueNextLayersTree = aa_tree_insert(nextLayerOut, uniqueNextLayersTree, &flag);
            /* if (inList(nextLayerOut, uniqueNextLayersList, totalNext)) continue; */
            /* uniqueNextLayersList[totalNext] = nextLayerOut; */

            nextValidLayers[conf * 800 + nextLayerSize] = conf2;
            nextValidLayerLuts[conf * 800 + nextLayerSize] = nextValidLayerLuts[799*800 + conf2];
            totalNext++;
            nextLayerSize++;
        }
        for (int i=0; i<4; i++)
            nextValidLayerLuts[conf*800 + nextLayerSize + i] = 0;
        nextValidLayersSize[conf] = nextLayerSize;
    }

    aa_tree_free(uniqueNextLayersTree);
    /* free(uniqueNextLayersList); */

    layerConfAll[group-1] = layerConf;
    nextValidLayersAll[group-1] = nextValidLayers;
    nextValidLayerLutsAll[group-1] = nextValidLayerLuts;
    nextValidLayersSizeAll[group-1] = nextValidLayersSize;
    layerPrecomputesFinished |= 1 << (group -1);

    /* printf("layers computed:%d, total next layers:%ld\n", layerCount, totalNext - layerCount); */
}


//faster implementation of searching over the last layer while checking if you found the goal, unexpectedly big optimization
int fastLastLayerSearch(uint64_t input, int prevLayerConf) {

    int search_size = nextValidLayersSize[prevLayerConf];
    iter += search_size;
    int index = search_last_layer(input, nextValidLayerLuts + 800*prevLayerConf, search_size, wanted);
    if (index == -1) return 0;

    iter -= index;

    uint16_t config = layerConf[nextValidLayers[prevLayerConf * 800 + index]];
    /* printf("depth: %d configuration: %03hx\n", currLayer - 1, config); */
    if (_outputChain != 0) _outputChain[currLayer - 1] = config;
    return 1;
}

//cache related code, used for removing identical or worse solutions
long sameDepthHits = 0;
long difLayerHits = 0;
long misses = 0;
long bucketUtil = 0;


typedef struct cache_entry_s {
    uint64_t map;
    uint32_t trial;
    uint8_t depth;
} cache_entry_t;

cache_entry_t* cacheArr;
uint64_t cacheMask;
int cacheChecksTotal;
uint32_t cacheTrialGlobal = 0;

void clearCache() {
    for (int i = 0; i< (1<<cacheSize); i++) {
        cacheArr[i].map = 0;
        cacheArr[i].depth = 0;
        cacheArr[i].trial = 0;
    }
}

int cacheCheck(uint64_t output, int depth) {
    uint32_t pos = _mm_crc32_u32(output & UINT32_MAX, output >> 32) & cacheMask;
    cache_entry_t* entry = cacheArr + pos;
    if (entry->map == output && entry->depth <= depth && entry->trial == cacheTrialGlobal) return 1;

    entry->map = output;
    entry->depth = depth;
    entry->trial = cacheTrialGlobal;

    return 0;
}

void invalidateCache() {
    cacheTrialGlobal++;
    if (!cacheTrialGlobal) {
        clearCache();
        // trial 0 should always mean blank
        cacheTrialGlobal++;
    }
}


//main dfs recursive search function
int dfs(uint64_t input, int depth, int prevLayerConf) {
    if(depth == currLayer - 1) return fastLastLayerSearch(input, prevLayerConf);
    uint64_t potentialLayers[1600];
    iter += nextValidLayersSize[prevLayerConf];

    int totalNextLayersIdentified = batch_apply_and_check(
            input,
            nextValidLayerLuts + prevLayerConf*800,
            potentialLayers,
            nextValidLayersSize[prevLayerConf],
            currLayer - depth - 1
            );
    for(int i = 0; i < totalNextLayersIdentified << 1; i+=2) {
        int conf = potentialLayers[i];
        /* uint64_t output = apply_and_check(input, nextValidLayerLuts[800*prevLayerConf + conf], currLayer - depth - 1); */
        /* if (output == 0) printf("a"); */
        uint64_t output = potentialLayers[i + 1];

        //cache check
        if(cacheCheck(output, depth)) continue;



        int index = nextValidLayers[prevLayerConf * 800 + conf];
        //call next layers
        if(dfs(output, depth + 1, index)) {
            if (_outputChain != 0) _outputChain[depth] = layerConf[index];
            /* printf("depth: %d configuration: %03hx\n", depth, layerConf[index]); */
            return 1;
        }
        /* if(depth == 0 & currLayer > 8) printf("done:%d/%d\n", conf, nextValidLayersSize[prevLayerConf]); */
    }
    return 0;
}

void init() {
    if (!cacheArr) cacheArr = calloc((1 << cacheSize), sizeof(cache_entry_t));
    cacheMask = (1 << cacheSize) - 1;
}


//main search loop
int search(uint64_t m, uint16_t* outputChain, int maxDepth) {
    if (maxDepth < 0 || maxDepth > 31) maxDepth = 32;
    if (m == 0) {
        if (outputChain) outputChain[0] = 0x2f0;
        return 1;
    }
    if (m == hlpStartPos) {
        return 0;
    }

    init();

    iter = 0;
    currLayer = 1;
    cacheChecksTotal = 0;
    wanted = fix_uint(m);
    _outputChain = outputChain;

    /* clock_t programStartT = clock(); */
    precomputeLayers(getGroup(wanted));
    /* printf("layer precompute done at %fs\n", (double)(clock() - programStartT) / CLOCKS_PER_SEC); */
    /* printf("starting search!\n"); */
    uint_to_array(wanted, goal);

    while (currLayer <= maxDepth) {
        if(dfs(startPos, 0, 799)) return currLayer;
        invalidateCache();
        /* printf("search over layer: %d done!\n",currLayer); */
        /* printf("layer search done after %fs\n", (double)(clock() - programStartT) / CLOCKS_PER_SEC); */
        /* printf("iterations: %ld\n", iter); */
        /* printf("same depth hits:%ld dif layer hits:%ld misses: %ld bucket utilization: %ld\n", sameDepthHits, difLayerHits, misses, bucketUtil); */
        sameDepthHits = 0;
        difLayerHits = 0;
        misses = 0;
        bucketUtil = 0;
        //iter = 0;
        currLayer++;
    }
    return maxDepth + 1;
    /* printf("total iter over all: %ld\n", iter); */
    /* printf("cache checks: %ld; same depth hits:%ld; dif layer hits:%ld; misses: %ld; bucket utilization: %ld\n", cacheChecksTotal, sameDepthHits, difLayerHits, misses, bucketUtil); */
}

void hlpSetCacheSize(int size) {
    if (cacheArr) {
        free(cacheArr);
        cacheArr = 0;
    }
    cacheSize = size;
    cacheMask = (1 << size) - 1;
}

#pragma GCC pop_options
