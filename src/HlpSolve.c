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

typedef struct branch_layer_s {
    uint64_t map;
    uint16_t configIndex;
    uint8_t separations;
    uint8_t padding[5];
} branch_layer_t;

extern uint64_t apply_mapping(uint64_t input, uint64_t map);
extern uint64_t apply_and_check(uint64_t input, uint64_t map, int threshhold);

extern void uint_to_array(uint64_t uint, uint8_t* array);

// takes a uint from a pretty form to one that works nicely for vectorizing
extern uint64_t fix_uint(uint64_t uint);

extern uint64_t layer(uint64_t map, uint16_t config);

extern int batch_apply_and_check(
        uint64_t start,
        uint64_t* input_maps,
        branch_layer_t* outputs,
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
int _uniqueOutputs;
int _solutionsFound;
int _searchAccuracy;
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

    /* printf("found solution\n"); */
    uint16_t config = layerConf[nextValidLayers[prevLayerConf * 800 + index]];
    /* printf("depth: %d configuration: %03hx\n", currLayer - 1, config); */
    if (_solutionsFound != -1) {
        _solutionsFound++;
        return 0;
    }
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

int cmpfunc (const void * a, const void * b) {
   return ( ((branch_layer_t*)a)->separations - ((branch_layer_t*)b)->separations );
}



// the most number of separations that can be found in the distance check before it prunes
int getDistThreshold(int remainingLayers) {
    if (_searchAccuracy == ACCURACY_REDUCED) return remainingLayers - 1;
    // n is always sufficient anyways for 15-16 outputs
    if (_searchAccuracy == ACCURACY_NORMAL || _uniqueOutputs > 14) return remainingLayers;
    // n+1 is always sufficient for 14 outputs
    if (_searchAccuracy == ACCURACY_INCREASED || _uniqueOutputs > 13) return remainingLayers + 1;

    // currently the best known general threshhold
    // +/-1 is for round up division
    return ((remainingLayers * 3 - 1) >> 1) + 1;
}

branch_layer_t potentialLayers[800*32];

//main dfs recursive search function
int dfs(uint64_t input, int depth, int prevLayerConf) {
    if(depth == currLayer - 1) return fastLastLayerSearch(input, prevLayerConf);
    iter += nextValidLayersSize[prevLayerConf];

    int totalNextLayersIdentified = batch_apply_and_check(
            input,
            nextValidLayerLuts + prevLayerConf*800,
            potentialLayers + 800*depth,
            nextValidLayersSize[prevLayerConf],
            getDistThreshold(currLayer - depth - 1)
            );

    // this adds a very slight boost by checking more promising branches first
    if (depth < currLayer >> 1)
        qsort(potentialLayers + 800*depth, totalNextLayersIdentified, sizeof(branch_layer_t), cmpfunc);

    for(int i = 0; i < totalNextLayersIdentified; i++) {
        branch_layer_t* entry = potentialLayers + 800*depth + i;
        int conf = entry->configIndex;
        /* uint64_t output = apply_and_check(input, nextValidLayerLuts[800*prevLayerConf + conf], currLayer - depth - 1); */
        /* if (output == 0) printf("a"); */
        uint64_t output = entry->map;

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

void init(uint64_t map) {
    if (!cacheArr) cacheArr = calloc((1 << cacheSize), sizeof(cache_entry_t));
    cacheMask = (1 << cacheSize) - 1;
    iter = 0;
    cacheChecksTotal = 0;
    wanted = fix_uint(map);
    _uniqueOutputs = getGroup(wanted);
    precomputeLayers(_uniqueOutputs);
    uint_to_array(wanted, goal);
}


int searchOneDepth(int depth) {
    return 0;
}

//main search loop
int singleSearchInner(int maxDepth) {
    currLayer = 1;

    /* clock_t programStartT = clock(); */
    /* printf("layer precompute done at %fs\n", (double)(clock() - programStartT) / CLOCKS_PER_SEC); */
    /* printf("starting search!\n"); */

    while (currLayer <= maxDepth) {
        if(dfs(startPos, 0, 799)) return currLayer;
        invalidateCache();
        /* printf("search over layer: %d done!\n",currLayer); */
        /* printf("layer search done after %fs\n", (double)(clock() - programStartT) / CLOCKS_PER_SEC); */
        currLayer++;
    }
    return maxDepth + 1;
    /* printf("total iter over all: %ld\n", iter); */
    /* printf("cache checks: %ld; same depth hits:%ld; dif layer hits:%ld; misses: %ld; bucket utilization: %ld\n", cacheChecksTotal, sameDepthHits, difLayerHits, misses, bucketUtil); */
}

int singleSearch(uint64_t m, uint16_t* outputChain, int maxDepth, enum SearchAccuracy accuracy) {
    if (maxDepth < 0 || maxDepth > 31) maxDepth = 31;
    if (m == 0) {
        if (outputChain) outputChain[0] = 0x2f0;
        return 1;
    }
    if (m == hlpStartPos) return 0;


    init(m);

    _outputChain = outputChain;
    _solutionsFound = -1;
    _searchAccuracy = accuracy;
    return singleSearchInner(maxDepth);
}

int solve(uint64_t m, uint16_t* outputChain, enum SearchAccuracy accuracy) {
    if (m == 0) {
        if (outputChain) outputChain[0] = 0x2f0;
        return 1;
    }
    if (m == hlpStartPos) return 0;

    init(m);

    _outputChain = outputChain;
    _solutionsFound = -1;
    int solutionLength = 31;

    // reduced accuracy search is sometimes faster than the others but
    // still often gets an optimal solution, so we start with that so the
    // "real" search can cut short if it doesn't find a better solution.
    // when it's not faster, the solution is found pretty fast anyways.
    _searchAccuracy = ACCURACY_REDUCED;
    solutionLength = singleSearchInner(solutionLength);

    if (solutionLength == 31) solutionLength = 31;
    if (accuracy == ACCURACY_REDUCED) return solutionLength;

    _searchAccuracy = accuracy;
    return singleSearchInner(solutionLength - 1);
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
