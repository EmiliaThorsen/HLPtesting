#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <x86intrin.h>
#include "aa_tree.h"
#include <stdbool.h>

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

uint64_t startPos = 0x0123456789ABCDEF; //DO NOT CHANGE
int cacheSize = 25;

//the goal you want to search to
uint64_t wanted;

uint8_t goal[16]; //goal in array form instead of packed nibbles in uint
/* uint64_t goal_uint; //trust me */

//precomputed layer lookup tables
uint16_t* layerConf;
uint16_t* nextValidLayers;
uint64_t* nextValidLayerLuts;
int* nextValidLayersSize;


uint16_t* layerConfAll[16];
uint16_t* nextValidLayersAll[16];
uint64_t* nextValidLayerLutsAll[16];
int* nextValidLayersSizeAll[16];
int layerPrecomputesFinished[16] = {0};

int iter = 0;
int currLayer = 1;
int layerCount = 0;

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
        for (int i=0; i<layerCount; i++)
            if (list[i] == item)
                return 1;
        return 0;
}
//precompute of layers into lut, proceding layers deduplicated for lower branching, and distance estimate table
void precomputeLayers(int group) {
    if (layerPrecomputesFinished[group]) {
        layerConf = layerConfAll[group];
        nextValidLayers = nextValidLayersAll[group];
        nextValidLayerLuts = nextValidLayerLutsAll[group];
        nextValidLayersSize = nextValidLayersSizeAll[group];
        return;
    }

    layerConf = malloc(800*sizeof(uint16_t));
    nextValidLayers = malloc(800*800*sizeof(uint16_t));
    nextValidLayerLuts = calloc(800*800, sizeof(uint64_t));
    nextValidLayersSize = malloc(800*sizeof(int));

    /* printf("starting layer precompute\n"); */
    int64_t flag;
    aatree_node* uniqueNextLayersTree = aa_tree_insert(startPos, NULL, &flag);
    /* uint64_t uniqueNextLayersList[800*800]; */
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

    layerConfAll[group] = layerConf;
    nextValidLayersAll[group] = nextValidLayers;
    nextValidLayerLutsAll[group] = nextValidLayerLuts;
    nextValidLayersSizeAll[group] = nextValidLayersSize;
    printf("layers computed:%d, total next layers:%ld\n", layerCount, totalNext - layerCount);
}


//faster implementation of searching over the last layer while checking if you found the goal, unexpectedly big optimization
int fastLastLayerSearch(uint64_t input, int prevLayerConf) {

    int search_size = nextValidLayersSize[prevLayerConf];
    iter += search_size;
    int index = search_last_layer(input, nextValidLayerLuts + 800*prevLayerConf, search_size, wanted);
    if (index == -1) return 0;

    iter -= index;

    uint16_t config = layerConf[nextValidLayers[prevLayerConf * 800 + index]];
    printf("depth: %d configuration: %03hx\n", currLayer - 1, config);
    return 1;
}

//cache related code, used for removing identical or worse solutions
long sameDepthHits = 0;
long difLayerHits = 0;
long misses = 0;
long bucketUtil = 0;


typedef struct cache_entry_s {
    uint64_t map;
    uint8_t depth;
    uint8_t round;
} cache_entry_t;

cache_entry_t* cacheArr;
uint64_t cacheMask;
int cacheChecksTotal;

void clearCache() {
    for (int i = 0; i< (1<<cacheSize); i++) {
        cacheArr[i].map = 0;
        cacheArr[i].depth = 0;
        cacheArr[i].round = 0;
    }
}

int cacheCheck(uint64_t output, int depth) {
    uint32_t pos = _mm_crc32_u32(_mm_crc32_u32(0, output & 0xFFFFFFFF), (output >> 32) & 0xFFFFFFFF) & cacheMask;
    cache_entry_t* entry = cacheArr + pos;
    cacheChecksTotal++;
    if (entry->map == output && entry->depth <= depth && entry->round == currLayer ){
        if (entry->depth == depth) sameDepthHits++;
        else difLayerHits++;
        return 1;
    }

    if (entry->map == 0) bucketUtil++;
    else misses++;

    /* entry->goal = wanted; */
    entry->map = output;
    entry->depth = depth;
    entry->round = currLayer;

    return 0;
}

int tmp=0;

//main dfs recursive search function
int dfs(uint64_t startPos, int depth, int prevLayerConf) {
    if(depth == currLayer - 1) return fastLastLayerSearch(startPos, prevLayerConf);
    int dedupeArrSize = 0;
    uint64_t potentialLayers[1600];
    iter += nextValidLayersSize[prevLayerConf];
    int totalNextLayersIdentified = batch_apply_and_check(
            startPos,
            nextValidLayerLuts + prevLayerConf*800,
            potentialLayers,
            nextValidLayersSize[prevLayerConf],
            currLayer - depth - 1
            );
    for(int i = 0; i < totalNextLayersIdentified*2; i+=2) {
        int conf = potentialLayers[i];
        /* uint64_t output = apply_and_check(startPos, nextValidLayerLuts[800*prevLayerConf + conf], currLayer - depth - 1); */
        /* if (output == 0) printf("a"); */
        uint64_t output = potentialLayers[i + 1];

        //cache check
        if(cacheCheck(output, depth)) continue;



        int index = nextValidLayers[prevLayerConf * 800 + conf];
        //call next layers
        if(dfs(output, depth + 1, index)) {
            printf("depth: %d configuration: %03hx\n", depth, layerConf[index]);
            return 1;
        }
        /* if(depth == 0 & currLayer > 8) printf("done:%d/%d\n", conf, nextValidLayersSize[prevLayerConf]); */
    }
    return 0;
}


//main search loop
void search(uint64_t m) {
    iter = 0;
    currLayer = 1;
    layerCount = 0;
    cacheChecksTotal = 0;
    wanted = fix_uint(m);

    clock_t programStartT = clock();
    precomputeLayers(getGroup(wanted));
    /* printf("layer precompute done at %fs\n", (double)(clock() - programStartT) / CLOCKS_PER_SEC); */
    /* printf("starting search!\n"); */
    uint_to_array(wanted, goal);

    while (1) {
        if(dfs(startPos, 0, 799)) break;
        /* clearCache(); */
        printf("search over layer: %d done!\n",currLayer);
        /* printf("layer search done after %fs\n", (double)(clock() - programStartT) / CLOCKS_PER_SEC); */
        /* printf("iterations: %ld\n", iter); */
        /* printf("same depth hits:%ld dif layer hits:%ld misses: %ld bucket utilization: %ld\n", sameDepthHits, difLayerHits, misses, bucketUtil); */
        sameDepthHits = 0;
        difLayerHits = 0;
        misses = 0;
        bucketUtil = 0;
        //iter = 0;
        currLayer++;
        if(currLayer > 42) break;
    }
    printf("total iter over all: %ld\n", iter);
    printf("cache checks: %ld; same depth hits:%ld; dif layer hits:%ld; misses: %ld; bucket utilization: %ld\n", cacheChecksTotal, sameDepthHits, difLayerHits, misses, bucketUtil);
}

int main(int argc, char** argv) {

    /*
    uint64_t asdf = startPos;

    int applies[] = { 0x143, 0x187, 0x121, 0x3ff, 0x143, 0x121, 0x1cb, 0x121, 0x143 };
    for(int i=0; i<sizeof(applies)/sizeof(int); i++) {
        asdf = layer(asdf, applies[i]);
    }
    printf("%lx\n", asdf);
    return 0; */

    startPos = fix_uint(startPos);

    //alocating the cache
    cacheArr = calloc((1 << cacheSize), sizeof(cache_entry_t));
    cacheMask = (1 << cacheSize) - 1;

    for (int i = 1; i < argc; i++) {

        uint64_t to_find = strtoull(argv[i], 0, 16); // this is totally safe, right?
        printf("searching for %016lx\n", to_find);
        search(to_find);
        clearCache();
    }
}
