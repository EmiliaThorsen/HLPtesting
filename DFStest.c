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
/* extern void bitonic_sort16x8(uint8_t* array); */
/* extern uint64_t array_to_uint(uint8_t* array); */
extern void uint_to_array(uint64_t uint, uint8_t* array);
/* extern void pretty_uint_to_array(uint64_t uint, uint8_t* array); */
/* extern void store_mapping(uint64_t map, uint8_t* location); */

// takes a uint from a pretty form to one that works nicely for vectorizing
extern uint64_t fix_uint(uint64_t uint);

extern uint64_t layer(uint64_t map, uint16_t config);
extern uint64_t apply_and_check(uint64_t input, uint64_t map, int threshhold);
/* extern uint64_t check(uint64_t input, int threshhold); */


extern void batch_apply(
        uint64_t start,
        uint64_t* output,
        uint16_t* configurations,
        int quantity);

extern int batch_apply_and_check(
        uint64_t start,
        uint64_t* input_maps,
        uint16_t* output_ids,
        int quantity,
        int threshhold,
        uint64_t goal
        );

extern int search_last_layer(
        uint64_t input,
        uint64_t* maps,
        int quantity,
        uint64_t goal);


const uint64_t broadcast8 = 0x0101010101010101;
const uint64_t broadcast16 = broadcast8 | (broadcast8 << 4);
const uint64_t barrier8 = broadcast8 * 0x80;
const uint64_t low_halves64 = broadcast8 * 15;
const uint64_t high_halves64 = low_halves64 << 4;

uint64_t discount_simd_max_half(uint64_t a, uint64_t b) {
    uint64_t mask = (((barrier8 + a - b) ^ barrier8) >> 4) & low_halves64;
    return (~mask & a) | (mask & b);
}

uint64_t discount_simd_max(uint64_t a, uint64_t b) {
    uint64_t low = discount_simd_max_half(a & low_halves64, b & low_halves64);
    uint64_t high = discount_simd_max_half((a & high_halves64) >> 4, (b & high_halves64) >> 4);
    return low | (high << 4);
}

uint64_t discount_simd_comparator_half(uint64_t back, uint64_t side, bool mode) {
    uint64_t diff = barrier8 + back - side;
    uint64_t cancel = ((diff ^ barrier8 ^ high_halves64) >> 4) & low_halves64;
    return (mode ? diff : back) & cancel;
}

uint64_t discount_simd_comparator(uint64_t back, uint64_t side, bool mode) {
    uint64_t low = discount_simd_comparator_half(back & low_halves64, side & low_halves64, mode);
    uint64_t high = discount_simd_comparator_half((back & high_halves64) >> 4, (side & high_halves64) >> 4, mode);
    return low | (high << 4);
}

uint64_t startPos = 0x0123456789ABCDEF; //DO NOT CHANGE

//the goal you want to search to
uint64_t wanted = 0x3141592653589793; //0x77239AB34567877E 0x0022002288AA88AA 0x1122334455667788 0x1111222233334444 0x91326754CDFEAB98

uint8_t goal[16]; //goal in array form instead of packed nibbles in uint
/* uint64_t goal_uint; //trust me */

//precomputed layer lookup tables
uint16_t layerConf[800];
uint16_t nextValidLayers[800*800];
uint64_t nextValidLayerLuts[800*800];
int nextValidLayersSize[800];
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

int cmpfunc (void * a, void * b) {
   return ( *(uint64_t*)a - *(uint64_t*)b );
}

//precompute of layers into lut, proceding layers deduplicated for lower branching, and distance estimate table
void precomputeLayers(int group) {
    printf("starting layer precompute\n");
    int64_t flag;
    aatree_node* uniqueNextLayersTree = aa_tree_insert(startPos, NULL, &flag);

    for (int i=0; i<800*800; i++) nextValidLayerLuts[i] = 0;

    for(int conf = 0; conf < 1536; conf++) {
        uint64_t output = layer(startPos, conf);
        if(getGroup(output) < group) continue;
        if (aa_tree_search(uniqueNextLayersTree, output)) continue;

        layerConf[layerCount] = conf;
        nextValidLayers[799 * 800 + layerCount] = layerCount;
        nextValidLayerLuts[799 * 800 + layerCount] = output;
        uniqueNextLayersTree = aa_tree_insert(output, uniqueNextLayersTree, &flag);
        layerCount++;
    }
    nextValidLayersSize[799] = layerCount;
    long totalNext = layerCount;
    printf("starting next layer compute\n");
    for(int conf = 0; conf < layerCount; conf++) {
        uint64_t layerOut = nextValidLayerLuts[799*800 + conf];
        int nextLayerSize = 0;
        for (int conf2 = 0; conf2 < layerCount; conf2++) {
            uint64_t nextLayerOut = apply_mapping(layerOut, nextValidLayerLuts[799*800 + conf2]);
            if(getGroup(nextLayerOut) < group) continue;
            if (aa_tree_search(uniqueNextLayersTree, nextLayerOut)) continue;

            nextValidLayers[conf * 800 + nextLayerSize] = conf2;
            nextValidLayerLuts[conf * 800 + nextLayerSize] = nextValidLayerLuts[799*800 + conf2];
            uniqueNextLayersTree = aa_tree_insert(nextLayerOut, uniqueNextLayersTree, &flag);
            totalNext++;
            nextLayerSize++;
        }
        for (int i=0; i<4; i++)
            nextValidLayerLuts[conf*800 + nextLayerSize + i] = 0;
        nextValidLayersSize[conf] = nextLayerSize;
    }
    aa_tree_free(uniqueNextLayersTree);
    printf("layers computed:%d, total next layers:%ld\n", layerCount, totalNext - layerCount);
}


long iter = 0;

int currLayer = 1;

int mismatches = 0;

//fast lut based layer, also does a check to see if the output is invalid and inposible to use to find goal
uint64_t fastLayer(uint64_t input, uint64_t map, int threshhold) {
    iter++;
    return apply_and_check(input, map, threshhold);
}

//faster implementation of searching over the last layer while checking if you found the goal, unexpectedly big optimization
int fastLastLayerSearch(uint64_t input, int prevLayerConf) {

    int search_size = nextValidLayersSize[prevLayerConf];
    iter += search_size;
    int index = search_last_layer(input, nextValidLayerLuts + 800*prevLayerConf, search_size, wanted);
    if (index == -1) return 0;

    iter -= index;

    uint16_t config = layerConf[nextValidLayers[prevLayerConf * 800 + index]];
    printf("deapth: %d configuration: %03hx\n", currLayer - 1, config);
    return 1;
}

//cache related code, used for removing identical or worse solutions
long sameDeapthHits = 0;
long difLayerHits = 0;
long misses = 0;
long bucketUtil = 0;

uint64_t *casheArr;
uint8_t *casheDeapthArr;
uint64_t casheMask;


int casheCheck(uint64_t output, int deapth) {
    uint32_t pos = _mm_crc32_u32(_mm_crc32_u32(0, output & 0xFFFFFFFF), (output >> 32) & 0xFFFFFFFF) & casheMask;
    if(casheArr[pos] == output & casheDeapthArr[pos] <= deapth) {
        if(casheDeapthArr[pos] == deapth) {
            sameDeapthHits++;
            return 1;
        }
        difLayerHits++;
        return 1;
    }
    if(casheArr[pos] == 0) {
        bucketUtil++;
    } else {
        misses++;
    }
    casheArr[pos] = output;
    casheDeapthArr[pos] = deapth;
    return 0;
}

int tmp = 0;

//main dfs recursive search function
int dfs(uint64_t startPos, int deapth, int prevLayerConf) {
    if(deapth == currLayer - 1) return fastLastLayerSearch(startPos, prevLayerConf);
    int dedupeArrSize = 0;
    uint16_t potentialLayers[800];
    iter += nextValidLayersSize[prevLayerConf];
    int totalNextLayersIdentified = batch_apply_and_check(
            startPos,
            nextValidLayerLuts + prevLayerConf*800,
            potentialLayers,
            nextValidLayersSize[prevLayerConf],
            currLayer - deapth - 1,
            wanted
            );
    for(int i = 0; i < totalNextLayersIdentified; i++) {
        /* int conf = i; */
        int conf = potentialLayers[i];
        uint64_t map = nextValidLayerLuts[prevLayerConf * 800 + conf];
        uint64_t output = apply_mapping(startPos, map);
        /* uint64_t output = apply_and_check(startPos, map, currLayer - deapth - 1); */
        /* if (output == 0) continue; */

        //cashe check
        if(casheCheck(output, deapth)) continue;

        int index = nextValidLayers[prevLayerConf * 800 + conf];
        //call next layers
        if(dfs(output, deapth + 1, index)) {
            printf("deapth: %d configuration: %03hx\n", deapth, layerConf[index]);
            return 1;
        }
        if(deapth == 0 & currLayer > 8) printf("done:%d/%d\n", conf, nextValidLayersSize[prevLayerConf]);
    }
    return 0;
}

//main search loop
void search() {
    clock_t programStartT = clock();
    precomputeLayers(getGroup(wanted));
    printf("layer precompute done at %fs\n", (double)(clock() - programStartT) / CLOCKS_PER_SEC);
    printf("starting search!\n");
    uint_to_array(wanted, goal);

    /* nextValidLayerLuts[799*800 + 415] = wanted; */
    /* fastLastLayerSearch(startPos, 0); */

    while (1) {
        for(int i = 0; i < casheMask + 1; i++) casheArr[i] = 0;
        if(dfs(startPos, 0, 799)) break;
        printf("search over layer: %d done!\n",currLayer);
        printf("layer search done after %fs\n", (double)(clock() - programStartT) / CLOCKS_PER_SEC);
        printf("iterations: %ld\n", iter);
        printf("same deapth hits:%ld dif layer hits:%ld misses: %ld bucket utilization: %ld\n", sameDeapthHits, difLayerHits, misses, bucketUtil);
        sameDeapthHits = 0;
        difLayerHits = 0;
        misses = 0;
        bucketUtil = 0;
        //iter = 0;
        currLayer++;
        if(currLayer > 42) break;
    }
    printf("total iter over all: %ld\n", iter);
}

int main() {

    wanted = fix_uint(wanted);
    startPos = fix_uint(startPos);

    //alocating the cache
    int casheSize = 16;
    casheArr = calloc((1 << casheSize), 8);
    casheDeapthArr = calloc((1 << casheSize), 1);
    casheMask = (1 << casheSize) - 1;

    search();
}
