#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <x86intrin.h>

extern uint64_t apply_mapping(uint64_t input, uint8_t* map);
extern uint64_t apply_and_check(uint64_t input, uint8_t* map, int threshhold);
extern void bitonic_sort16x8(uint8_t* array);
extern uint64_t array_to_uint(uint8_t* array);
extern void pretty_uint_to_array(uint64_t uint, uint8_t* array);
extern void store_mapping(uint64_t map, uint8_t* location);

//naive implementation of a layer
int max(int a, int b) {return a > b ? a : b;}

int compMode(int back, int side) {return (back >= side) * back;}

int subMode(int back, int side) {
    int dif = back - side;
    if(dif > 0) return dif;
    return 0;
}

uint64_t layer(uint64_t input, int configuration) {
    int ss1 = (configuration) & 15;
    int ss2 = (configuration >> 4) & 15;
    long output = 0;
    for(int i = 0; i < 64; i += 4) {
        int ss = (input >> i) & 15;
        int result = 0;
        switch ((configuration >> 8) & 7) {
            case 0: result = max(compMode(ss1, ss), compMode(ss, ss2)); break;
            case 1: result = max(subMode(ss1, ss) , compMode(ss, ss2)); break;
            case 2: result = max(compMode(ss1, ss), subMode(ss, ss2) ); break;
            case 3: result = max(subMode(ss1, ss) , subMode(ss, ss2) ); break;
            case 4: result = max(subMode(ss1, ss) , compMode(ss2, ss)); break;
            case 5: result = max(compMode(ss1, ss), subMode(ss2, ss) ); break;
        }
        output = output | ((long)(result) << i);
    }
    return output;
}


//uint64_t startPos = 0x0123456789ABCDEF; //DO NOT CHANGE
uint64_t startPos = 0xf7e6d5c4b3a29180; //how bout i do anyway

//the goal you want to search to
uint64_t wanted = 0x3141592653589793; //0x77239AB34567877E 0x0022002288AA88AA 0x1122334455667788 0x1111222233334444 0x91326754CDFEAB98

uint8_t goal[16]; //goal in array form instead of packed nibbles in uint
uint64_t goal_uint; //trust me

//precomputed layer lookup tables
uint8_t *layers[800];
uint16_t layerConf[800];
uint8_t fineAsLastLayer[800];
uint16_t nextValidLayers[800*800];
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

//precompute of layers into lut, proceding layers deduplicated for lower branching, and distance estimate table
void precomputeLayers(int group) {
    printf("starting layer precompute\n");
    uint64_t totalNextLayers[800*800];
    for(int conf = 0; conf < 1536; conf++) {
        uint64_t output = layer(startPos, conf);
        if(output == startPos) continue;
        if(getGroup(output) < group) continue;
        for(int i = 0; i < layerCount; i++) if(totalNextLayers[i] == output) goto skip;
        totalNextLayers[layerCount] = output;
        layerConf[layerCount] = conf;
        nextValidLayers[799 * 800 + layerCount] = layerCount;
        uint8_t *specificLayer = malloc(16);
        store_mapping(output, specificLayer);
        /* for(int i = 0; i < 16; i++) specificLayer[15 - i] = (output >> (i * 4)) & 15; */
        layers[layerCount] = specificLayer;
        layerCount++;
        skip: continue;
    }
    nextValidLayersSize[799] = layerCount;
    long totalNext = layerCount;
    printf("starting next layer compute\n");
    for(int conf = 0; conf < layerCount; conf++) {
        uint64_t layerOut = apply_mapping(startPos, layers[conf]);
        int nextLayerSize = 0;
        for (int conf2 = 0; conf2 < layerCount; conf2++) {
            uint64_t nextLayerOut = apply_mapping(layerOut, layers[conf2]);
            if(nextLayerOut == startPos) continue;
            if(getGroup(nextLayerOut) < group) continue;
            for(int i = 0; i < totalNext; i++) if(totalNextLayers[i] == nextLayerOut) goto nextSkip;
            totalNextLayers[totalNext] = nextLayerOut;
            totalNext++;
            nextValidLayers[conf * 800 + nextLayerSize] = conf2;
            nextLayerSize++;
            nextSkip: continue;
        }
        nextValidLayersSize[conf] = nextLayerSize;
    }
    printf("layers computed:%d, total next layers:%ld\n", layerCount, totalNext - layerCount);
}


long iter = 0;

int currLayer = 1;

int mismatches;

uint64_t avx_rerun(uint64_t input, uint8_t* map) {
    apply_mapping(input, map);
}

//fast lut based layer, also does a check to see if the output is invalid and inposible to use to find goal
uint64_t fastLayer(uint64_t input, int configuration, int threshhold) {
    iter++;
    return apply_and_check(input, layers[configuration], threshhold);
}



//faster implementation of searching over the last layer while checking if you found the goal, unexpectedly big optimization
int fastLastLayerSearch(uint64_t startPos, int prevLayerConf) {
    for(int conf = 0; conf < nextValidLayersSize[prevLayerConf]; conf++) {
        int l = nextValidLayers[prevLayerConf * 800 + conf];
        uint8_t *specificLayer = layers[l];
        iter++;
        if (apply_mapping(startPos, layers[l]) != goal_uint) continue;
        printf("deapth: %d configuration: %03hx\n", currLayer - 1, layerConf[l]);
        return 1;
    }
    return 0;
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

//main dfs recursive search function
int dfs(uint64_t startPos, int deapth, int prevLayerConf) {
    if(deapth == currLayer - 1) return fastLastLayerSearch(startPos, prevLayerConf);
    int dedupeArrSize = 0;
    for(int conf = 0; conf < nextValidLayersSize[prevLayerConf]; conf++) {
        int i = nextValidLayers[prevLayerConf * 800 + conf];
        uint64_t output = fastLayer(startPos, i, currLayer - deapth - 1);
        if (output == 0) continue;

        /* if(check_mapping_fast(output)) continue; */

        //cashe check
        if(casheCheck(output, deapth)) continue;
        //call next layers
        if(dfs(output, deapth + 1, i)) {
            printf("deapth: %d configuration: %03hx\n", deapth, layerConf[i]);
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
    pretty_uint_to_array(wanted, goal);
    goal_uint = array_to_uint(goal);
    /* for(int i = 0; i < 16; i++) goal[i] = (wanted >> (i * 4)) & 15; */
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
    //alocating the cache
    int casheSize = 25;
    casheArr = calloc((1 << casheSize), 8);
    casheDeapthArr = calloc((1 << casheSize), 1);
    casheMask = (1 << casheSize) - 1;

    search();
}
