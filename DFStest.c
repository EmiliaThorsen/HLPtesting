#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <x86intrin.h>

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


uint64_t startPos = 0x0123456789ABCDEF; //DO NOT CHANGE

//the goal you want to search to
uint64_t wanted = 0x3141592653589793; //0x77239AB34567877E 0x0022002288AA88AA 0x1122334455667788 0x1111222233334444

uint8_t goal[16]; //goal in array form instead of packed nibbles in uint

//precomputed layer lookup tables
uint8_t *layers[800];
uint16_t layerConf[800];
uint8_t fineAsLastLayer[800];
uint16_t nextValidLayers[800*800];
int nextValidLayersSize[800];
int layerCount = 0;

//low estimate distance function table compute
uint8_t arr1[65536];
uint8_t arr2[65536];
uint8_t arr3[65536];
uint8_t arr4[65536];
int total1 = 0;
int total2 = 0;
int total3 = 0;
int total4 = 0;

void distsearchthing(int dist) {
    for(int conf = 0; conf < layerCount; conf++) {
        if(dist == 0 & fineAsLastLayer[conf] == 0) continue;
        for(uint64_t input = 0; input < 65536; input++) {
            uint64_t output = layer(input, layerConf[conf]) & 0xFFFF;
            if(arr1[output] == dist & arr1[input] == 100) {
                arr1[input] = dist + 1;
                total1++;
            }
            if(arr2[output] == dist & arr2[input] == 100) {
                arr2[input] = dist + 1;
                total2++;
            }
            if(arr3[output] == dist & arr3[input] == 100) {
                arr3[input] = dist + 1;
                total3++;
            }
            if(arr4[output] == dist & arr4[input] == 100) {
                arr4[input] = dist + 1;
                total4++;
            }
        }
    }
}

//check to see if the set of outputs x encompases the output, ie its a valid layer to use at the end of the search
int unionCheck(uint64_t x) {
    uint16_t bitFeildX = 0;
    for(int i = 0; i < 16; i++) {
        int ss = (x >> (i * 4)) & 15;
        bitFeildX |= 1 << ss;
    }
    uint16_t bitFeildG = 0;
    for(int i = 0; i < 16; i++) {
        int ss = (wanted >> (i * 4)) & 15;
        bitFeildG |= 1 << ss;
    }
    return bitFeildG == (bitFeildG & bitFeildX);
}

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
        fineAsLastLayer[layerCount] = unionCheck(output);
        nextValidLayers[799 * 800 + layerCount] = layerCount;
        uint8_t *specificLayer = malloc(16);
        for(int i = 0; i < 16; i++) specificLayer[15 - i] = (output >> (i * 4)) & 15;
        layers[layerCount] = specificLayer;
        layerCount++;
        skip: continue;
    }
    nextValidLayersSize[799] = layerCount;
    long totalNext = layerCount;
    printf("starting next layer compute\n");
    for(int conf = 0; conf < layerCount; conf++) {
        uint64_t layerOut = layer(startPos, layerConf[conf]);
        int nextLayerSize = 0;
        for (int conf2 = 0; conf2 < layerCount; conf2++) {
            uint64_t nextLayerOut = layer(layerOut, layerConf[conf2]);
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
    memset(arr1, 100, sizeof(arr1));
    memset(arr2, 100, sizeof(arr2));
    memset(arr3, 100, sizeof(arr3));
    memset(arr4, 100, sizeof(arr4));
    arr1[(wanted      ) & 0xFFFF] = 0;
    arr2[(wanted >> 16) & 0xFFFF] = 0;
    arr3[(wanted >> 32) & 0xFFFF] = 0;
    arr4[(wanted >> 48) & 0xFFFF] = 0;
    int prevTot = 0;
    int distFuncDeapth = 0;
    while (1) {
        distsearchthing(distFuncDeapth);
        printf("dist search, total1:%d total2:%d total3:%d total4:%d\n", total1, total2, total3, total4);
        int total = total1 + total2 + total3 + total4;
        if(total == prevTot) break;
        prevTot = total;
        distFuncDeapth++;
    }
}


long iter = 0;

int currLayer = 1;

//fast lut based layer, also does a check to see if the output is invalid and inposible to use to find goal
uint64_t fastLayer(uint64_t input, int configuration) {
    iter++;
    uint8_t mappings[16] = {69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69};
    uint64_t output = 0;
    uint8_t *specificLayer = layers[configuration];
    for(int i = 0; i < 16; i++) {
        uint8_t result = specificLayer[(input >> (i << 2)) & 15];
        if(mappings[result] != goal[i] & mappings[result] != 69) return 0;
        mappings[result] = goal[i];
        output |= ((uint64_t)(result) << (i << 2));
    }
    return output;
}



//faster implementation of searching over the last layer while checking if you found the goal, unexpectedly big optimization
int fastLastLayerSearch(uint64_t startPos, int prevLayerConf) {
    for(int conf = 0; conf < nextValidLayersSize[prevLayerConf]; conf++) {
        int l = nextValidLayers[prevLayerConf * 800 + conf];
        uint8_t *specificLayer = layers[l];
        iter++;
        for(int i = 0; i < 16; i++) {
            if(specificLayer[(startPos >> (i << 2)) & 15] != goal[i]) goto wrong;
        }
        printf("deapth: %d configuration: %03hx\n", currLayer - 1, layerConf[l]);
        return 1;
        wrong: continue;
    }
    return 0;
}

//aproximate distance function using precomputed tables
int distCheck(uint64_t input, int threshhold) {
    if(arr1[(input      ) & 0xFFFF] > threshhold) return 1;
    if(arr2[(input >> 16) & 0xFFFF] > threshhold) return 1;
    if(arr3[(input >> 32) & 0xFFFF] > threshhold) return 1;
    if(arr4[(input >> 48) & 0xFFFF] > threshhold) return 1;
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
        uint64_t output = fastLayer(startPos, i);
        if(output == 0) continue;
        //distance check removal
        if(distCheck(output, currLayer - deapth - 1)) continue;
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
    for(int i = 0; i < 16; i++) goal[i] = (wanted >> (i * 4)) & 15;
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
        iter = 0;
        currLayer++;
        if(currLayer > 42) break;
    }
}


int main() {
    //alocating the cache
    int casheSize = 25;
    casheArr = calloc((1 << casheSize), 8);
    casheDeapthArr = calloc((1 << casheSize), 1);
    casheMask = (1 << casheSize) - 1;

    search();
}
