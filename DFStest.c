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

 
//precomputed layer lookup tables
uint8_t *layers[800];
uint16_t layerConf[800];
uint8_t fineAsLastLayer[800];
uint16_t nextValidLayers[800*800];
uint8_t layerGroups[800*800];
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
void precomputeLayers() {
    printf("starting layer precompute\n");
    uint64_t totalNextLayers[800*800];
    for(int conf = 0; conf < 1536; conf++) {
        uint64_t output = layer(startPos, conf);
        if(output == startPos) continue;
        for(int i = 0; i < layerCount; i++) if(totalNextLayers[i] == output) goto skip;
        totalNextLayers[layerCount] = output;
        layerConf[layerCount] = conf;
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



int currLayer = 0;

//fast lut based layer, also does a check to see if the output is invalid and inposible to use to find goal
uint64_t fastLayer(uint64_t input, int configuration, uint8_t *goal) {
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
int fastLastLayerSearch(uint64_t startPos, int prevLayerConf, uint8_t *goal) {
    for(int conf = 0; conf < nextValidLayersSize[prevLayerConf]; conf++) {
        int l = nextValidLayers[prevLayerConf * 800 + conf];
        uint8_t *specificLayer = layers[l];
        for(int i = 0; i < 16; i++) {
            if(specificLayer[(startPos >> (i << 2)) & 15] != goal[i]) goto wrong;
        }
        return 1;
        wrong: continue;
    }
    return 0;
}

int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

int abs(x) {
    if(x < 0) return -x;
    return x;
}

//aproximate distance function using precomputed tables
int distCheck(uint64_t input, uint8_t *goal) {
    int arr[16];
    for(int i = 0; i < 16; i++) { //zip(start, goal)
        arr[i] = (((input >> (i << 2)) & 15) << 4) | (goal[i] & 15);
    }
    qsort(arr, 16, sizeof(int), cmpfunc); //sorted()
    int total = 0;
    for(int i = 0; i < 15; i++) {//"big jump" magic
        if(abs((arr[i] & 15) - (arr[i + 1] & 15)) > abs(((arr[i] >> 4) & 15) - ((arr[i + 1] >> 4) & 15))) total++;
    }
    return total;
}


struct output {
    int i;
    uint64_t output;
    int group;
};


//main dfs recursive search function
int dfs(uint64_t startPos, int deapth, int prevLayerConf, int safeMode, uint8_t *goal) {
    if(deapth == currLayer - 1) return fastLastLayerSearch(startPos, prevLayerConf, goal);
    struct output outputs[787];
    int count = 0;
    int minG = 16;
    for(int conf = 0; conf < nextValidLayersSize[prevLayerConf]; conf++) {
        int i = nextValidLayers[prevLayerConf * 800 + conf];
        uint64_t output = fastLayer(startPos, i, goal);
        if(output == 0) continue;

        if((distCheck(output, goal) - safeMode) > (currLayer - deapth - 1)) continue;

        int group = getGroup(output);
        outputs[count].output = output;
        outputs[count].i = i;
        outputs[count].group = group;
        count++;
        if(group < minG) minG = group;
    }
    if(safeMode == 0) {
        int overwriteIndex = 0;
        for (int i = 0; i < count; i++) {
            if((outputs[i].group > minG)) continue;
            outputs[overwriteIndex] = outputs[i];
            overwriteIndex++;
        }
        count = overwriteIndex;

    }
    for(int i = 0; i < count; i++) {
        //call next layers
        if(dfs(outputs[i].output, deapth + 1, outputs[i].i, safeMode, goal)) {
            return 1;
        }
    }
    return 0;
}

uint64_t randUint64() {
  uint64_t r = 0;
  for (int i=0; i<64; i++) {
    r = r*2 + rand()%2;
  }
  return r;
}

int totalSearched = 0;
int totalSearchedLayers = 0;
int highestLayerCount = 0;
//main search loop
void search() {
    printf("starting search!\n");
    #pragma omp parallel
    {
        while (1) {
            uint64_t wanted = randUint64();
            uint8_t goal[16];
            for(int i = 0; i < 16; i++) goal[i] = (wanted >> (i * 4)) & 15;
            currLayer = distCheck(startPos, goal) - 1;
            while (1) {
                currLayer++;
                if(currLayer > 42) break;
                if(dfs(startPos, 0, 799, 0, goal)) break;
            }
            if(currLayer > 18) {
                currLayer = 0;
                while (1) {
                    currLayer++;
                    if(currLayer > 42) break;
                    if(dfs(startPos, 0, 799, 1, goal)) break;
                }
            }
            #pragma omp critical
            {
                totalSearched++;
                totalSearchedLayers += currLayer;
                if(highestLayerCount < currLayer) highestLayerCount = currLayer;
                printf("found %16lx in layer: %d, total searched cases: %d, avrage case dificulity: %f, hardest yet case took: %d layers\n", wanted, currLayer, totalSearched, (double)totalSearchedLayers / totalSearched, highestLayerCount);
                currLayer = 0;
            }
        }
    }
}


int main() {
    //seed rng to time
    srand(time(NULL));
    //srand(69420);

    printf("starting precompute");
    precomputeLayers();
    search();
}
