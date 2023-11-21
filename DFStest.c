#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>

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

uint8_t goal[16];
uint64_t wanted = 0x3141592653589793; //0x77239AB34567877E 0x0022002288AA88AA 0x1111222233334444


uint8_t *layers[800];
uint16_t layerConf[800];
uint16_t nextValidLayers[800*800];
int nextValidLayersSize[800];
int layerCount = 0;

uint64_t startPos = 0x0123456789ABCDEF;

void precomputeLayers(int group) {
    printf("starting layer precompute\n");
    uint64_t layerSet[800];
    for(int conf = 0; conf < 1536; conf++) {
        uint64_t output = layer(startPos, conf);
        if(output == startPos) continue;
        if(getGroup(output) < group) continue;
        for(int i = 0; i < layerCount; i++) if(layerSet[i] == output) goto skip;
        layerSet[layerCount] = output;
        layerConf[layerCount] = conf;
        nextValidLayers[799 * 800 + layerCount] = layerCount;
        uint8_t *specificLayer = malloc(16);
        for(int i = 0; i < 16; i++) specificLayer[15 - i] = (output >> (i * 4)) & 15;
        layers[layerCount] = specificLayer;
        layerCount++;
        skip: continue;
    }
    nextValidLayersSize[799] = layerCount;
    uint64_t totalNextLayers[800*800];
    long totalNext = 0;
    printf("starting next layer compute\n");
    for(int conf = 0; conf < layerCount; conf++) {
        uint64_t layerOut = layer(startPos, layerConf[conf]);
        int nextLayerSize = 0;
        for (int conf2 = 0; conf2 < layerCount; conf2++) {
            uint64_t nextLayerOut = layer(layerOut, layerConf[conf2]);
            if(nextLayerOut == startPos) continue;
            if(nextLayerOut == layerOut) continue;
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
    printf("layers computed:%d\n", layerCount);
}


uint64_t *casheArr;
uint8_t *casheDeapthArr;
uint64_t casheMask;

long iter = 0;

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

int currLayer = 1;

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

long sameDeapthHits = 0;
long difLayerHits = 0;
long misses = 0;
long bucketUtil = 0;

int casheCheck(uint64_t output, int deapth) {
    uint64_t h = output * 0x9E3779B97F4A7C15L;
    uint32_t pos = (h ^= h >> 32) & casheMask;
    if(casheArr[pos] == output & casheDeapthArr[pos] <= deapth) {
        if(casheDeapthArr[pos] == deapth) {
            sameDeapthHits++;
            return 1;
        }
        difLayerHits++;
        return 1;
    }
    if(casheArr[pos] != 0) {
        misses++;
    } else {
        bucketUtil++;
    }
    casheArr[pos] = output;
    casheDeapthArr[pos] = deapth;
    return 0;
}


int dfs(uint64_t startPos, int deapth, int prevLayerConf) {
    if(deapth == currLayer - 1) return fastLastLayerSearch(startPos, prevLayerConf);
    for(int conf = 0; conf < nextValidLayersSize[prevLayerConf]; conf++) {
        int i = nextValidLayers[prevLayerConf * 800 + conf];
        uint64_t output = fastLayer(startPos, i);
        if(output == 0) continue;
        //cache duplicate removal check
        if(casheCheck(output, deapth)) continue;
        //call next layers
        if(dfs(output, deapth + 1, i)) {
            printf("deapth: %d configuration: %03hx\n", deapth, layerConf[i]);
            return 1;
        }
        if(deapth == 0 & currLayer > 6) printf("done:%d/%d\n", conf, nextValidLayersSize[prevLayerConf]);
    }
    return 0;
}


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

    int casheSize = 25;
    casheArr = calloc((1 << casheSize), 8);
    casheDeapthArr = calloc((1 << casheSize), 1);
    casheMask = (1 << casheSize) - 1;

    search();
}
