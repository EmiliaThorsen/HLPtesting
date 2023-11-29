#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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


uint64_t wanted = 0x3141592653589793; //0x77239AB34567877E 0x0022002288AA88AA 0x1111222233334444
uint8_t arr1[65536];
uint8_t arr2[65536];
uint8_t arr3[65536];
uint8_t arr4[65536];
int total1 = 0;
int total2 = 0;
int total3 = 0;
int total4 = 0;

uint16_t layerConf[800];
uint8_t fineAsLastLayer[800];
int layerCount = 0;


void distsearchthing(int dist) {
    for(uint64_t input = 0; input < 65536; input++) {
        for(int conf = 0; conf < layerCount; conf++) {
            if(dist == 0) {
                if(fineAsLastLayer[conf] == 0) continue;
            }
            uint64_t output = layer(input, layerConf[conf]);
            if(arr1[output & 0xFFFF] == dist) {
                if(arr1[input] == 100) {
                    arr1[input] = dist + 1;
                    total1++;
                }
            }
            if(arr2[output & 0xFFFF] == dist) {
                if(arr2[input] == 100) {
                    arr2[input] = dist + 1;
                    total2++;
                }
            }
            if(arr3[output & 0xFFFF] == dist) {
                if(arr3[input] == 100) {
                    arr3[input] = dist + 1;
                    total3++;
                }
            }
            if(arr4[output & 0xFFFF] == dist) {
                if(arr4[input] == 100) {
                    arr4[input] = dist + 1;
                    total4++;
                }
            }
        }
    }
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


uint64_t startPos = 0x0123456789ABCDEF;

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



int main() {
    int group = getGroup(wanted);
    uint64_t layerSet[800];
    for(int conf = 0; conf < 1536; conf++) {
        uint64_t output = layer(startPos, conf);
        if(output == startPos) continue;
        if(getGroup(output) < group) continue;
        for(int i = 0; i < layerCount; i++) if(layerSet[i] == output) goto skip;
        layerSet[layerCount] = output;
        layerConf[layerCount] = conf;
        fineAsLastLayer[layerCount] = unionCheck(output);
        layerCount++;
        skip: continue;
    }
    for(uint64_t i = 0; i < 65536; i++) { //set all distances very high
        arr1[i] = 100;
        arr2[i] = 100;
        arr3[i] = 100;
        arr4[i] = 100;
    }
    arr1[(wanted      ) & 0xFFFF] = 0;
    arr2[(wanted >> 16) & 0xFFFF] = 0;
    arr3[(wanted >> 32) & 0xFFFF] = 0;
    arr4[(wanted >> 48) & 0xFFFF] = 0;

    int prevTot = 0;
    int distFuncDeapth = 0;
    while (1) {
        distsearchthing(distFuncDeapth); //calculate everything 1 away from deapth
        printf("dist search, total1:%d total2:%d total3:%d total4:%d\n", total1, total2, total3, total4);
        int total = total1 + total2 + total3 + total4;
        if(total == prevTot) break;
        prevTot = total;
        distFuncDeapth++;
    }
}
