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

uint16_t layer(uint16_t input, int configuration) {
    int ss1 = (configuration) & 15;
    int ss2 = (configuration >> 4) & 15;
    uint16_t output = 0;
    for(int i = 0; i < 16; i++) {
        if((input & (1 << i)) == 0) continue;
        int ss = i;
        int result = 0;
        switch ((configuration >> 8) & 7) {
            case 0: result = max(compMode(ss1, ss), compMode(ss, ss2)); break;
            case 1: result = max(subMode(ss1, ss) , compMode(ss, ss2)); break;
            case 2: result = max(compMode(ss1, ss), subMode(ss, ss2) ); break;
            case 3: result = max(subMode(ss1, ss) , subMode(ss, ss2) ); break;
            case 4: result = max(subMode(ss1, ss) , compMode(ss2, ss)); break;
            case 5: result = max(compMode(ss1, ss), subMode(ss2, ss) ); break;
        }
        output |= ((uint16_t)(1) << result);
    }
    return output;
}


uint64_t wanted = 0x314159265358979; //0x77239AB34567877E 0x0022002288AA88AA 0x1111222233334444
uint8_t arr1[65536];
int total1 = 0;

void distsearchthing(int dist) {
    for(uint64_t input = 0; input < 65536; input++) {
        for(int conf = 0; conf < 1536; conf++) {
            uint64_t output = layer(input, conf);
            if(arr1[output & 0xFFFF] == dist) {
                if(arr1[input] > dist + 1) {
                    arr1[input] = dist + 1;
                    total1++;
                }
            }
        }
    }
}

uint16_t getBitFeild(uint64_t x) {
    uint16_t bitFeild = 0;
    for(int i = 0; i < 16; i++) {
        int ss = (x >> (i * 4)) & 15;
        bitFeild |= 1 << ss;
    }
    return bitFeild;
}

int main() {
    printf("test: %d\n", getBitFeild(wanted));
    printf("test: %d\n", layer(getBitFeild(wanted), 0));
    for(uint64_t i = 0; i < 65536; i++) {
        arr1[i] = 100;

    }
    arr1[getBitFeild(wanted)] = 0;
    distsearchthing(0);
    printf("total:%d\n", total1);
    total1 = 0;
    distsearchthing(1);
    printf("total:%d\n", total1);
    total1 = 0;
    distsearchthing(2);
    printf("total:%d\n", total1);
    total1 = 0;
    distsearchthing(3);
    printf("total:%d\n", total1);
    total1 = 0;
    distsearchthing(4);
    printf("total:%d\n", total1);
    total1 = 0;
    distsearchthing(4);
    printf("total:%d\n", total1);
}
