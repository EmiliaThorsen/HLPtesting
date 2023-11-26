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
    for(int i = 0; i < 16; i += 4) {
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


uint64_t wanted = 0x1111222233334444; //0x77239AB34567877E 0x0022002288AA88AA 0x314159265358979
uint8_t arr1[65536];
uint8_t arr2[65536];
uint8_t arr3[65536];
uint8_t arr4[65536];
int total1 = 0;
int total2 = 0;
int total3 = 0;
int total4 = 0;

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
            if(arr2[output & 0xFFFF] == dist) {
                if(arr2[input] > dist + 1) {
                    arr2[input] = dist + 1;
                    total2++;
                }
            }
            if(arr3[output & 0xFFFF] == dist) {
                if(arr3[input] > dist + 1) {
                    arr3[input] = dist + 1;
                    total3++;
                }
            }
            if(arr4[output & 0xFFFF] == dist) {
                if(arr4[input] > dist + 1) {
                    arr4[input] = dist + 1;
                    total4++;
                }
            }
        }
    }
}


int main() {
    for(uint64_t i = 0; i < 65536; i++) {
        arr1[i] = 100;
        arr2[i] = 100;
        arr3[i] = 100;
        arr4[i] = 100;

    }
    arr1[(wanted      ) & 0xFFFF] = 0;
    arr2[(wanted >> 16) & 0xFFFF] = 0;
    arr3[(wanted >> 32) & 0xFFFF] = 0;
    arr4[(wanted >> 48) & 0xFFFF] = 0;
    distsearchthing(0);
    printf("total1:%d total2:%d total3:%d total4:%d\n", total1, total2, total3, total4);
    total1 = 0;
    total2 = 0;
    total3 = 0;
    total4 = 0;
    distsearchthing(1);
    printf("total1:%d total2:%d total3:%d total4:%d\n", total1, total2, total3, total4);
    total1 = 0;
    total2 = 0;
    total3 = 0;
    total4 = 0;
    distsearchthing(2);
    printf("total1:%d total2:%d total3:%d total4:%d\n", total1, total2, total3, total4);
    total1 = 0;
    total2 = 0;
    total3 = 0;
    total4 = 0;
    distsearchthing(3);
    printf("total1:%d total2:%d total3:%d total4:%d\n", total1, total2, total3, total4);
    total1 = 0;
    total2 = 0;
    total3 = 0;
    total4 = 0;
    distsearchthing(4);
    printf("total1:%d total2:%d total3:%d total4:%d\n", total1, total2, total3, total4);
    total1 = 0;
    total2 = 0;
    total3 = 0;
    total4 = 0;
    distsearchthing(4);
    printf("total1:%d total2:%d total3:%d total4:%d\n", total1, total2, total3, total4);
}
