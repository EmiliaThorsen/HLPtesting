#include <stdio.h>
#include <stdlib.h>
#include "HlpSolve.h"

uint64_t randUint64() {
    uint64_t result = 0;
    for (int i = 0; i < 8; i++) result = (result << 8) | (rand() & 0xff);
    return result;
}

void printSearch(uint64_t map) {
    uint16_t result[32];
    printf("searching for %016lx\n", map);
    int length = solve(map, result, ACCURACY_NORMAL);

    if (length == -1) {
        printf("\tno result found\n");
    } else {
        printf("\tresult found, length %d: ", length);
        for (int i=0; i<length; i++) {
            printf("\t%03hx", result[i]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {

    for (int i = 1; i < argc; i++) {
        uint64_t to_find = strtoull(argv[i], 0, 16); // this is totally safe, right?
        printSearch(to_find);
    }
    return 0;

    for (int i = 0; i < 50; i++) {
        printf("%d\n\e[A", i);
        solve(randUint64(), 0, ACCURACY_NORMAL);
    }
}
