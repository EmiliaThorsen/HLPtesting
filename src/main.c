#include <stdio.h>
#include <stdlib.h>
#include "HlpSolve.h"


void printSearch(uint64_t map) {
    uint16_t result[32];
    printf("searching for %016lx\n", map);
    int length = search(map, result, 32);

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
}
