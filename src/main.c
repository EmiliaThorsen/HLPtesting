#include <stdio.h>
#include <stdlib.h>
#include "HlpSolve.h"
#include <argp.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>

int global_verbosity;

void printSearch(char* map, int maxDepth, int accuracy) {
    uint16_t result[32];
    printf("searching for %s\n", map);
    int length = solve(map, result, maxDepth, accuracy);

    if (length > maxDepth) {
        printf("no result found\n");
    } else {
        printf("result found, length %d: ", length);
        for (int i=0; i<length; i++) {
            printf("\t%03hx", result[i]);
        }
        printf("\n");
    }
}

uint64_t randUint64() {
    uint64_t result = 0;
    for (int i = 0; i < 8; i++) result = (result << 8) | (rand() & 0xff);
    return result;
}

uint64_t randHexKPerm(int n, int k) {
    uint64_t result = 0;
    uint16_t used = 0;
    for (int i=n; i > n-k; i--) {
        int value = rand() % i;
        value = _tzcnt_u16(_pdep_u32(1 << value, ~used));
        used |= 1 << value;
        result = result << 4 | value;
    }
    return result ;
}

uint64_t randHexPerm(int length) {
    return randHexKPerm(length, length);
}


char hexDigit(int value) {
    if (value < 10) return value + '0';
    return value - 10 + 'a';
}

void randomizeMap(char* dest, int group) {
    dest[16] = 0;
    if (!group) {
        for (int i=0; i<16; i++) {
            dest[i] = hexDigit(rand() % 16);
        }
        return;
    }

    // technically should get a random combination, but a kperm is
    // easier and has the same effect
    uint64_t kperm = randHexKPerm(16, group);
    uint64_t map = kperm;

    // fill in the rest of values
    for (int i=0; i < 16-group; i++) {
        int index = rand() % group;
        int digit = (kperm >> index * 4) & 15;
        map = (map << 4) | digit;
    }

    uint64_t shufflePerm = randHexPerm(16);
    // shuffle to make the beginning not a kperm
    for (int i=0; i < 16; i++) {
        int index = shufflePerm & 15;
        shufflePerm >>= 4;
        dest[i] = hexDigit((map >> index*4) & 15);
    }
}

void randomSearch(int count, int group, int maxDepth, int accuracy) {
    for (int i=0; i<count; i++) {
        char map[17];
        randomizeMap(map, group);
        printSearch(map, maxDepth, accuracy);
    }
}


enum LONG_OPTIONS {
    OPTION_ACCURACY=1000,
    OPTION_CACHE,
    OPTION_MAX_DEPTH,
    OPTION_RANDOM_SEARCH,
    OPTION_RANDOM_SEARCH_GROUP,
    OPTION_RANDOM_SEED
};

const char doc[] = "Find an HLP chain for the given function.";

struct arg_settings {
    int verbosity;
    enum SearchAccuracy accuracy;
    int randomSearchCount;
    int randomSearchGroup;
    int randomSeed;
    int maxDepth;
    char* map;
};

char** appendStr(char** str1, char* str2) { 
    if (!str2) return str1;

    if (*str1) {
        char* newStr = malloc(strlen(*str1) + strlen(str2));
        strcpy(newStr, *str1);
        if (str2) strcat(newStr, str2);
        free(*str1);
        *str1 = newStr;
    } else {
        *str1 = malloc(strlen(str2));
        strcpy(*str1, str2);
    }

    return str1;
}

static struct argp_option options[] = {
    { "verbose", 'v', "LEVEL", OPTION_ARG_OPTIONAL, "Increase or set verbosity" },
    { "quiet", 'q', 0, 0, "Suppress additional info" },
    { "cache", OPTION_CACHE, "N", 0, "Set the cache size to 2**N bytes. default: 26 (64MB)" },
    { "max-length", OPTION_MAX_DEPTH, "N", 0, "Limit results to chains up to N layers long" },
    { "accuracy", OPTION_ACCURACY, "LEVEL", 0, "Set search accuracy from -1 to 2, 0 being normal, 2 being perfect" },
    { "fast", 'f', 0, 0, "Equivilant to --accuracy -1" },
    { "perfect", 'p', 0, 0, "Equivilant to --accuracy 2" },
    { "random", OPTION_RANDOM_SEARCH, "N", 0, "Search N random cases instead of the desired map" },
    { "random-group", OPTION_RANDOM_SEARCH_GROUP, "N", 0, "Only check cases with N unique outputs when searching randomly, 0 for any. default: 0" },
    { "seed", OPTION_RANDOM_SEED, "SEED", 0, "Set the random seed" },
    { 0 }
};

static error_t parse_opt(int key, char* arg, struct argp_state *state) {
    struct arg_settings* settings = state->input;
    switch (key) {
        case 'v':
            if (arg)
                settings->verbosity = atoi(arg);
            else
                settings->verbosity++;
            break;
        case 'q':
            settings->verbosity = 0;
            break;
        case OPTION_ACCURACY:
            int level = atoi(arg);
            if (level < -1 || level > 2)
                argp_error(state, "%s is not a valid accuracy", arg);
            else
                settings->accuracy = level;
            break;
        case 'f':
            return parse_opt(OPTION_ACCURACY, "-1", state);
        case 'p':
            return parse_opt(OPTION_ACCURACY, "2", state);
        case OPTION_RANDOM_SEARCH:
            settings->randomSearchCount = atoi(arg);
            break;
        case OPTION_RANDOM_SEARCH_GROUP:
            int group = atoi(arg);
            if (group < 1 || group > 16)
                argp_error(state, "%s unique outputs is impossible", arg);
            else
                settings->randomSearchGroup = group;
            break;
        case OPTION_RANDOM_SEED:
            settings->randomSeed = atoi(arg);
            break;
        case OPTION_MAX_DEPTH:
            settings->maxDepth = atoi(arg);
            break;
        case OPTION_CACHE:
            hlpSetCacheSize(atoi(arg) - 4);
        case ARGP_KEY_ARG:
            appendStr(&(settings->map), arg);
            break;
        case ARGP_KEY_INIT:
            settings->verbosity = 1;
            settings->accuracy = ACCURACY_NORMAL;
            settings->map = 0;
            settings->randomSearchCount = 0;
            settings->randomSearchGroup = 0;
            settings->maxDepth = 31;
            settings->randomSeed = time(NULL);
            break;
    }
    return 0;
}

static struct argp argp = {
    options,
    parse_opt,
    "MAP",
    doc
};


int main(int argc, char** argv) {
    struct arg_settings settings;
    error_t argpError = argp_parse(&argp, argc, argv, 0, 0, &settings);
    if (argpError) return argpError;

    srand(settings.randomSeed);

    if (settings.randomSearchCount == 0)
        printSearch(settings.map, settings.maxDepth, settings.accuracy);
    else {
        randomSearch(settings.randomSearchCount, settings.randomSearchGroup, settings.maxDepth, settings.accuracy);
    }
    return 0;
}
