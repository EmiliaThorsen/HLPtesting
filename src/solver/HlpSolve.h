#ifndef HLP_SOLVE
#define HLP_SOLVE
#include "../arg_global.h"
#include "../redstone.h"
#include <stdint.h>

enum SearchAccuracy { ACCURACY_REDUCED=-1, ACCURACY_NORMAL, ACCURACY_INCREASED, ACCURACY_PERFECT };
enum SolveConfigError { HLP_ERROR_BLANK=1, HLP_ERROR_NULL, HLP_ERROR_MALFORMED, HLP_ERROR_TOO_LONG };
enum HlpSolveType { HLP_SOLVE_TYPE_EXACT, HLP_SOLVE_TYPE_PARTIAL, HLP_SOLVE_TYPE_RANGED };

typedef struct hlp_request_s {
    uint64_t mins;
    uint64_t maxs;
    enum HlpSolveType solveType;
    enum SolveConfigError error;
} hlp_request_t;

// the start position, or at least the pretty one that can be used outside the solver
extern const uint64_t hlpStartPos;

extern const uint64_t broadcastH16; // 0x1111...

extern int hlpSolveVerbosity;

/* search for a solution for the given map
 * returns length of chain
 */
int solve(hlp_request_t request, uint16_t* outputChain, int maxDepth, enum SearchAccuracy accuracy);

/* parse the string into a solve request
 */
hlp_request_t parseHlpRequestStr(char* str);

void printHlpRequest(hlp_request_t request);

void printHlpMap(uint64_t map);

/* safely set the cache size (changes will apply on next search)
 * cache will have 2 ** size entries, with 16 byte entries
 * default: 22
 */
void hlpSetCacheSize(int size);

// access the layer information
uint16_t getLayerConf(int group, int layerId);
uint16_t getNextValidLayerId(int group, int prevLayerId, int index);
uint16_t getNextValidLayerSize(int group, int layerId);

void hlpPrintSearch(char* map);

uint64_t applyChain(uint64_t start, uint16_t* chain, int length);

void printChain(uint16_t* chain, int length);

struct arg_settings_solver_hex {
    struct arg_settings_global* global;
    struct arg_settings_redstone settings_redstone;
};

extern struct argp argp_solver_hex;

#endif
