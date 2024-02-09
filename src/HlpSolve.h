#ifndef HLP_SOLVE
#define HLP_SOLVE
#include <stdint.h>



// the start position, or at least the pretty one that can be used outside the solver
extern const uint64_t hlpStartPos;

extern const uint64_t broadcastH16; // 0x1111...

/* apply a layer to the map
 * config: 00000mmmaaaabbbb
 *      mmm: mode; 0 = c/c, 1 = c/s, 2 = s/c, 3 = s/s, 4 = rotated c/s
 *      aaaa: first barrel value (the one for the comparator that points
 *      directly into another comparator)
 *      bbbb: second barrel value
 */
extern uint64_t layer(uint64_t map, uint16_t config);

/* search for a solution for the given map
 * returns length of chain
 */
int search(uint64_t m, uint16_t* outputChain, int maxDepth);

/* safely set the cache size (changes will apply on next search)
 * cache will have 2 ** size entries, with 16 byte entries
 * default: 22
 */
void hlpSetCacheSize(int size);

// access the layer information
uint16_t getLayerConf(int group, int layerId);
uint16_t getNextValidLayerId(int group, int prevLayerId, int index);
uint16_t getNextValidLayerSize(int group, int layerId);


#endif
