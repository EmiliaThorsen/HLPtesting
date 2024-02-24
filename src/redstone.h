#ifndef REDSTONE_H
#define REDSTONE_H
#include <stdint.h>
#include "arg_global.h"

struct precomputed_hex_layer {
    uint64_t map;
    uint64_t* next_layer_luts;
    struct precomputed_hex_layer** next_layers;
    uint16_t config;
    uint16_t next_layer_count;
};


/* apply a layer to the map
 * config: 00000mmmaaaabbbb
 *      mmm: mode; 0 = c/c, 1 = c/s, 2 = s/c, 3 = s/s, 4 = rotated c/s
 *      aaaa: first barrel value (the one for the comparator that points
 *      directly into another comparator)
 *      bbbb: second barrel value
 */
extern uint64_t hex_layer64(uint64_t map, uint16_t config);


/* get precomputed layers
 * 
 * returns an identity layer, which by definition can be followed by any valid
 * layer, as given by the next_layers and next_layer_luts elements. this can be
 * followed recursively to always only check unique pairs of layers
 */
extern struct precomputed_hex_layer* precompute_hex_layers(int group);

/* free precomputed hex layers
 *
 * when a set of hex layers are precomputed, they get saved for future use in
 * case they are needed again. this function frees all of them.
 */
extern void free_precomputed_hex_layers();

struct arg_settings_redstone {
    struct arg_settings_global* global;
};

extern struct argp argp_redstone;

#endif
