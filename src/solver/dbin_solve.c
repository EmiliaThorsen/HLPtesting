#include "dbin_solve.h"
#include <stdint.h>
#include "../aa_tree.h"
#include "../redstone.h"
#include "../vector_tools.h"
#include "../cache.h"

struct precomputed_dbin_layer {
    uint32_t map;
    uint16_t config;
};

struct dbin_solve_globals {
    struct __config__ {
        int current_bfs_depth;
        int group;
        int unique_dbin_layers;
        uint8_t* prune_table;
        struct precomputed_dbin_layer* dbin_layers;
    } config;
    struct __output__ {
        uint16_t* chain;
        int length;
    } output;
    struct __stats__ {
    } stats;
};

// array of values if you look at the index in binary, and read it directly as a ternary number
int* bct_half_values = NULL;

static int verbosity;
static int global_max_depth;

/* BCT Increment
 * add 1 to number in binary coded ternary
 */
uint64_t bct_inc(uint64_t x) {
    // inc, with added boost to make sure 2's carry
    x += LO_HALVES_1_64 + 1;
    // any pair of bits that is now 00 remains as is, all others dec
    return x - (((x >> 1) | x) & LO_HALVES_1_64);
}

int bct_any_twos(uint64_t x) {
    return x & HI_HALVES_1_64;
}

int bct_lowest_two(uint64_t x) {
    return _tzcnt_u64(x & HI_HALVES_1_64) / 2;
}

// temporarily using uint8 for simplicity
uint8_t uint4_array_get(uint8_t* array, int index) {
    //return array[index];
    return 15 & (array[index / 2] >> ((index & 1) * 4));
}

void uint4_array_set(uint8_t* array, int index, uint8_t value) {
    //array[index] = value;
    //return;
    int shift = (index % 2) * 4;
    array[index / 2] = (array[index / 2] & (0xf0 >> shift)) | ((value & 15) << shift);
}

uint32_t dbin_exact_prepend_map_packed64(uint64_t map, uint32_t state) {
    return _mm256_movemask_epi8(_mm256_shuffle_epi8(reverse_movmask_256(state), DOUBLE_XMM(unpack_uint_to_xmm(map))));
}

uint16_t dbin_exact_prepend_half_map_packed64(uint64_t map, uint16_t state, int half) {
    return _pext_u32(dbin_exact_prepend_map_packed64(map, _pdep_u32(state, LO_HALVES_8_64 << (half * 8))), LO_HALVES_8_64 << (half * 8));
}

/* partial dbin format: (bit 1 0's) (bit 2 0's) (bit 1 1's) (bit 2 1's)
 * we have a 1 for every bit that needs to be that specific value
 */

uint64_t dbin_partial_unprepend_map_packed64(uint64_t map, uint64_t state) {
    map = little_endian_xmm_to_uint(unpack_uint_to_xmm(map));
    uint64_t result = 0;
    for (int i = 0; i < 16; i++) {
        result |= ((state >> i) & BROADCAST_4x(16, 1)) << ((map >> (i * 4)) & 15);
    }
    return result;
}

int get_dbin_exact_group(uint32_t mask) {
    uint16_t first_bits = mask & UINT16_MAX;
    uint16_t second_bits = mask >> 16;
    return ((first_bits & second_bits) != 0) +
        ((first_bits & ~second_bits) != 0) +
        ((~first_bits & second_bits) != 0) +
        ((~first_bits & ~second_bits) != 0);
}

#define DBIN_CONFIG_COUNT (16 * 16 * 4)
#define PRETABLE_SIZE (UINT16_MAX + 1)
#define PRUNE_TABLE_ENTRY_COUNT 43046721 // 3 ** 16
#define PRUNE_TABLE_BYTES PRUNE_TABLE_ENTRY_COUNT
//#define PRUNE_TABLE_BYTES (PRUNE_TABLE_ENTRY_COUNT / 2 + 1) 

int in_list(struct precomputed_dbin_layer* array, int length, uint32_t value) {
    for (int i = 0; i < length; i++) { 
        if (array[i].map == value) return 1;
    }
    return 0;
}

int precompute_dbin_layers(struct precomputed_dbin_layer* dest, int group) {
    uint64_t flag;
    aatree_node* unique_layers_tree = NULL;
    int layer_count = 0;
    for (int config = 0; config < DBIN_CONFIG_COUNT; config++) {
        uint32_t output = dbin_layer128(SHUFB_IDENTITY_128, config);
        if (aa_tree_search(unique_layers_tree, output)) continue;
        /* if (in_list(dest, layer_count, output)) continue; */
        unique_layers_tree = aa_tree_insert(output, unique_layers_tree, &flag);
        dest[layer_count] = (struct precomputed_dbin_layer) { output, config };
        layer_count++;
    }
    if (verbosity > 3) {
        printf("%d 2bin layers\n", layer_count);
    }
    aa_tree_free(unique_layers_tree);
    return layer_count;
}

/*
 * create the prune table, combined for both bits as they are nearly identical.
 * the only difference is that on group < 4, bit 2 can't actually do 0111... in
 * a single dbin layer.
 */

/* ignore the above. create two prune tables, one for the low signal strengths and one for the high signal strengths
 */
uint8_t* get_prune_table(int group, int offset) {
    struct precomputed_hex_layer* hex_layers = precompute_hex_layers(group, -1);
    // format: bits 0-3: distance, 4-14: layer index, 15: emptiness flag
    int16_t* pretable = malloc(PRETABLE_SIZE * sizeof(int16_t));

    // this should get optimized into a constant
    int powers_of_3[16];
    powers_of_3[0] = 1;
    for (int i = 1; i < 16; i++) {
        powers_of_3[i] = 3 * powers_of_3[i - 1];
    }

    if (verbosity > 2) {
        printf("generating pretable for prune table %d\n", offset);
    }
    // fill with unassigned values
    for (int i = 0; i < PRETABLE_SIZE; i++) pretable[i] = -1;

#if 0
    for (int config = 0; config < DBIN_CONFIG_COUNT; config++) {
        int map = _pext_u32(dbin_layer128(SHUFB_IDENTITY_128, config), LO_HALVES_8_64 << (offset*8));
        pretable[map] = 0;
    }
#endif
#if 1
    // fill the distance 0 parts, any index whos binary representation fits the regex /1*0*1*/
    // slight amount of redundancy, not worth complicating
    for (int i = 0; i < 17; i++) {
        int high_part = (-1 << i);
        for (int j = 0; j < i ; j++) {
            pretable[UINT16_MAX & (high_part | ~(-1 << j))] = 0;
        }
    }

    // remove spots that aren't possible by setting them to distance 15
    if (group > 2) {
        // this is technically possible with group 2, although somewhat trivial
        pretable[0] = 15;
        pretable[UINT16_MAX] = 15;
    }
    if (group == 4) {
        // group 4 requires that both masks contain at least 2 0's and 2 1's
        for (int i = 0; i < 16; i++) {
            pretable[1 << i] = 15;
            pretable[UINT16_MAX & ~(1 << i)] = 15;
        }
    }
#endif

    // fill in the rest of the pretable
    for (int search_distance = 0; ; search_distance++) {
        int found = 0;
        for (int map = 0; map < PRETABLE_SIZE; map++) {
            uint16_t entry = pretable[map];
            if ((entry & 15) != search_distance) continue;
            found++;
            // add next layers
            struct precomputed_hex_layer* current_layer = hex_layers + (entry >> 4);
            for (int next_layer_i = 0; next_layer_i < current_layer->next_layer_count; next_layer_i++) {
                struct precomputed_hex_layer* next_layer = current_layer->next_layers[next_layer_i];
                int next_map = dbin_exact_prepend_map_packed64(next_layer->map, map);
                // skip already filled entries
                if (pretable[next_map] >= 0) continue;
                pretable[next_map] = ((next_layer - hex_layers) << 4) | (search_distance + 1);
            }
        }
        if (verbosity > 3) {
            printf("maps of distance %d: %d\n", search_distance, found);
        }
        if (!found) break;
        if (search_distance == 12) {
            printf("reached too much distance\n");
            break;
        }
    }

    if (verbosity > 2) printf("generating prune table\n");

    // now make the actual table
    /* uint8_t* prune_table = malloc((PRUNE_TABLE_ENTRY_COUNT + 1) / 2 * sizeof(uint8_t)); */
    uint8_t* prune_table = malloc(PRUNE_TABLE_ENTRY_COUNT * sizeof(uint8_t));

    uint16_t* next_pretable_entry = pretable;
    uint64_t bct_index = 0;
    for (int index = 0; index < PRUNE_TABLE_ENTRY_COUNT; index++, bct_index = bct_inc(bct_index)) {
        uint8_t value;
        if (bct_any_twos(bct_index)) {
            // dont care value, take better of two others that will always have already been filled out
            int offset = powers_of_3[bct_lowest_two(bct_index)];
            uint8_t distance0 = uint4_array_get(prune_table, index - offset * 2);
            uint8_t distance1 = uint4_array_get(prune_table, index - offset);
            value = distance0 < distance1 ? distance0 : distance1;
        } else {
            // no dont care values, pull from pretable
            // we could easily convert the number using PEXTR, but they happen in order anyway
            value = *next_pretable_entry & 15;
            next_pretable_entry++;
        }
        uint4_array_set(prune_table, index, value);
    }

    free(pretable);
    if (verbosity > 2) printf("prune table generated\n");
    return prune_table;
}

void fill_bct_halve_values() {
    if (bct_half_values) return;

    int powers_of_3[16];
    powers_of_3[0] = 1;
    for (int i = 1; i < 16; i++) {
        powers_of_3[i] = 3 * powers_of_3[i - 1];
    }

    bct_half_values = malloc(65536 * sizeof(int));
    for (int i = 0; i < 65536; i++) {
        int value = 0;
        for (int j = 0; j < 16; j++) {
            if ((i >> j) & 1)
                value += powers_of_3[j];
        }
        bct_half_values[i] = value;
    }
}

int get_ternary_index(uint16_t zeroes, uint16_t ones) {
    return bct_half_values[UINT16_MAX ^ ((uint16_t) zeroes | ones)] * 2 + bct_half_values[ones];
}

int check_dbin_partial(uint32_t exact, uint64_t partial) {
    // are there any 0's that shouldn't be there
    if (partial & exact) return 0;
    // are there any 1's that shouldn't be there
    if ((partial >> 32) & ~exact) return 0;
    return 1;
}

static int check_last_layer(struct dbin_solve_globals* globals, uint64_t remaining_map) {
    // check to see if both can be satisfied at once
    for (int i = 0; i < globals->config.unique_dbin_layers; i++) {
        uint32_t final_layer_map = globals->config.dbin_layers[i].map;
        if (!check_dbin_partial(final_layer_map, remaining_map)) continue;
        // found
        if (globals->output.chain != NULL)
            globals->output.chain[globals->config.current_bfs_depth] = globals->config.dbin_layers[i].config;

        if (verbosity > 3) printf("%03x: %08x\n", globals->config.dbin_layers[i].config, final_layer_map);
        return 1;
    }
    return 0;
}

static int dfs(struct dbin_solve_globals* globals, struct precomputed_hex_layer* layer, uint64_t remaining_map, int remaining_depth) {
    if (remaining_depth == 0) return check_last_layer(globals, remaining_map);

    for (int i = 0; i < layer->next_layer_count; i++) {
        struct precomputed_hex_layer* next_layer = layer->next_layers[i];
        uint64_t next_remaining_map = dbin_partial_unprepend_map_packed64(next_layer->map, remaining_map);
        // legality check
        if (next_remaining_map & (next_remaining_map >> 32)) continue;
        // prune table check
        if (uint4_array_get(globals->config.prune_table, get_ternary_index(next_remaining_map, (next_remaining_map >> 32))) > remaining_depth) continue;
        if (uint4_array_get(globals->config.prune_table, get_ternary_index(next_remaining_map >> 16, next_remaining_map >> 48)) > remaining_depth) continue;

        // passed, check further
        if (cache_check(&main_cache, next_remaining_map, 99 - remaining_depth)) continue;
        
        int success = dfs(globals, next_layer, next_remaining_map, remaining_depth - 1);
        if (success) {
            if (globals->output.chain != NULL)
                globals->output.chain[globals->config.current_bfs_depth - remaining_depth] = next_layer->config;
            if (verbosity > 3) printf("%03x (%016lx): %016lx\n", next_layer->config, little_endian_xmm_to_uint(unpack_uint_to_xmm(next_layer->map)), next_remaining_map);
            return 1;
        }
    }

    return 0;
}

int dbin_solve(uint32_t map, uint16_t* output_chain, int max_depth) {
    if (max_depth < 0) return max_depth - 1;

    fill_bct_halve_values();
    struct dbin_solve_globals globals = {0};
    globals.config.group = get_dbin_exact_group(map);
    globals.output.chain = output_chain;
    
    cache_init(&main_cache);

    globals.config.prune_table = get_prune_table(globals.config.group, 0);

    struct precomputed_dbin_layer dbin_layers[DBIN_CONFIG_COUNT];
    globals.config.dbin_layers = dbin_layers;
    globals.config.unique_dbin_layers = precompute_dbin_layers(dbin_layers, globals.config.group);

    struct precomputed_hex_layer* identity_layer = precompute_hex_layers(globals.config.group, -1);

    uint64_t partial_map = (uint64_t) map << 32 | (map ^ UINT32_MAX);

    for (int depth = 0; depth < max_depth; depth++) {
        if (verbosity > 1) printf("checking depth %d\n", depth);
        globals.config.current_bfs_depth = depth;
        if (dfs(&globals, identity_layer, partial_map, depth)) {
            return depth + 1;
        }
        invalidate_cache(&main_cache);
    }
    free(globals.config.prune_table);
    if (verbosity > 2) cache_print_stats(&main_cache);
    return max_depth - 1;
}

void dbin_print_solve(uint32_t map) { 
    printf("solving %08x\n", map);
    uint16_t chain[64];
    int length = dbin_solve(map, chain, 64);
    printf("solution, length %d:", length);
    if (verbosity > 2) {
        uint64_t post_hex = apply_hex_chain(IDENTITY_PERM_LE64, chain, length - 1);
        printf(" (%08x)", dbin_layer64(post_hex, chain[length - 1]));
    }
    for (int i = 0; i < length; i++) {
        printf("\t%03x", chain[i]);
    }
    printf("\n");
}

enum LONG_OPTIONS {
    LONG_OPTION_MAX_DEPTH = 1000,
    LONG_OPTION_CACHE_SIZE
};

static const struct argp_option options[] = {
    { "max-layers", LONG_OPTION_MAX_DEPTH, "N", 0, "Limit results to chains up to N layers long, including the final 2bin layer" },
    { "cache", LONG_OPTION_CACHE_SIZE, "N", 0, "Set the cache size to 2**N bytes. default: 26 (64MB)" },
    { 0 }
};

static error_t parse_opt(int key, char* arg, struct argp_state *state) {
    struct arg_settings_solver_dbin* settings = state->input;
    switch (key) {
        case LONG_OPTION_MAX_DEPTH:
            global_max_depth = atoi(arg);
            break;
        case LONG_OPTION_CACHE_SIZE:
            main_cache.size_log = (atoi(arg) - 4);
            break;
        case ARGP_KEY_SUCCESS:
            verbosity = settings->global->verbosity;
            break;
    }
    return 0;
}

struct argp argp_solver_dbin = {
    options,
    parse_opt
};


