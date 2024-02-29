#ifndef DBIN_SOLVE_H
#define DBIN_SOLVE_H
#include <stdint.h>
#include "../arg_global.h"


int dbin_solve(uint32_t map, uint16_t* output_chain, int max_depth);

void dbin_print_solve(uint32_t map);

struct arg_settings_solver_dbin {
    struct arg_settings_global* global;
};

extern struct argp argp_solver_dbin;

#endif
