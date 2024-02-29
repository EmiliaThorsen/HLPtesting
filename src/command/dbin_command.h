#ifndef COMMAND_DBIN_H
#define COMMAND_DBIN_H
#include <stdint.h>
#include "../arg_global.h"
#include "../solver/dbin_solve.h"

struct arg_settings_command_dbin {
    struct arg_settings_global* global;
    uint16_t first_bits, second_bits;
    int arg_count;
    struct arg_settings_solver_dbin settings_solver_dbin;
};

extern struct argp argp_command_dbin;

#endif

