#ifndef COMMAND_HEX_H
#define COMMAND_HEX_H
#include "../arg_global.h"
#include "../solver/HlpSolve.h"

struct arg_settings_command_hex {
    struct arg_settings_global* global;
    char* map;
    struct arg_settings_solver_hex settings_solver_hex;
};

extern struct argp argp_command_hex;

#endif
