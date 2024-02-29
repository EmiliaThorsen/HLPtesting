#include "dbin_command.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char** append_str(char** str1, char* str2) { 
    if (!str2) return str1;

    if (*str1) {
        char* new_str = malloc(strlen(*str1) + strlen(str2) + 1);
        strcpy(new_str, *str1);
        if (str2) strcat(new_str, str2);
        free(*str1);
        *str1 = new_str;
    } else {
        *str1 = malloc(strlen(str2) + 1);
        strcpy(*str1, str2);
    }

    return str1;
}
enum LONG_OPTIONS {
    LONG_OPTION_MAX_DEPTH = 1000,
    LONG_OPTION_ACCURACY,
    LONG_OPTION_CACHE_SIZE
};

static const char doc[] =
"Find a solution to the dual binary problem"
;

static const struct argp_option options[] = {
    { 0 }
};

static error_t parse_opt(int key, char* arg, struct argp_state *state) {
    struct arg_settings_command_dbin* settings = state->input;
    switch (key) {
        case ARGP_KEY_ARG:
            settings->arg_count++;
            if (settings->arg_count == 1) {
                settings->first_bits = strtol(arg, 0, 16);
            } else if (settings->arg_count == 2) {
                settings->second_bits = strtol(arg, 0, 16);
            } else {
                argp_error(state, "too many functions");
            }
            break;
        case ARGP_KEY_INIT:
            settings->arg_count = 0;
            settings->settings_solver_dbin.global = settings->global;
            state->child_inputs[0] = &settings->settings_solver_dbin;
            break;
        case ARGP_KEY_SUCCESS:
            dbin_print_solve(((uint32_t) settings->second_bits << 16) | settings->first_bits);
            break;
        case ARGP_KEY_NO_ARGS:
            argp_state_help(state, stderr, ARGP_HELP_USAGE | ARGP_HELP_SHORT_USAGE | ARGP_HELP_SEE);
            return 1;
    }
    return 0;
}

static struct argp_child argp_children[] = {
    {&argp_solver_dbin, 0, 0, 0},
    { 0 }
};

struct argp argp_command_dbin = {
    options,
    parse_opt,
    "FUNCTION",
    doc,
    argp_children
};


