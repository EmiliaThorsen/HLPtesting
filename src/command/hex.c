#include "hex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



char** appendStr(char** str1, char* str2) { 
    if (!str2) return str1;

    if (*str1) {
        char* newStr = malloc(strlen(*str1) + strlen(str2) + 1);
        strcpy(newStr, *str1);
        if (str2) strcat(newStr, str2);
        free(*str1);
        *str1 = newStr;
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
"Find a solution to the vanilla hex layer problem"
;

static const struct argp_option options[] = {
    { 0 }
};

static error_t parse_opt(int key, char* arg, struct argp_state *state) {
    struct arg_settings_command_hex* settings = state->input;
    switch (key) {
        case ARGP_KEY_ARG:
            appendStr(&(settings->map), arg);
            break;
        case ARGP_KEY_INIT:
            settings->map = 0;
            settings->settings_solver_hex.global = settings->global;
            state->child_inputs[0] = &settings->settings_solver_hex;
            break;
        case ARGP_KEY_SUCCESS:
            hlpPrintSearch(settings->map);
            break;
        case ARGP_KEY_NO_ARGS:
            argp_state_help(state, stderr, ARGP_HELP_USAGE | ARGP_HELP_SHORT_USAGE | ARGP_HELP_SEE);
            return 1;
    }
    return 0;
}

static struct argp_child argp_children[] = {
    {&argp_solver_hex, 0, 0, 0},
    { 0 }
};

struct argp argp_command_hex = {
    options,
    parse_opt,
    "FUNCTION",
    doc,
    argp_children
};


