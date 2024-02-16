
AC_INIT([hlpsolve], [1.0])
AM_INIT_AUTOMAKE

AC_PROG_CC
gl_EARLY
gl_INIT
AM_PROG_AS([nasm])
AC_PATH_PROG([nasm])
# AX_PROG_NASM

# Checks for header files.
AC_CHECK_HEADERS([stdint.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([setlocale strtoull])

AC_CONFIG_HEADERS([config.h])

AC_CONFIG_FILES([Makefile lib/Makefile])
AC_OUTPUT
