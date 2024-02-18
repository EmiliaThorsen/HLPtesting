AUTOMAKE_OPTIONS = subdir-objects
SUBDIRS = lib
bin_PROGRAMS = hlpsolve

hlpsolve_SOURCES = ./src/main.c ./src/HlpSolve.h ./src/HlpSolve.c ./src/bitonicSort.h ./src/bitonicSort.c ./src/aa_tree.h ./src/aa_tree.c

EXTRA_DIST = m4/gnulib-cache.m4

ACLOCAL_AMFLAGS = -I m4
CCAS = nasm
AM_CFLAGS = -march=native
AM_CPPFLAGS = -I$(top_builddir)/lib -I$(top_srcdir)/lib
LDADD = lib/libgnu.a

# AM_LDFLAGS = -z noexecstack 
# AM_CCASFLAGS = -felf64

debug: AM_CFLAGS += -DDEBUG -g
debug: AM_CCASFLAGS += -dDEBUG -g

src/asm_parts.o: src/asm_parts.s
	@$(CCAS) $(AM_CCASFLAGS) $< -o $@

