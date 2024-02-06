
# uncomment for debug
# DEBUG := 1

FLAGS := 

ifdef DEBUG
	FLAGS += -g
endif

a.out: DFStest.o asm_parts.o aa_tree.o makefile
	gcc *.o -z noexecstack $(FLAGS)

aa_tree.o: aa_tree.c makefile
	gcc -o aa_tree.o aa_tree.c -O3 -march=native -c $(FLAGS)

# do NOT set to even -O1, at least until the bugs have been resolved.
# for some reason, the optimization breaks something that has yet to be
# determined
DFStest.o: DFStest.c makefile aa_tree.h
	gcc -o DFStest.o DFStest.c -O0 -march=native -c $(FLAGS)

asm_parts.o: asm_parts.s makefile
	nasm -o asm_parts.o asm_parts.s -felf64 $(FLAGS)

