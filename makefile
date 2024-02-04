

a.out: DFStest.o asm_parts.o aa_tree.o
	gcc *.o -g -z noexecstack

aa_tree.o: aa_tree.c
	gcc -o aa_tree.o aa_tree.c -O3 -march=native -c -g

DFStest.o: DFStest.c
	gcc -o DFStest.o DFStest.c -O3 -march=native -c -g

asm_parts.o: asm_parts.s
	nasm -o asm_parts.o asm_parts.s -felf64 -g

