default rel
BITS 64
%use altreg

global array_to_uint, uint_to_array
global fix_uint
global apply_mapping
global apply_and_check
global layer
global search_last_layer
global batch_apply_and_check
global bitonic_sort4x16x8

extern wanted
extern goal


struc branch_layer
.map resq 1
.configIndex resw 1
.separations resb 1
resb 5
endstruc


section .data

align 32
low_byte_mask: times 32 db 15
barrel_unpack_shifts: dq 4,4,0,0
batch_unpack_shifts: dq 4,0,4,0
bitonic_2swap: times 2 db 1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14
sort_zip: times 2 db 0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15
sort_zip2: times 2 db 8,0,9,1,10,2,11,3,12,4,13,5,14,6,15,7
word_reverse: times 2 db 14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1
word_reverse_accross_qwords: times 2 db 6,7,4,5,2,3,0,1,14,15,12,13,10,11,8,9
swap_adjacent_words: times 2 db 2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13


        ; align 16
fix_uint_perm: db 7,15,6,14,5,13,4,12,3,11,2,10,1,9,0,8


; section .note.GNU-stack,"",@progbits

section .text

%macro layer_basic 0
        ; rdi: map
        ; si: config (RABaaaabbbb, b being the side comparator)

        ; unpack map
    vmovq xmm0, rdi
    vmovdqa xmm8, [low_byte_mask]
    vpslldq xmm1, xmm0, 8
    vpsrlq xmm0, 4
    vpor xmm0, xmm1
    vpand xmm0, xmm8

        ; unpack config
        ; adjust mode if rotated
    mov rax, rsi
    shr rax, 10
    shl rax, 8
    add rsi, rax

        ; ss
    vmovd xmm4, esi
    vpbroadcastb xmm2, xmm4
    vpsrlq xmm3, xmm2, 4
    vpand xmm2, xmm8
    vpand xmm3, xmm8

        ; modes & rotation
    vpshufd xmm4, xmm4, 0
    vpslld xmm6, xmm4, 23 - 2
    vpslld xmm5, xmm4, 23 - 1
    vpslld xmm4, xmm4, 23 - 0
    vpsrad xmm6, 31
    vpsrad xmm5, 31
    vpsrad xmm4, 31

        ; swap if rotated
    vpxor xmm7, xmm0, xmm3
    vpand xmm6, xmm6, xmm7
    vpxor xmm1, xmm3, xmm6
    vpxor xmm3, xmm0, xmm6

        ; xmm0: side side
        ; xmm1: forth side
        ; xmm2: side back
        ; xmm3: forth back
        ; xmm4: side mode (-1 = s, 0 = c)
        ; xmm5: forth mode
        ; apply the comparators
    vpcmpgtb xmm6, xmm0, xmm2
    vpcmpgtb xmm7, xmm1, xmm3
    vpand xmm0, xmm0, xmm4
    vpand xmm1, xmm1, xmm5
    vpsubb xmm0, xmm2, xmm0
    vpsubb xmm1, xmm3, xmm1
    vpandn xmm0, xmm6, xmm0
    vpandn xmm1, xmm7, xmm1

    vpmaxub xmm0, xmm1

        ; pack back into return
    vpsrldq xmm1, xmm0, 8
    vpsllq xmm2, xmm0, 4
    vpor xmm1, xmm1, xmm2
    vmovq rax, xmm1
%endmacro
    

layer:
    layer_basic
    ret

%macro bitonicsortstep_minmax 4
    vpminub ymm%1, ymm%3, ymm%4
    vpmaxub ymm%2, ymm%3, ymm%4
%endmacro

%macro bitonicsortstep_blend 5
    vpblendd ymm%1, ymm%3, ymm%4, %5 * 16 + %5
    vpblendd ymm%2, ymm%4, ymm%3, %5 * 16 + %5
%endmacro

%macro bitonicsortstep_perm 2
    vpshufd ymm%1, ymm%1, %2
%endmacro

%macro bitonicsortstep_blendnt 5
    bitonicsortstep_perm %4,%5
    bitonicsortstep_minmax %1,%2,%3,%4
%endmacro

%macro bitonicsortstep 6
    bitonicsortstep_blend %3,%4,%1,%2,%5
    bitonicsortstep_perm %4,%6
    bitonicsortstep_minmax %1,%2,%3,%4
%endmacro

%define shufd_qswap 01001110b
%define shufd_dswap 10110001b
%define shufd_drev 00011011b

%macro m_bitonic_sort4x16x8 0
        ; interleave the vectors
        ; preserving any sort of order here is unnessesary
    bitonicsortstep_blend 2,3,0,1,1100b
    vpshufb ymm2, ymm2, [sort_zip]
    vpshufb ymm3, ymm3, [sort_zip2]

        ; 2

        ; 6 4 2 0
        ; 7 5 3 1
    bitonicsortstep_minmax 0,1,2,3

        ; 4

        ; 6 4 2 0   5 4 1 0
        ; 5 7 1 3   6 7 2 3
    bitonicsortstep_blendnt 2,3,0,1,shufd_dswap
        ; 6 4 2 0   6 4 2 0   6 4 2 0
        ; 5 7 1 3   7 5 3 1   7 5 3 1
    bitonicsortstep 2,3,0,1,1010b,shufd_dswap

        ; 8

        ; 6 4 2 0   1 3 2 0
        ; 1 3 5 7   6 4 5 7
    bitonicsortstep_blendnt 0,1,2,3,shufd_drev
        ; 5 3 6 0   5 3 6 0   5 1 4 0
        ; 1 7 2 4   7 1 4 2   7 3 6 2
    bitonicsortstep 0,1,2,3,1010b,shufd_dswap
        ; 7 3 4 0   7 3 4 0   6 2 4 0
        ; 5 1 6 2   6 2 5 1   7 3 5 1
    bitonicsortstep 0,1,2,3,1100b,shufd_qswap
        ; E 6 C 4 A 2 8 0
        ; F 7 D 5 B 3 9 1

        ; 16

        ; E 6 A 2 C 4 8 0
        ; 1 9 5 D 3 B 7 F
    vpshufb ymm1, ymm1, [word_reverse]
        ; 1 6 5 2 3 4 7 0
        ; E 9 A D C B 8 F
    vpminub ymm2, ymm0, ymm1
    vpmaxub ymm3, ymm0, ymm1
        ; E 6 A 2 C 4 8 0
        ; 1 9 5 D 3 B 7 F
    vpblendw ymm0, ymm2, ymm3, 10101010b
    vpblendw ymm1, ymm3, ymm2, 10101010b
        ; E 6 A 2 C 4 8 0
        ; B 3 F 7 9 1 D 5
    vpshufb ymm1, ymm1, [word_reverse_accross_qwords]
        ; 6 2 4 0
        ; 3 7 1 5

        ; 3 2 1 0   3 2 1 0   3 2 1 0
        ; 6 7 4 5   7 6 5 4   7 6 5 4
    bitonicsortstep 0,1,2,3,1010b,shufd_dswap
        ; 7 6 1 0   7 6 1 0   5 4 1 0
        ; 3 2 5 4   5 4 3 2   7 6 3 2
    bitonicsortstep 0,1,2,3,1100b,shufd_qswap
        ; 7 4 3 0   7 4 3 0   6 4 2 0
        ; 5 6 1 2   6 5 2 1   7 5 3 1
    bitonicsortstep 0,1,2,3,1010b,shufd_dswap

        ; E 6 C 4 A 2 8 0  ->  7 6 5 4 3 2 1 0
        ; F 7 D 5 B 3 9 1      F E D C B A 9 8
    vpshufb ymm1, ymm1, [swap_adjacent_words]
    vpblendw ymm2, ymm0, ymm1, 10101010b
    vpcmpeqq ymm4, ymm4
    vpblendw ymm3, ymm1, ymm0, 10101010b
    vpsrlw ymm4, ymm4, 8
    vpshufb ymm3, ymm3, [swap_adjacent_words]

        ; sorting complete, now we just need to pack them back together
    vpsrlw ymm0, ymm2, 8
    vpsrlw ymm1, ymm3, 8
    vpand ymm2, ymm4, ymm2
    vpand ymm3, ymm4, ymm3
    vpackuswb ymm1, ymm0, ymm1
    vpackuswb ymm0, ymm2, ymm3

%endmacro


fix_uint:
        ; unpack map
    vmovq xmm0, rdi
    vmovdqa xmm8, [low_byte_mask]
    vpslldq xmm1, xmm0, 8
    vpsrlq xmm0, 4
    vpor xmm0, xmm1
    vpand xmm0, xmm8

    vmovdqa xmm3, [fix_uint_perm]
    vpshufb xmm0, xmm0, xmm3

        ; pack back into return
    vpsrldq xmm1, xmm0, 8
    vpsllq xmm2, xmm0, 4
    vpor xmm1, xmm1, xmm2
    vmovq rax, xmm1

    ret


uint_to_array:
    vmovq xmm0, rdi
    vmovdqu xmm4, [low_byte_mask]
    vpslldq xmm1, xmm0, 8
    vpsrlq xmm0, 4
    vpor xmm0, xmm1
    vpand xmm0, xmm4

    vmovdqu [rsi], xmm0
    ret


array_to_uint:
    vmovdqu xmm0, [rdi]

    vmovdqa xmm2, [low_byte_mask]
    vpsrldq xmm1, xmm0, 8
    vpsllq xmm0, 4
    vpand xmm0, xmm2
    vpor xmm1, xmm0
    vmovq rax, xmm1

    ret


%macro apply_map_basic 0
        ;vpermq?
    vmovq xmm0, rdi
    vmovq xmm2, rsi
    vmovdqa xmm4, [low_byte_mask]
    vpslldq xmm1, xmm0, 8
    vpslldq xmm3, xmm2, 8
    vpsrlq xmm0, 4
    vpsrlq xmm2, 4
    vpor xmm0, xmm1
    vpor xmm2, xmm3
    vpand xmm0, xmm4
    vpand xmm2, xmm4
    
    vpshufb xmm0, xmm2, xmm0

    vpsrldq xmm1, xmm0, 8
    vpsllq xmm2, xmm0, 4
    vpor xmm1, xmm2
    ; vpand xmm0, xmm4
    vmovq rax, xmm1
%endmacro

apply_mapping:
    apply_map_basic
    ret

bitonic_sort4x16x8:
    vmovdqu ymm0, [rdi]
    vmovdqu ymm1, [rdi+32]
    m_bitonic_sort4x16x8
    vmovdqu [rdi], ymm0
    vmovdqu [rdi+32], ymm1
    ret


search_last_layer:
        ; rdi: input
        ; rsi: maps
        ; rdx: quantity
        ; rcx: goal

    vmovq xmm0, rdi
    vmovq xmm2, rcx
    mov rcx, rdx
    dec rcx
    shr rcx, 2
    shl rcx, 5
    add rcx, rsi

    vmovdqa ymm7, [low_byte_mask]
    vpslldq xmm1, xmm0, 8
    vpslldq xmm3, xmm2, 8
    vpsrlq xmm0, 4
    vpsrlq xmm2, 4
    vpor xmm0, xmm1
    vpor xmm2, xmm3
    vpand xmm0, xmm7
    vpand xmm2, xmm7
    vpermq ymm15, ymm0, 0x44
    vpermq ymm14, ymm2, 0x44
        ; ymm15: input
        ; ymm14: goal

    xor rax, rax
    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11

    vpcmpeqq xmm5, xmm5, xmm5
    vmovdqa ymm4, [batch_unpack_shifts]

.loop:
        ; unpack the maps
    vmovdqu ymm6, [rcx]
    vpshufd ymm0, ymm6, 0x44
    vpshufd ymm1, ymm6, 0xee
    vpsrlvq ymm0, ymm4
    vpsrlvq ymm1, ymm4
    vpand ymm0, ymm7
    vpand ymm1, ymm7
    
        ; apply the maps
    vpshufb ymm0, ymm0, ymm15
    vpshufb ymm1, ymm1, ymm15

        ; test if any of the results were a match
    vpxor ymm0, ymm14
    vpxor ymm1, ymm14

    vptest ymm5, ymm1
    setz r10b
    setc r11b
    or r8, r10
    or r9, r11
    shl r8, 1
    shl r9, 1

    vptest ymm5, ymm0
    setz r10b
    setc r11b
    or r8, r10
    or r9, r11

        ; test if we have reached the end
    sub rcx, 32
    cmp rcx, rsi
    setl al

        ; unified loop condition
    or rax, r8
    or rax, r9

    jz .loop

    xor rdx, rdx
        ; determine what made the loop end
    shl r9, 2
    or r8, r9

        ; see if there were any matches
        ; if not, return -1, else return the actual index
    setz dl
    neg rdx
    bsf rax, r8
    add rax, 4
    sub rcx, rsi
    shr rcx, 3
    add rax, rcx
    or rax, rdx

    ret


%macro check_validity 1
    vpsrlq ymm2, ymm%1, 4
    vpand ymm%1, ymm7
    vpand ymm2, ymm7

        ; xmm0: dest values
        ; xmm2: current values
    vpsrldq ymm3, ymm%1, 1
    vpsrldq ymm4, ymm2, 1
    vpsubb ymm%1, ymm3, ymm%1
    vpsubb ymm2, ymm4, ymm2
    vpslldq ymm%1, 1
    vpslldq ymm2, 1

    vpxor ymm4, ymm4
        ; xmm0,2: same as before but now deltas
        ; check for illegal maps
    xor r10, r10
    xor r11, r11
    vpcmpeqb ymm3, ymm2, ymm4
    vpcmpeqq xmm4, xmm4
    vpand ymm3, ymm%1
    vptest ymm4, ymm3
    setz r10b
    setc r11b


        ; check distance and store
    vpabsb ymm%1, ymm%1
    vpsubb ymm3, ymm2, ymm%1
    vpmovmskb rdx, ymm3

        ; we'll conditionally move the pointer along, but store the value each
        ; time. there is probably a better way to do this, even just packing it
        ; all into a single register first. but the gains are (probably) very
        ; small there.

        ; lower
    xor r9, r9
    xor rax, rax
    mov [rdi + branch_layer.configIndex], rcx
    vmovq [rdi + branch_layer.map], xmm5
    vpermq ymm5, ymm5, 01001110b
    add rcx, 2

    popcnt ax, dx
    mov [rdi + branch_layer.separations], al
    shr rdx, 16
    cmp rax, r8
    setle r9b
    and r9, r10
    shl r9, 4
    add rdi, r9

        ; upper
    xor r9, r9
    mov [rdi + branch_layer.configIndex], rcx
    vmovq [rdi + branch_layer.map], xmm5

    popcnt rax, rdx
    mov [rdi + branch_layer.separations], al
    cmp rax, r8
    setle r9b
    and r9, r11
    shl r9, 4
    add rdi, r9
%endmacro

batch_apply_and_check:
        ; rdi: start (uint64)
        ; rsi: input maps (pointer)
        ; rdx: output ids (pointer) (must be allocated and large enough)
        ; rcx: quantity (int)
        ; r8: threshhold (int)
        ;
        ; returns: number of found valid cases

    push r15
    push r14
    push r13
    push r12
    push rdx


        ; similar start to last layer search
    vmovq xmm0, rdi
    vmovq xmm2, [wanted]
    mov rdi, rdx

    ; shl rcx, 3
    ; add rcx, rsi
    mov r12, rcx
    xor rcx,rcx

        ; rdi: current output pointer
        ; rcx: current index
        ; r8: threshold
        ; rsi: base input pointer
        ; other registers okay to clobber

        ; unpack the important maps
        ; same as last layer
    vmovdqa ymm7, [low_byte_mask]
    vpslldq xmm1, xmm0, 8
    vpslldq xmm3, xmm2, 8
    vpsrlq xmm0, 4
    vpsrlq xmm2, 4
    vpor xmm0, xmm1
    vpor xmm2, xmm3
    vpand xmm0, xmm7
    vpand xmm2, xmm7
    vpermq ymm15, ymm0, 0x44
    vpermq ymm14, ymm2, 0x44
        ; ymm15: input
        ; ymm14: goal
        ; ymm7: low byte mask
        ; ymm6: unpack shifts

    vmovdqa ymm6, [batch_unpack_shifts]



.loop:
        ; unpack the maps
        ; same as last layer
    shl rcx, 3
    vmovdqu ymm5, [rsi + rcx]
    vpshufd ymm0, ymm5, 0x44
    vpshufd ymm1, ymm5, 0xee
    vpsrlvq ymm0, ymm6
    vpsrlvq ymm1, ymm6
    vpand ymm0, ymm7
    vpand ymm1, ymm7
    shr rcx, 3
    
        ; apply the maps
        ; same as last layer
    vpshufb ymm0, ymm0, ymm15
    vpshufb ymm1, ymm1, ymm15

        ; pack them up for storing later
    vpsrldq ymm2, ymm0, 8
    vpsllq ymm3, ymm0, 4
    vpor ymm5, ymm2, ymm3

    vpsrldq ymm2, ymm1, 8
    vpsllq ymm3, ymm1, 4
    vpor ymm2, ymm3
    vpslldq ymm2, 8

    vpblendd ymm5, ymm2, 0xcc

    
        ; prepare for sorting
    vpsllq ymm0, 4
    vpsllq ymm1, 4
    vpor ymm0, ymm14
    vpor ymm1, ymm14

        ; sort
    m_bitonic_sort4x16x8

        ; set up for validity test
    ; xor r9, r9
    xor r10, r10
    xor r11, r11

        ; check the validity
    check_validity 0
    vpermq ymm5, ymm5, 00011011b
    dec rcx
    check_validity 1
    ; sub rcx, 7
    inc rcx
    cmp rcx, r12

    jl .loop

        ; finish off
        ; all we need to do is get the return value made
    mov rax, rdi
    pop rdx
    sub rax, rdx
    shr rax, 4

    pop r12
    pop r13
    pop r14
    pop r15

    ret
