default rel
BITS 64
%use altreg

global apply_and_check; input, configuration, threshold
; global bitonic_sort16x8; pointer to byte array
; global pretty_uint_to_array; uint, array
global array_to_uint, uint_to_array
global fix_uint
global apply_mapping, store_mapping
; global layer_inner_cc, layer_inner_cs, layer_inner_ss, layer_inner_sc, layer_inner_rot_sc
        ; global getGroup ; i could and it'd be really fast but it's not really necessary at all
global layer
global search_last_layer

extern goal

section .data

align 32
low_byte_mask: times 32 db 15
barrel_unpack_shifts: dq 4,4,0,0
last_layer_unpack_shifts: dq 4,0,4,0
; mode_unpack_shifts: dq 31,31,0,0

bitonic_swap:
times 2 db 1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14
times 2 db 2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13
times 2 db 4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11

bitonic_flip:
times 2 db 3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12
times 2 db 7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8
times 2 db 15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0

counting: db 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
fix_uint_perm: db 7,15,6,14,5,13,4,12,3,11,2,10,1,9,0,8
        ; unfix_uint_perm: db 14,12,10,8,6,4,2,0,15,13,11,9,7,5,3,1

; uint2xmm_perm: db 15,7,14,6,13,5,12,4,11,3,10,2,9,1,8,0
; uint2xmm_r_perm: db 0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15
; xmm2uint_perm: db 15,13,11,9,7,5,3,1,14,12,10,8,6,4,2,0

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

%macro bitonic_sort_step 3
        ; %1: type of mm
        ; %2: how many at once
        ; %3: which mm to use for swapping

        ; the strange method used here is because: 1) it improves the pipeline,
        ; and 2) it reduces the total memory loads needed as we don't need a
        ; mask

%if %2 = 1
    vpmaxub %1mm7, %1mm8, %1mm%3
    vpshufb %1mm1, %1mm0, %1mm%3
    vpshufb %1mm2, %1mm0, %1mm7
    vpmaxub %1mm1, %1mm0
    vpxor %1mm0, %1mm2
    vpxor %1mm0, %1mm1
%else
    vpmaxub %1mm7, %1mm8, %1mm%3

    vpshufb %1mm2, %1mm0, %1mm%3
    vpshufb %1mm3, %1mm1, %1mm%3

    vpshufb %1mm4, %1mm0, %1mm7
    vpshufb %1mm5, %1mm1, %1mm7

    vpmaxub %1mm2, %1mm0
    vpmaxub %1mm3, %1mm1

    vpxor %1mm0, %1mm4
    vpxor %1mm1, %1mm5

    vpxor %1mm0, %1mm2
    vpxor %1mm1, %1mm3
%endif
%endmacro

%macro bitonic_sort_main 2
        ; %1: type of mm
        ; %2: how many simultaneously (1-2)
        ; clobbers mm 1-2(2-5 if %2=2), 7-12
        ; in: mm0 (&mm1)
        ; out: mm0 (&mm1), sorted
    nop
    vmovdqa %1mm8, [counting]
    vmovdqa %1mm9, [bitonic_swap + 0]
    vmovdqa %1mm12, [bitonic_flip + 0]

        ; xmm9-14: permutation that swaps the values
        ; xmm8: list of indices
        ; xmm7: permutation to broadcast higher index
        ; xmm0: the working vector
        ; xmm1: the swapped, then maxed values
        ; xmm2: the higher index value, xored into xmm0

    bitonic_sort_step %1,%2,9
    bitonic_sort_step %1,%2,12
    vmovdqa %1mm12, [bitonic_flip + 32]
    bitonic_sort_step %1,%2,9
    vmovdqa %1mm10, [bitonic_swap + 32]
    bitonic_sort_step %1,%2,12
    vmovdqa %1mm12, [bitonic_flip + 64]
    bitonic_sort_step %1,%2,10
    vmovdqa %1mm11, [bitonic_swap + 64]
    bitonic_sort_step %1,%2,9
    bitonic_sort_step %1,%2,12
    bitonic_sort_step %1,%2,11
    bitonic_sort_step %1,%2,10
    bitonic_sort_step %1,%2,9

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


store_mapping:
    vmovq xmm0, rdi
    vmovdqa xmm2, [low_byte_mask]
    vpslldq xmm1, xmm0, 8
    vpsrlq xmm0, 4
    vpor xmm0, xmm1
    vpand xmm0, xmm2

    vmovdqu [rsi], xmm0

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


apply_and_check:
        ; rdi: input
        ; rsi: map
        ; rdx: threshold

    apply_map_basic

    ; call apply_mapping

        ; prepare for sorting
    vpsllq xmm0, 4
    vpor xmm0, [goal]

    bitonic_sort_main x,1

    vmovdqa xmm4, [low_byte_mask]

    vpsrlq xmm1, xmm0, 4
    vpand xmm0, xmm4
    vpand xmm1, xmm4

        ; xmm0: dest values
        ; xmm1: current values
    vpsrldq xmm2, xmm0, 1
    vpsrldq xmm3, xmm1, 1
    vpsubb xmm0, xmm2, xmm0
    vpsubb xmm1, xmm3, xmm1
    vpslldq xmm0, 1
    vpslldq xmm1, 1


        ; xmm0-1: same as before but now deltas
    vpxor xmm5, xmm5
    xor rcx, rcx
    vpcmpeqb xmm2, xmm1, xmm5
    vptest xmm2, xmm0
    setz cl
    neg rcx
    and rax, rcx

    xor rcx, rcx
    vpabsb xmm0, xmm0
    vpsubb xmm3, xmm1, xmm0
    vpmovmskb rdi, xmm3
    popcnt rdi, rdi
    cmp rdi, rdx
    setng cl
    neg rcx
    and rax, rcx

    ret

bitonic_sort16x8:
    vmovdqa xmm0, [rdi]
    bitonic_sort_main x,1
    vmovdqa [rdi], xmm0
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
    vmovdqa ymm4, [last_layer_unpack_shifts]

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

