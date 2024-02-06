default rel
BITS 64
%use altreg

global array_to_uint, uint_to_array
global fix_uint
global apply_mapping
global layer
global search_last_layer
global batch_apply_and_check


section .data

align 32
low_byte_mask: times 32 db 15
barrel_unpack_shifts: dq 4,4,0,0
batch_unpack_shifts: dq 4,0,4,0
bitonic_2swap: times 2 db 1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14


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

%macro bitonic_sort_step 3
        ; %1: type of mm
        ; %2: how many at once
        ; %3: which mm to use for swapping

        ; the strange method used here is because: 1) it improves the pipeline,
        ; and 2) it reduces the total memory loads needed as we don't need a
        ; mask

    vpmaxub %1mm8, %1mm9, %1mm%3
    vpshufb %1mm2, %1mm0, %1mm%3
    vpshufb %1mm3, %1mm0, %1mm8
    vpmaxub %1mm2, %1mm0
    vpxor %1mm0, %1mm3
    vpxor %1mm0, %1mm2
%if %2 = 2
    vpshufb %1mm2, %1mm1, %1mm%3
    vpshufb %1mm3, %1mm1, %1mm8
    vpmaxub %1mm2, %1mm1
    vpxor %1mm1, %1mm3
    vpxor %1mm1, %1mm2
%endif
%endmacro

%macro bitonic_sort_load 1
    nop
    vmovdqa %1mm10, [bitonic_2swap]
    vpshufb %1mm9, %1mm10, %1mm10
    vpshuflw %1mm11, %1mm9, 10110001b
    vpshufhw %1mm11, %1mm11, 10110001b
    vpshufd %1mm12, %1mm9, 10110001b
%endmacro

%macro bitonic_sort_loop 2
        ; we make clever use of shuf's to avoid loading as many vectors from
        ; memory, and instead get them from already loaded things. this doesn't
        ; take more instructions either
    vpshufb %1mm13, %1mm11, %1mm10
    bitonic_sort_step %1,%2,10
    bitonic_sort_step %1,%2,13
    vpshufd %1mm13, %1mm13, 10110001b
    bitonic_sort_step %1,%2,10
    bitonic_sort_step %1,%2,13
    vpshufd %1mm13, %1mm13, 01001110b
    bitonic_sort_step %1,%2,11
    bitonic_sort_step %1,%2,10
    bitonic_sort_step %1,%2,13
    bitonic_sort_step %1,%2,12
    bitonic_sort_step %1,%2,11
    bitonic_sort_step %1,%2,10

%endmacro

%macro bitonic_sort_main 2
        ; %1: type of mm
        ; %2: how many simultaneously (1-2)
        ; clobbers mm 2-3, 8-13
        ; in: mm0 (&mm1)
        ; out: mm0 (&mm1), sorted

        ; xmm10-13: permutation that swaps the values
        ; xmm9: list of indices
        ; xmm8: permutation to broadcast higher index
        ; xmm0: the working vector
        ; xmm1: the swapped, then maxed values
        ; xmm2: the higher index value, xored into xmm0
    bitonic_sort_load %1
    bitonic_sort_loop %1,%2

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

        ; we'll conditionally move the pointer along, but store the value each
        ; time. there is probably a better way to do this, even just packing it
        ; all into a single register first. but the gains are (probably) very
        ; small there.

        ; lower
        ; prefetch cache
    vmovq rax, xmm5
    mov rdx, rax
    shr rax, 32
    crc32 rax, rdx
    shl rax, 4
    add rax, r13
    prefetchnta [rax]

    vpmovmskb rdx, ymm3

    xor r9, r9
    mov [rdi], rcx
    vmovq [rdi+8], xmm5
    vpermq ymm5, ymm5, 01001110b
    add rcx, 2

        ; check distance
    xor rax, rax
    popcnt ax, dx
    shr rdx, 16
    cmp rax, r8

        ; conditional advance
    setle r9b
    and r9, r10
    shl r9, 4
    add rdi, r9

        ; upper
    xor r9, r9
    mov [rdi], rcx
    vmovq [rdi+8], xmm5

        ; dist and advance
    popcnt rax, rdx
    cmp rax, r8
    setle r9b
    and r9, r11
    shl r9, 4
    add rdi, r9

        ; prefetch
    vmovq rax, xmm5
    mov rdx, rax
    shr rax, 32
    crc32 rax, rdx
    shl rax, 4
    add rax, r13
    prefetchnta [rax]
%endmacro

batch_apply_and_check:
        ; rdi: start (uint64)
        ; rsi: input maps (pointer)
        ; rdx: output ids (pointer) (must be allocated and large enough)
        ; rcx: quantity (int)
        ; r8: threshhold (int)
        ; r9: goal (uint64)
        ; [rsp+8]: cache array pointer
        ;
        ; returns: number of found valid cases

    push r15
    push r14
    push r13
    push r12
    push rdx
    mov r13, [rsp + 8*6]

        ; similar start to last layer search
    vmovq xmm0, rdi
    vmovq xmm2, r9
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
        ;
        ; these barely won't be clobbered by bitonic sort

    bitonic_sort_load y
    vmovdqa ymm6, [batch_unpack_shifts]



.main_loop:
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

    vpblendw ymm5, ymm2, 0xf0

    
        ; prepare for sorting
    vpsllq ymm0, 4
    vpsllq ymm1, 4
    vpor ymm0, ymm14
    vpor ymm1, ymm14

        ; sort (x4)
    bitonic_sort_loop y,2

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

    jl .main_loop

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
