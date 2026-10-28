// refs_wrap: what the ime2_check stand needs that the built libggml-cpu.so does not export.
// 1) The vendor's reference kernels *_mrow_ref from ime2_kernels.cpp: disabled with `#if 0` in the
//    dispatcher and therefore not instantiated. Here they are instantiated from THE SAME llama.cpp source
//    tree with the same gcc and flags.
// 2) The weight repack repack<block_q4_K|block_q6_K, 32, 32> from repack.cpp: template specialisations
//    that are not exported from the library.
// The vector kernels *_m1/*_m4 and the activation quantisation are NOT taken from here: the stand loads
// them from the built libggml-cpu.so via dlsym, i.e. it checks the exact machine code llama-perplexity runs.
#include "repack.cpp"
#include "ime2_kernels.cpp"

namespace chk {

void ref_i8i4(size_t rows, size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * zp, float * c,
              size_t count_n, size_t k_blks, size_t ldc) {
    if (rows == 4) {
        spacemit_kernels::ime2::gemm_kernel_i8i4_mrow_ref<4, 32>(blk_len, a, b, zp, c, 4, count_n, k_blks, ldc);
    } else {
        spacemit_kernels::ime2::gemm_kernel_i8i4_mrow_ref<1, 32>(blk_len, a, b, zp, c, 1, count_n, k_blks, ldc);
    }
}

void ref_i8i8(size_t rows, size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * zp, float * c,
              size_t count_n, size_t k_blks, size_t ldc) {
    if (rows == 4) {
        spacemit_kernels::ime2::gemm_kernel_i8i8_mrow_ref<4, 32>(blk_len, a, b, zp, c, 4, count_n, k_blks, ldc);
    } else {
        spacemit_kernels::ime2::gemm_kernel_i8i8_mrow_ref<1, 32>(blk_len, a, b, zp, c, 1, count_n, k_blks, ldc);
    }
}

void ref_i8i4_hp(size_t rows, size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * zp, float * c,
                 size_t count_n, size_t k_blks, size_t ldc) {
    if (rows == 4) {
        spacemit_kernels::ime2::gemm_kernel_i8i4_hp_mrow_ref<4, 32>(blk_len, a, b, zp, c, 4, count_n, k_blks, ldc);
    } else {
        spacemit_kernels::ime2::gemm_kernel_i8i4_hp_mrow_ref<1, 32>(blk_len, a, b, zp, c, 1, count_n, k_blks, ldc);
    }
}

int repack_q4_0_256(ggml_tensor * t, const void * data, size_t size) {
    return ggml::cpu::riscv64_spacemit::repack<block_q4_0, 256, 32>(t, data, size);
}

int repack_q4_k(ggml_tensor * t, const void * data, size_t size) {
    return ggml::cpu::riscv64_spacemit::repack<block_q4_K, 32, 32>(t, data, size);
}

int repack_q6_k(ggml_tensor * t, const void * data, size_t size) {
    return ggml::cpu::riscv64_spacemit::repack<block_q6_K, 32, 32>(t, data, size);
}

}  // namespace chk
