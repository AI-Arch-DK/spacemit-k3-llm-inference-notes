// ime2_check: accuracy check of the SpacemiT IME2 matmul kernels in llama.cpp (ggml-cpu/spacemit)
// on real model weights. Written for llama.cpp 0a1ef30 built with -DGGML_CPU_RISCV64_SPACEMIT=ON.
//
// Everything is compared on the same data (weights: the first 256 rows of real tensors of a layer of
// qwen2.5-7b-instruct-q4_K_M.gguf; activations: synthetic, see make_acts):
//   true - double: float activations x weights dequantised by stock dequantize_row_q4_K/q6_K (ggml-base);
//   up   - upstream generic CPU path: quantize_row_q8_K_ref + ggml_vec_dot_q*_K_q8_K_generic;
//   aq   - double: activations after IME quantisation (quantize_a_row_i8) x true weights - cost of
//          activation quantisation;
//   rp   - double (Q4_K only): float activations x weights after repack (make_block_q4_1x32 formula:
//          zero point = round(min/d), clamped to 0..15) - cost of the repack;
//   ex   - double (Q4_K only): quantised activations x repacked weights - what a lossless kernel would give;
//   ref1/ref4 - the vendor's reference kernels *_mrow_ref<1|4,32> (refs_wrap.cpp, same source tree);
//   m1/m4     - the vector kernels from the BUILT libggml-cpu.so (via dlsym): exactly what llama-perplexity
//               runs for one token (m1) and for a batch (m4).
// Error = relative L2 over all 4x256 outputs. Must run as an AI thread (/proc/set_ai_thread): IME2
// instructions exist only on the A100 cores; main() switches itself.
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
void   dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k);
void   dequantize_row_q6_K(const block_q6_K * x, float * y, int64_t k);
void   quantize_row_q8_K_ref(const float * x, block_q8_K * y, int64_t k);
void   ggml_vec_dot_q4_K_q8_K_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy,
                                      size_t by, int nrc);
void   ggml_vec_dot_q6_K_q8_K_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy,
                                      size_t by, int nrc);
size_t quantize_row_q4_0_ref(const float * x, block_q4_0 * y, int64_t k);
void   dequantize_row_q4_0(const block_q4_0 * x, float * y, int64_t k);
void   quantize_row_q8_0_ref(const float * x, block_q8_0 * y, int64_t k);
void   ggml_vec_dot_q4_0_q8_0_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy,
                                      size_t by, int nrc);
}

namespace chk {
void ref_i8i4(size_t rows, size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * zp, float * c,
              size_t count_n, size_t k_blks, size_t ldc);
void ref_i8i8(size_t rows, size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * zp, float * c,
              size_t count_n, size_t k_blks, size_t ldc);
void ref_i8i4_hp(size_t rows, size_t blk_len, const uint8_t * a, const uint8_t * b, const uint8_t * zp, float * c,
                 size_t count_n, size_t k_blks, size_t ldc);
int  repack_q4_k(ggml_tensor * t, const void * data, size_t size);
int  repack_q6_k(ggml_tensor * t, const void * data, size_t size);
int  repack_q4_0_256(ggml_tensor * t, const void * data, size_t size);
}  // namespace chk

using gemm_fn  = size_t (*)(size_t, const uint8_t *, const uint8_t *, const uint8_t *, float *, size_t, size_t, size_t,
                           size_t);
using quant_fn = void (*)(size_t, const float *, size_t, uint8_t *);

static gemm_fn  lib_gemm_i8i4, lib_gemm_i8i8, lib_gemm_i8i4_hp;
static quant_fn lib_quant_row, lib_quant_4row, lib_quant_row_hp, lib_quant_4row_hp;

constexpr int    M       = 4;
constexpr size_t BLK     = 32;
constexpr size_t A_BLK   = 4 + 2 + BLK;  // q8_blk_size(32, true): [f32 scale][s16 sum][int8 x32]

static std::vector<uint8_t> read_file(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        fprintf(stderr, "missing file %s\n", p.c_str());
        exit(1);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static double rel(const std::vector<double> & y, const std::vector<double> & r) {
    double num = 0, den = 0;
    for (size_t i = 0; i < y.size(); i++) {
        num += (y[i] - r[i]) * (y[i] - r[i]);
        den += r[i] * r[i];
    }
    return std::sqrt(num / den);
}

// largest |y-r| in units of the reference RMS - outliers the mean error does not show
static double maxdev(const std::vector<double> & y, const std::vector<double> & r) {
    double den = 0, mx = 0;
    for (double v : r) den += v * v;
    den = std::sqrt(den / r.size());
    for (size_t i = 0; i < y.size(); i++) mx = std::max(mx, std::fabs(y[i] - r[i]) / den);
    return mx;
}

static void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Activations: kind=norm - z*g (RMSNorm output with the layer's real norm weights), swiglu - silu(u)*v (ffn_down input),
// gauss - z, pos - |z|*g (all of one sign: worst case for the block sum asum).
static std::vector<float> make_acts(const std::string & kind, const std::vector<float> & g, size_t K, unsigned seed) {
    std::mt19937                     rng(seed);
    std::normal_distribution<float>  nd(0.0f, 1.0f);
    std::vector<float>               a(M * K);
    for (int m = 0; m < M; m++) {
        for (size_t k = 0; k < K; k++) {
            float z = nd(rng);
            float v;
            if (kind == "norm") {
                v = z * g[k];
            } else if (kind == "pos") {
                v = std::fabs(z) * (g.empty() ? 1.0f : g[k]);
            } else if (kind == "swiglu") {
                float u = 1.5f * z, w = 1.5f * nd(rng);
                v = u / (1.0f + std::exp(-u)) * w;
            } else {
                v = z;
            }
            a[m * K + k] = v;
        }
    }
    return a;
}

struct row_stats {
    double asum_max   = 0;  // largest |block sum| of the quantised activations
    double ovf_frac   = 0;  // share of blocks with |asum|*15 > 32767 (int16 overflow possible at zp = 15)
};

static void run_case(const std::string & name, const std::string & type, size_t K, size_t R,
                     const std::vector<uint8_t> & raw, const std::vector<float> & a, const std::string & kind) {
    const bool   q4k     = type == "Q4_K";
    const size_t k_blks  = K / BLK;
    const size_t row_b   = raw.size() / R;

    // 1. Reference: weights dequantised by stock ggml, computed in double.
    std::vector<float> w(R * K);
    for (size_t n = 0; n < R; n++) {
        if (q4k) dequantize_row_q4_K((const block_q4_K *) (raw.data() + n * row_b), w.data() + n * K, K);
        else     dequantize_row_q6_K((const block_q6_K *) (raw.data() + n * row_b), w.data() + n * K, K);
    }
    std::vector<double> y_true(M * R), y_up(M * R), y_aq(M * R), y_rp(M * R, 0), y_ex(M * R, 0);
    for (int m = 0; m < M; m++)
        for (size_t n = 0; n < R; n++) {
            double s = 0;
            for (size_t k = 0; k < K; k++) s += (double) a[m * K + k] * w[n * K + k];
            y_true[m * R + n] = s;
        }

    // 2. Upstream generic CPU: activations in Q8_K, ggml scalar kernel.
    std::vector<block_q8_K> q8k(K / QK_K);
    for (int m = 0; m < M; m++) {
        quantize_row_q8_K_ref(a.data() + m * K, q8k.data(), K);
        for (size_t n = 0; n < R; n++) {
            float s;
            if (q4k) ggml_vec_dot_q4_K_q8_K_generic(K, &s, 0, raw.data() + n * row_b, 0, q8k.data(), 0, 1);
            else     ggml_vec_dot_q6_K_q8_K_generic(K, &s, 0, raw.data() + n * row_b, 0, q8k.data(), 0, 1);
            y_up[m * R + n] = s;
        }
    }

    // 3. Repack by the backend (the same code that runs at model load).
    ggml_tensor t;
    memset(&t, 0, sizeof(t));
    t.type  = q4k ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K;
    t.ne[0] = K; t.ne[1] = R; t.ne[2] = 1; t.ne[3] = 1;
    std::vector<uint8_t> packed(R * K * 2 + 4096);
    uint8_t * B = (uint8_t *) (((uintptr_t) packed.data() + 255) & ~(uintptr_t) 255);
    t.data = B;
    int rc = q4k ? chk::repack_q4_k(&t, raw.data(), raw.size()) : chk::repack_q6_k(&t, raw.data(), raw.size());
    if (rc != 0) {
        fprintf(stderr, "%s: repack rc=%d\n", name.c_str(), rc);
        exit(1);
    }
    const uint8_t * zp = q4k ? B : nullptr;  // as in ime.cpp: block_type_has_zp<q4_K> = true, <q6_K> = false
    gemm_fn gemm = q4k ? lib_gemm_i8i4 : lib_gemm_i8i8;
    auto    ref  = q4k ? chk::ref_i8i4 : chk::ref_i8i8;

    // 4. IME activation quantisation: one row (m1 path) and four rows (m4 path).
    std::vector<uint8_t> qa1(M * k_blks * A_BLK + 64), qa4(M * k_blks * A_BLK + 64);
    for (int m = 0; m < M; m++) lib_quant_row(BLK, a.data() + m * K, K, qa1.data() + m * k_blks * A_BLK);
    lib_quant_4row(BLK, a.data(), K, qa4.data());

    row_stats st;
    size_t    nblk_total = 0, nblk_ovf = 0;
    for (int m = 0; m < M; m++)
        for (size_t b = 0; b < k_blks; b++) {
            const uint8_t * blk = qa1.data() + (m * k_blks + b) * A_BLK;
            float           sc;
            int16_t         as;
            memcpy(&sc, blk, 4);
            memcpy(&as, blk + 4, 2);
            int s = 0;
            for (size_t i = 0; i < BLK; i++) s += (int8_t) blk[6 + i];
            st.asum_max = std::max(st.asum_max, (double) std::abs(s));
            nblk_total++;
            if (std::abs(s) * 15 > 32767) nblk_ovf++;
            for (size_t i = 0; i < BLK; i++) {
                double aq = (double) sc * (int8_t) blk[6 + i];
                for (size_t n = 0; n < R; n++) y_aq[m * R + n] += aq * w[n * K + b * BLK + i];
            }
        }
    st.ovf_frac = (double) nblk_ovf / nblk_total;

    // 5. Q4_K: weights after repack by the make_block_q4_1x32 formula (d1 and -m1 via fp16, zp = -round(m/d) in [0,15]).
    size_t nsub = 0, nclamp = 0;
    double zp_err_max = 0;
    if (q4k) {
        std::vector<float> wr(R * K);
        for (size_t n = 0; n < R; n++) {
            const block_q4_K * x = (const block_q4_K *) (raw.data() + n * row_b);
            for (size_t sb = 0; sb < K / QK_K; sb++) {
                ggml_fp16_t dh, dmh;  // block_q4_K: d, dmin - the first two fp16 of the block
                memcpy(&dh, (const uint8_t *) &x[sb], 2);
                memcpy(&dmh, (const uint8_t *) &x[sb] + 2, 2);
                const float d    = ggml_fp16_to_fp32(dh);
                const float dmin = ggml_fp16_to_fp32(dmh);
                for (int j = 0; j < 8; j++) {
                    uint8_t sc, mm;
                    get_scale_min_k4(j, x[sb].scales, &sc, &mm);
                    const float d1  = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d * sc));
                    const float m1  = ggml_fp16_to_fp32(ggml_fp32_to_fp16(-(dmin * mm)));
                    float       raw_mid = d1 != 0 ? -m1 / d1 : 0;
                    float       mid = -std::nearbyintf(m1 / d1);
                    mid             = std::min(15.0f, std::max(0.0f, mid));
                    nsub++;
                    if (raw_mid > 15.5f || raw_mid < -0.5f) nclamp++;
                    zp_err_max = std::max(zp_err_max, (double) std::fabs(raw_mid - mid));
                    const uint8_t * q = x[sb].qs + (j / 2) * 32;
                    for (int l = 0; l < 32; l++) {
                        int qv = (j % 2 == 0) ? (q[l] & 0xF) : (q[l] >> 4);
                        wr[n * K + sb * QK_K + j * 32 + l] = d1 * (qv - mid);
                    }
                }
            }
        }
        for (int m = 0; m < M; m++)
            for (size_t n = 0; n < R; n++) {
                double s1 = 0, s2 = 0;
                for (size_t b = 0; b < k_blks; b++) {
                    const uint8_t * blk = qa1.data() + (m * k_blks + b) * A_BLK;
                    float           sc;
                    memcpy(&sc, blk, 4);
                    for (size_t i = 0; i < BLK; i++) {
                        const size_t k = b * BLK + i;
                        s1 += (double) a[m * K + k] * wr[n * K + k];
                        s2 += (double) sc * (int8_t) blk[6 + i] * wr[n * K + k];
                    }
                }
                y_rp[m * R + n] = s1;
                y_ex[m * R + n] = s2;
            }
    }

    // 6. Kernels: m1 and ref<1> - one row; m4 and ref<4> - four rows. Called as in ime.cpp (forward_mul_mat):
    // in tiles of NB_COLS = 32 columns, tile weight pointer = B + ni*row_stride_b, zero points - the same
    // pointer (block_type_has_zp), ldc = number of columns. row_stride_b = k_blks * get_repacked_block_type_size:
    // Q4_K -> (sizeof(block_q4_0) + 1) = 19 bytes per column and block, Q6_K -> sizeof(block_q8_0) = 34.
    const size_t row_stride_b = k_blks * (q4k ? sizeof(block_q4_0) + 1 : sizeof(block_q8_0));
    auto tiles = [&](int rows, const uint8_t * qa, float * c, bool use_ref) {
        for (size_t ni = 0; ni < R; ni += 32) {
            const uint8_t * bt = B + ni * row_stride_b;
            const uint8_t * zt = zp ? bt : nullptr;
            const size_t    nb = std::min<size_t>(32, R - ni);
            if (use_ref) {
                ref(rows, BLK, qa, bt, zt, c + ni, nb, k_blks, R);
            } else {
                size_t got = gemm(BLK, qa, bt, zt, c + ni, rows, nb, k_blks, R);
                if (got != (size_t) rows) { fprintf(stderr, "kernel returned %zu instead of %d\n", got, rows); exit(1); }
            }
        }
    };
    std::vector<float>  c(M * R);
    std::vector<double> y_m1(M * R), y_r1(M * R), y_m4(M * R), y_r4(M * R);
    for (int m = 0; m < M; m++) {
        tiles(1, qa1.data() + m * k_blks * A_BLK, c.data(), false);
        for (size_t n = 0; n < R; n++) y_m1[m * R + n] = c[n];
        tiles(1, qa1.data() + m * k_blks * A_BLK, c.data(), true);
        for (size_t n = 0; n < R; n++) y_r1[m * R + n] = c[n];
    }
    tiles(4, qa4.data(), c.data(), false);
    for (size_t i = 0; i < M * R; i++) y_m4[i] = c[i];
    tiles(4, qa4.data(), c.data(), true);
    for (size_t i = 0; i < M * R; i++) y_r4[i] = c[i];

    printf("%s\t%s\t%zu\t%s\t%.5f\t%.5f\t%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%s\t%.3f\t%.0f\t%.4f\t%s\n",
           name.c_str(), type.c_str(), K, kind.c_str(),
           rel(y_up, y_true), rel(y_aq, y_true),
           q4k ? std::to_string(rel(y_rp, y_true)).substr(0, 7).c_str() : "-",
           q4k ? std::to_string(rel(y_ex, y_true)).substr(0, 7).c_str() : "-",
           rel(y_r1, y_true), rel(y_m1, y_true), rel(y_r4, y_true), rel(y_m4, y_true),
           rel(y_m1, y_r1), rel(y_m4, y_r4),
           q4k ? std::to_string(rel(y_r1, y_ex)).substr(0, 7).c_str() : "-",
           maxdev(y_m1, y_r1), st.asum_max, st.ovf_frac,
           q4k ? (std::to_string(nclamp) + "/" + std::to_string(nsub)).c_str() : "-");
    fflush(stdout);
}

// Q4_0 with K a multiple of 256: the backend uses q4_0_32x256 -> kernel i8i4_hp (fp16 accumulation) and activation
// quantisation quantize_a_row_i8_hp. Weights: the same Q4_K rows, dequantised and requantised by stock quantize_row_q4_0_ref;
// the repack repack_q4_0_to_q4_0_256_32_bl_ref is lossless (the Q4_0 zero point 8 is already an integer).
// Activation layout per 256-block of one row (rvv_kernels.cpp, vlenb 128): 8 x [fp16 sub-block scale][32 int8],
// then 8 x fp16 sums, then the fp16 mean scale; value = q * sub-block scale * mean scale.
static void run_case_q40(const std::string & name, size_t K, size_t R, const std::vector<float> & w_src,
                         const std::vector<float> & a, const std::string & kind) {
    const size_t QB = 256, k_blks = K / QB, A_HP = 256 + 8 * 2 + 8 * 2 + 2;  // q8_hp_blk_size(256, true, true)
    const size_t row_b = (K / QK4_0) * sizeof(block_q4_0);
    std::vector<uint8_t> raw(R * row_b);
    std::vector<float>   w(R * K);
    for (size_t n = 0; n < R; n++) {
        quantize_row_q4_0_ref(w_src.data() + n * K, (block_q4_0 *) (raw.data() + n * row_b), K);
        dequantize_row_q4_0((const block_q4_0 *) (raw.data() + n * row_b), w.data() + n * K, K);
    }
    std::vector<double> y_true(M * R), y_up(M * R), y_aq(M * R, 0), y_ex(M * R, 0);
    for (int m = 0; m < M; m++)
        for (size_t n = 0; n < R; n++) {
            double s = 0;
            for (size_t k = 0; k < K; k++) s += (double) a[m * K + k] * w[n * K + k];
            y_true[m * R + n] = s;
        }
    std::vector<block_q8_0> q8(K / QK8_0);
    for (int m = 0; m < M; m++) {
        quantize_row_q8_0_ref(a.data() + m * K, q8.data(), K);
        for (size_t n = 0; n < R; n++) {
            float s;
            ggml_vec_dot_q4_0_q8_0_generic(K, &s, 0, raw.data() + n * row_b, 0, q8.data(), 0, 1);
            y_up[m * R + n] = s;
        }
    }
    ggml_tensor t;
    memset(&t, 0, sizeof(t));
    t.type  = GGML_TYPE_Q4_0;
    t.ne[0] = K; t.ne[1] = R; t.ne[2] = 1; t.ne[3] = 1;
    std::vector<uint8_t> packed(R * K + 4096);
    uint8_t * B = (uint8_t *) (((uintptr_t) packed.data() + 255) & ~(uintptr_t) 255);
    t.data = B;
    if (chk::repack_q4_0_256(&t, raw.data(), raw.size()) != 0) { fprintf(stderr, "%s: repack q4_0 failed\n", name.c_str()); exit(1); }

    std::vector<uint8_t> qa1(M * k_blks * A_HP + 64), qa4(M * k_blks * A_HP + 64);
    for (int m = 0; m < M; m++) lib_quant_row_hp(QB, a.data() + m * K, K, qa1.data() + m * k_blks * A_HP);
    lib_quant_4row_hp(QB, a.data(), K, qa4.data());
    for (int m = 0; m < M; m++)
        for (size_t b = 0; b < k_blks; b++) {
            const uint8_t * blk = qa1.data() + (m * k_blks + b) * A_HP;
            ggml_fp16_t     avh;
            memcpy(&avh, blk + A_HP - 2, 2);
            const double avg = ggml_fp16_to_fp32(avh);
            for (size_t kk = 0; kk < 8; kk++) {
                ggml_fp16_t sh;
                memcpy(&sh, blk + kk * 34, 2);
                const double sc = ggml_fp16_to_fp32(sh) * avg;
                for (size_t i = 0; i < 32; i++) {
                    const size_t k  = b * QB + kk * 32 + i;
                    const double aq = sc * (int8_t) blk[kk * 34 + 2 + i];
                    for (size_t n = 0; n < R; n++) y_aq[m * R + n] += aq * w[n * K + k];
                }
            }
        }
    y_ex = y_aq;  // lossless repack: exact result with quantised activations = aq

    const size_t row_stride_b = k_blks * sizeof(block_q4_0) * QB / QK4_0;  // get_repacked_block_type_size<q4_0,256,32>
    auto tiles = [&](int rows, const uint8_t * qa, float * c, bool use_ref) {
        for (size_t ni = 0; ni < R; ni += 32) {
            const uint8_t * bt = B + ni * row_stride_b;
            const size_t    nb = std::min<size_t>(32, R - ni);
            if (use_ref) {
                chk::ref_i8i4_hp(rows, QB, qa, bt, nullptr, c + ni, nb, k_blks, R);
            } else {
                size_t got = lib_gemm_i8i4_hp(QB, qa, bt, nullptr, c + ni, rows, nb, k_blks, R);
                if (got != (size_t) rows) { fprintf(stderr, "hp kernel returned %zu\n", got); exit(1); }
            }
        }
    };
    std::vector<float>  c(M * R);
    std::vector<double> y_m1(M * R), y_r1(M * R), y_m4(M * R), y_r4(M * R);
    for (int m = 0; m < M; m++) {
        tiles(1, qa1.data() + m * k_blks * A_HP, c.data(), false);
        for (size_t n = 0; n < R; n++) y_m1[m * R + n] = c[n];
        tiles(1, qa1.data() + m * k_blks * A_HP, c.data(), true);
        for (size_t n = 0; n < R; n++) y_r1[m * R + n] = c[n];
    }
    tiles(4, qa4.data(), c.data(), false);
    for (size_t i = 0; i < M * R; i++) y_m4[i] = c[i];
    tiles(4, qa4.data(), c.data(), true);
    for (size_t i = 0; i < M * R; i++) y_r4[i] = c[i];

    printf("%s\tQ4_0hp\t%zu\t%s\t%.5f\t%.5f\t-\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.3f\t-\t-\t-\n",
           name.c_str(), K, kind.c_str(), rel(y_up, y_true), rel(y_aq, y_true), rel(y_ex, y_true),
           rel(y_r1, y_true), rel(y_m1, y_true), rel(y_r4, y_true), rel(y_m4, y_true),
           rel(y_m1, y_r1), rel(y_m4, y_r4), rel(y_r1, y_ex), maxdev(y_m1, y_r1));
    fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <tensors_dir> <path/to/build/bin/libggml-cpu.so>\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];
    const char *      lib = argv[2];

    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd < 0 || write(fd, "0", 1) != 1) {
        fprintf(stderr, "set_ai_thread failed - IME2 is only available on A100 (AI) threads\n");
        return 1;
    }
    close(fd);

    void * h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    lib_gemm_i8i4  = (gemm_fn) dlsym(h, "_ZN16spacemit_kernels4ime216gemm_kernel_i8i4EmPKhS2_S2_Pfmmmm");
    lib_gemm_i8i8  = (gemm_fn) dlsym(h, "_ZN16spacemit_kernels4ime216gemm_kernel_i8i8EmPKhS2_S2_Pfmmmm");
    lib_quant_row  = (quant_fn) dlsym(h, "_ZN16spacemit_kernels3rvv17quantize_a_row_i8EmPKfmPh");
    lib_quant_4row = (quant_fn) dlsym(h, "_ZN16spacemit_kernels3rvv18quantize_a_4row_i8EmPKfmPh");
    lib_gemm_i8i4_hp  = (gemm_fn) dlsym(h, "_ZN16spacemit_kernels4ime219gemm_kernel_i8i4_hpEmPKhS2_S2_Pfmmmm");
    lib_quant_row_hp  = (quant_fn) dlsym(h, "_ZN16spacemit_kernels3rvv20quantize_a_row_i8_hpEmPKfmPh");
    lib_quant_4row_hp = (quant_fn) dlsym(h, "_ZN16spacemit_kernels3rvv21quantize_a_4row_i8_hpEmPKfmPh");
    if (!lib_gemm_i8i4 || !lib_gemm_i8i8 || !lib_quant_row || !lib_quant_4row || !lib_gemm_i8i4_hp ||
        !lib_quant_row_hp || !lib_quant_4row_hp) {
        fprintf(stderr, "dlsym: kernel symbols not found\n");
        return 1;
    }

    // manifest.tsv: name type ne0 rows row_bytes
    std::ifstream man(dir + "/manifest.tsv");
    std::string   line;
    std::getline(man, line);
    struct ent { std::string name, type; size_t ne0, rows, row_bytes; };
    std::vector<ent> ents;
    while (std::getline(man, line)) {
        std::istringstream is(line);
        ent e;
        is >> e.name >> e.type >> e.ne0 >> e.rows >> e.row_bytes;
        ents.push_back(e);
    }
    auto norm_of = [&](const std::string & name, const char * which) {
        std::string layer = name.substr(0, name.find('.', 4));  // blk.N
        auto        raw   = read_file(dir + "/" + layer + "." + which + ".weight.bin");
        std::vector<float> g(raw.size() / 4);
        memcpy(g.data(), raw.data(), raw.size());
        return g;
    };

    printf("tensor\ttype\tK\tacts\tup\taq\trp\tex\tref1\tm1\tref4\tm4\tm1_vs_ref1\tm4_vs_ref4\tref1_vs_ex\t"
           "m1_maxdev\tasum_max\tasum_ovf15\tzp_clamped\n");
    unsigned seed = 1;
    for (const auto & e : ents) {
        if (e.type != "Q4_K" && e.type != "Q6_K") continue;
        auto raw = read_file(dir + "/" + e.name + ".bin");
        std::vector<std::string> kinds;
        std::vector<float>       g;
        if (e.name.find("attn_q") != std::string::npos || e.name.find("attn_v") != std::string::npos) {
            g = norm_of(e.name, "attn_norm");
            kinds = { "norm", "pos" };
        } else if (e.name.find("ffn_gate") != std::string::npos || e.name.find("ffn_up") != std::string::npos) {
            g = norm_of(e.name, "ffn_norm");
            kinds = { "norm", "pos" };
        } else if (e.name.find("ffn_down") != std::string::npos) {
            kinds = { "swiglu", "pos" };
        } else {
            kinds = { "gauss", "pos" };
        }
        for (const auto & kind : kinds) {
            auto a = make_acts(kind, g, e.ne0, seed++);
            run_case(e.name, e.type, e.ne0, e.rows, raw, a, kind);
            if (e.type == "Q4_K") {  // the same weight rows and activations, through Q4_0 and the hp kernel
                std::vector<float> w(e.rows * e.ne0);
                for (size_t n = 0; n < e.rows; n++)
                    dequantize_row_q4_K((const block_q4_K *) (raw.data() + n * e.row_bytes), w.data() + n * e.ne0, e.ne0);
                run_case_q40(e.name, e.ne0, e.rows, w, a, kind);
            }
        }
    }
    return 0;
}
