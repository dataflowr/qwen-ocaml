/* C stubs for hot kernels. Compiled with -mcpu=native; on Apple Silicon this
 * gives NEON + FEAT_DotProd (vdotq_s32). Each [@@noalloc] external that does
 * real work releases the OCaml runtime so other domains don't stall behind it.
 *
 * Milestone 2: fp32 GEMV via Accelerate cblas (AMX-backed) + bulk bf16->f32.
 * Milestone 3 will add the fused Q4_K x Q8_K GEMV.
 */
#include <caml/mlvalues.h>
#include <caml/bigarray.h>
#include <caml/threads.h>
#include <caml/alloc.h>
#include <caml/fail.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <arm_neon.h>

/* NEON fp32 dot product, 8-wide. n is a multiple of 32 for our quant blocks. */
static inline float qwen_dot_f32(const float *a, const float *b, int n) {
  float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  float s = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; i < n; i++) s += a[i] * b[i];
  return s;
}

/* Sanity stub so the library links from day one. Returns the dot product of
 * two float32 Bigarray.Array1 of length n. Replace/extend with SIMD kernels. */
CAMLprim value caml_qwen_sdot(value vA, value vB, value vN) {
  const float *a = (const float *) Caml_ba_data_val(vA);
  const float *b = (const float *) Caml_ba_data_val(vB);
  intnat n = Long_val(vN);
  caml_release_runtime_system();
  double acc = 0.0;
  for (intnat i = 0; i < n; i++) acc += (double)a[i] * (double)b[i];
  caml_acquire_runtime_system();
  return caml_copy_double(acc);
}

/* y = W * x, W row-major [rows x cols] (float32), x length cols, y length rows.
 * Delegates to Accelerate's cblas_sgemv (uses AMX on Apple Silicon). */
CAMLprim value caml_qwen_sgemv(value vW, value vX, value vY, value vRows, value vCols) {
  const float *W = (const float *) Caml_ba_data_val(vW);
  const float *X = (const float *) Caml_ba_data_val(vX);
  float *Y = (float *) Caml_ba_data_val(vY);
  intnat rows = Long_val(vRows);
  intnat cols = Long_val(vCols);
  caml_release_runtime_system();
  cblas_sgemv(CblasRowMajor, CblasNoTrans,
              (int) rows, (int) cols,
              1.0f, W, (int) cols,
              X, 1, 0.0f, Y, 1);
  caml_acquire_runtime_system();
  return Val_unit;
}

/* Bulk bf16 -> f32: read n little-endian uint16 at (src + byte_off), widen each
 * to f32 by placing it in the high 16 bits, store into dst[0..n). */
CAMLprim value caml_qwen_bf16_to_f32(value vSrc, value vOff, value vDst, value vN) {
  const uint8_t *src = (const uint8_t *) Caml_ba_data_val(vSrc);
  float *dst = (float *) Caml_ba_data_val(vDst);
  intnat off = Long_val(vOff);
  intnat n = Long_val(vN);
  const uint8_t *p = src + off;
  caml_release_runtime_system();
  for (intnat i = 0; i < n; i++) {
    uint32_t bits = ((uint32_t)(p[2*i] | (p[2*i + 1] << 8))) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    dst[i] = f;
  }
  caml_acquire_runtime_system();
  return Val_unit;
}

/* ===================================================================== */
/* Milestone 3: ggml quant dequant + fused GEMV kernels.                 */
/* Block math ported verbatim from llama.cpp ggml-quants.c semantics     */
/* (see scripts/ggml_block_layouts.md).                                  */
/* ===================================================================== */

/* ggml type ids */
#define GGML_TYPE_F32  0
#define GGML_TYPE_Q5_0 6
#define GGML_TYPE_Q8_0 8
#define GGML_TYPE_Q4_K 12
#define GGML_TYPE_Q6_K 14

/* block element counts and byte sizes */
#define QK8_0 32
#define QK5_0 32
#define QK_K  256
#define Q8_0_BYTES 34   /* fp16 d + int8 qs[32]                        */
#define Q5_0_BYTES 22   /* fp16 d + uint8 qh[4] + uint8 qs[16]         */
#define Q4_K_BYTES 144  /* fp16 d + fp16 dmin + uint8 sc[12] + qs[128] */
#define Q6_K_BYTES 210  /* uint8 ql[128] + uint8 qh[64] + int8 sc[16] + fp16 d */

/* IEEE half (LE u16) -> float. */
static inline float qwen_half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp  = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;                 /* +/- 0 */
    } else {
      /* subnormal: normalize */
      exp = 127 - 15 + 1;
      while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
      mant &= 0x3FF;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);  /* inf / nan */
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

static inline uint16_t qwen_rd_u16le(const uint8_t *p) {
  return (uint16_t)(p[0] | (p[1] << 8));
}

/* --- Q8_0: 34 bytes / 32 weights.  w[j] = d * qs[j] --- */
static void dequant_row_q8_0(const uint8_t *src, float *dst, int n) {
  int nb = n / QK8_0;
  for (int b = 0; b < nb; b++) {
    const uint8_t *p = src + (size_t)b * Q8_0_BYTES;
    float d = qwen_half_to_float(qwen_rd_u16le(p));
    const int8_t *qs = (const int8_t *)(p + 2);
    float *y = dst + (size_t)b * QK8_0;
    for (int j = 0; j < QK8_0; j++) y[j] = d * (float)qs[j];
  }
}

/* --- Q5_0: 22 bytes / 32 weights --- */
static void dequant_row_q5_0(const uint8_t *src, float *dst, int n) {
  int nb = n / QK5_0;
  for (int b = 0; b < nb; b++) {
    const uint8_t *p = src + (size_t)b * Q5_0_BYTES;
    float d = qwen_half_to_float(qwen_rd_u16le(p));
    const uint8_t *qh = p + 2;
    const uint8_t *qs = p + 6;
    uint32_t QH;
    memcpy(&QH, qh, 4);
    float *y = dst + (size_t)b * QK5_0;
    for (int j = 0; j < 16; j++) {
      uint8_t xh0 = (uint8_t)(((QH >> (j +  0)) << 4) & 0x10);
      uint8_t xh1 = (uint8_t)(((QH >> (j + 12))     ) & 0x10);
      int32_t q0 = ((qs[j] & 0x0F) | xh0) - 16;
      int32_t q1 = ((qs[j] >>   4) | xh1) - 16;
      y[j]      = d * (float)q0;
      y[j + 16] = d * (float)q1;
    }
  }
}

/* get_scale_min_k4 for Q4_K (6-bit scale + 6-bit min). */
static inline void q4k_get_scale_min(int j, const uint8_t *sc, uint8_t *d6, uint8_t *m6) {
  if (j < 4) {
    *d6 = sc[j]   & 63;
    *m6 = sc[j+4] & 63;
  } else {
    *d6 = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4);
    *m6 = (sc[j+4] >>   4) | ((sc[j-0] >> 6) << 4);
  }
}

/* --- Q4_K: 144 bytes / 256 weights --- */
static void dequant_row_q4_k(const uint8_t *src, float *dst, int n) {
  int nb = n / QK_K;
  for (int b = 0; b < nb; b++) {
    const uint8_t *p = src + (size_t)b * Q4_K_BYTES;
    float d    = qwen_half_to_float(qwen_rd_u16le(p));
    float dmin = qwen_half_to_float(qwen_rd_u16le(p + 2));
    const uint8_t *sc = p + 4;
    const uint8_t *q  = p + 16;
    float *y = dst + (size_t)b * QK_K;
    int is = 0;
    for (int n2 = 0; n2 < QK_K; n2 += 64) {
      uint8_t d6, m6;
      q4k_get_scale_min(is + 0, sc, &d6, &m6);
      float d1 = d * (float)d6, m1 = dmin * (float)m6;
      q4k_get_scale_min(is + 1, sc, &d6, &m6);
      float d2 = d * (float)d6, m2 = dmin * (float)m6;
      for (int l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0xF) - m1;
      for (int l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >>  4) - m2;
      q += 32; is += 2;
    }
  }
}

/* --- Q6_K: 210 bytes / 256 weights --- */
static void dequant_row_q6_k(const uint8_t *src, float *dst, int n) {
  int nb = n / QK_K;
  for (int b = 0; b < nb; b++) {
    const uint8_t *p = src + (size_t)b * Q6_K_BYTES;
    const uint8_t *ql = p;
    const uint8_t *qh = p + 128;
    const int8_t  *sc = (const int8_t *)(p + 192);
    float d = qwen_half_to_float(qwen_rd_u16le(p + 208));
    float *y = dst + (size_t)b * QK_K;
    for (int n2 = 0; n2 < QK_K; n2 += 128) {
      for (int l = 0; l < 32; l++) {
        int is = l / 16;
        int32_t q1 = ((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        int32_t q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        int32_t q3 = ((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        int32_t q4 = ((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l +  0] = d * (float)sc[is + 0] * (float)q1;
        y[l + 32] = d * (float)sc[is + 2] * (float)q2;
        y[l + 64] = d * (float)sc[is + 4] * (float)q3;
        y[l + 96] = d * (float)sc[is + 6] * (float)q4;
      }
      ql += 64; qh += 32; sc += 8; y += 128;
    }
  }
}

/* per-row byte stride = (cols/block_elems) * block_bytes for a given type. */
static size_t qwen_row_bytes(int ty, int cols) {
  switch (ty) {
    case GGML_TYPE_F32:  return (size_t)cols * 4;
    case GGML_TYPE_Q8_0: return (size_t)(cols / QK8_0) * Q8_0_BYTES;
    case GGML_TYPE_Q5_0: return (size_t)(cols / QK5_0) * Q5_0_BYTES;
    case GGML_TYPE_Q4_K: return (size_t)(cols / QK_K)  * Q4_K_BYTES;
    case GGML_TYPE_Q6_K: return (size_t)(cols / QK_K)  * Q6_K_BYTES;
    default:             return 0;
  }
}

/* dispatch dequant of one contiguous run of n weights of type ty. */
static void qwen_dequant_dispatch(int ty, const uint8_t *src, float *dst, int n) {
  switch (ty) {
    case GGML_TYPE_F32:  memcpy(dst, src, (size_t)n * 4);          break;
    case GGML_TYPE_Q8_0: dequant_row_q8_0(src, dst, n);            break;
    case GGML_TYPE_Q5_0: dequant_row_q5_0(src, dst, n);            break;
    case GGML_TYPE_Q4_K: dequant_row_q4_k(src, dst, n);            break;
    case GGML_TYPE_Q6_K: dequant_row_q6_k(src, dst, n);            break;
    default: break;
  }
}

/* caml_qwen_dequant_row(blob, byte_off, ty, dst, n):
 * dequant n weights of type ty at (blob_base + byte_off) into dst[0..n). */
CAMLprim value caml_qwen_dequant_row(value vBlob, value vOff, value vTy,
                                     value vDst, value vN) {
  const uint8_t *base = (const uint8_t *) Caml_ba_data_val(vBlob);
  float *dst = (float *) Caml_ba_data_val(vDst);
  intnat off = Long_val(vOff);
  int ty = (int) Long_val(vTy);
  intnat n  = Long_val(vN);
  /* validate while the runtime is still held (caml_invalid_argument may raise). */
  intnat blob_len = Caml_ba_array_val(vBlob)->dim[0];
  size_t span = qwen_row_bytes(ty, (int) n);
  if (n <= 0 || off < 0 || (intnat)span < 0 ||
      off + (intnat) span > blob_len)
    caml_invalid_argument("dequant_row: out-of-range tensor access");
  const uint8_t *src = base + off;
  caml_release_runtime_system();
  qwen_dequant_dispatch(ty, src, dst, (int) n);
  caml_acquire_runtime_system();
  return Val_unit;
}

/* Number of performance (P) cores, queried once. M3 Max = 12. */
static int qwen_pcores(void) {
  static int v = -1;
  if (v < 0) {
    int n = 0; size_t sz = sizeof(n);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &n, &sz, NULL, 0) != 0 || n < 1) {
      sz = sizeof(n);
      if (sysctlbyname("hw.logicalcpu", &n, &sz, NULL, 0) != 0 || n < 1) n = 8;
    }
    v = n;
  }
  return v;
}

/* P3 tunables (read once): dispatch chunk count and the serial-cutoff row
 * count. Lets us sweep threading granularity without recompiling.
 *
 * KEY P3 RESULT: the original fixed 32 chunks is SLOWER than matching the chunk
 * count to the available P-cores. dispatch_apply(32, AUTO) spills work onto the
 * 4 slow E-cores, which become stragglers at the implicit barrier. Sweep on
 * M3 Max (12 P-cores), interleaved decode tok/s: 32->23.8, 12->27.0, 10->28.9.
 * The optimum is ~2 BELOW the P-core count: using all 12 P-cores starves the
 * OCaml main thread + GCD coordinator, so leaving ~2 free is best. Default =
 * max(1, P-cores - 2); this is numerically EXACT (only repartitions rows) and
 * gives ~+21% decode here. Override with QWEN_NCHUNKS for the scaling sweep. */
static int qwen_nchunks(void) {
  static int v = -1;
  if (v < 0) {
    const char *s = getenv("QWEN_NCHUNKS");
    v = s ? atoi(s) : qwen_pcores() - 2;
    if (v < 1) v = 1;
  }
  return v;
}
static int qwen_serial_rows(void) {
  static int v = -1;
  if (v < 0) { const char *s = getenv("QWEN_SERIAL_ROWS"); v = s ? atoi(s) : 256; if (v < 0) v = 0; }
  return v;
}

/* Fused GEMV: out[r] = dot(dequant(row r), x[0..cols)) for r in 0..rows.
 * Rows are contiguous; per-row byte stride from qwen_row_bytes. */
CAMLprim value caml_qwen_qmatvec(value vBlob, value vBase, value vTy,
                                 value vX, value vOut, value vRows, value vCols) {
  const uint8_t *base = (const uint8_t *) Caml_ba_data_val(vBlob);
  const float *x = (const float *) Caml_ba_data_val(vX);
  float *out = (float *) Caml_ba_data_val(vOut);
  intnat base_off = Long_val(vBase);
  int ty   = (int) Long_val(vTy);
  intnat rows = Long_val(vRows);
  intnat cols = Long_val(vCols);

  /* Validate while the runtime is still held (caml_invalid_argument raises).
     The 16384 cap keeps the per-task `float scratch[cols]` VLA well within the
     GCD worker stack (all Qwen widths are <= 6144). */
  intnat blob_len = Caml_ba_array_val(vBlob)->dim[0];
  size_t stride = qwen_row_bytes(ty, (int) cols);
  if (cols <= 0 || cols > 16384 || rows < 0 || base_off < 0 ||
      base_off + rows * (intnat) stride > blob_len)
    caml_invalid_argument("qmatvec: out-of-range tensor access");

  caml_release_runtime_system();
  const uint8_t *src0 = base + base_off;

  if (ty == GGML_TYPE_F32) {
    /* no dequant needed; Accelerate GEMV directly on the mmap. */
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int) rows, (int) cols,
                1.0f, (const float *)src0, (int) cols, x, 1, 0.0f, out, 1);
    caml_acquire_runtime_system();
    return Val_unit;
  }

  /* All quant types: fp32 dequant into per-task scratch + NEON fp dot. This is
     numerically exact (fp32 activations) and matches llama.cpp's greedy output.
     Parallelize across rows with a FIXED chunk count (~cores) so each GCD task
     does substantial work and launch overhead is amortized; small matvecs run
     serially to skip dispatch overhead entirely. */
  const int icols = (int) cols;
  if (rows <= qwen_serial_rows()) {
    float scratch[icols];
    for (intnat r = 0; r < rows; r++) {
      qwen_dequant_dispatch(ty, src0 + (size_t)r * stride, scratch, icols);
      out[r] = qwen_dot_f32(scratch, x, icols);
    }
  } else {
    size_t n_chunks = (size_t) qwen_nchunks();
    size_t chunk = ((size_t)rows + n_chunks - 1) / n_chunks;
    dispatch_apply(n_chunks, DISPATCH_APPLY_AUTO, ^(size_t c) {
      float scratch[icols];
      size_t r0 = c * chunk;
      size_t r1 = r0 + chunk; if (r1 > (size_t)rows) r1 = rows;
      for (size_t r = r0; r < r1; r++) {
        qwen_dequant_dispatch(ty, src0 + r * stride, scratch, icols);
        out[r] = qwen_dot_f32(scratch, x, icols);
      }
    });
  }
  caml_acquire_runtime_system();
  return Val_unit;
}

/* bytecode entry point (7 args). */
CAMLprim value caml_qwen_qmatvec_bc(value *argv, int argn) {
  (void)argn;
  return caml_qwen_qmatvec(argv[0], argv[1], argv[2], argv[3],
                           argv[4], argv[5], argv[6]);
}
