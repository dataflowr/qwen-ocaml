# GGML quant block layouts (QK_K=256, QK=32) — exact, for the dequant kernels

All scales `d`/`dmin` are fp16 (little-endian). Port these verbatim from
llama.cpp `ggml/src/ggml-quants.c`. Half→float: read u16 LE, standard IEEE half.

## Q8_0 — 34 bytes / 32 weights
struct { fp16 d; int8 qs[32]; }   // w[j] = d * qs[j]

## Q5_0 — 22 bytes / 32 weights
struct { fp16 d; uint8 qh[4]; uint8 qs[16]; }
  uint32 QH; memcpy(&QH, qh, 4);
  for j in 0..15:
    xh0 = ((QH >> (j+ 0)) << 4) & 0x10;
    xh1 = ((QH >> (j+12))     ) & 0x10;
    w[j]    = d * (((qs[j] & 0x0F) | xh0) - 16);
    w[j+16] = d * (((qs[j] >>   4) | xh1) - 16);

## Q4_K — 144 bytes / 256 weights ("M" superblock)
struct { fp16 d; fp16 dmin; uint8 scales[12]; uint8 qs[128]; }
  get_scale_min_k4(j, sc[], &d6, &m6):   // 6-bit scale + 6-bit min per sub-block
    if j < 4:  d6 = sc[j] & 63;            m6 = sc[j+4] & 63;
    else:      d6 = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4);
               m6 = (sc[j+4] >>   4) | ((sc[j-0] >> 6) << 4);
  dequant: is=0; q=qs;  for n in 0,64,128,192 (step 64):
    get_scale_min_k4(is+0,...): d1 = d*d6; m1 = dmin*m6;
    get_scale_min_k4(is+1,...): d2 = d*d6; m2 = dmin*m6;
    for l in 0..31: *y++ = d1*(q[l] & 0xF) - m1;
    for l in 0..31: *y++ = d2*(q[l] >> 4) - m2;
    q += 32; is += 2;

## Q6_K — 210 bytes / 256 weights
struct { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; fp16 d; }
  for n in 0,128 (step 128):  // ql+=64, qh+=32, sc+=8, y+=128 each pass
    for l in 0..31:
      is = l/16;
      q1 = ((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      q2 = ((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      q3 = ((ql[l+ 0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      q4 = ((ql[l+32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      y[l+ 0] = d*sc[is+0]*q1;  y[l+32] = d*sc[is+2]*q2;
      y[l+64] = d*sc[is+4]*q3;  y[l+96] = d*sc[is+6]*q4;

## GGML type enum ids (subset, for dispatch)
F32=0, F16=1, Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9,
Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15

## This model uses: F32(0), Q5_0(6), Q8_0(8), Q4_K(12), Q6_K(14).

## Block byte sizes (bytes per (super)block): Q8_0=34/32, Q5_0=22/32,
## Q4_K=144/256, Q6_K=210/256, F32=4/1.
