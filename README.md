# ABFT FP8 GEMM for Hopper

This project implements an FP8 E4M3 ABFT GEMM optimization chain for NVIDIA
Hopper/SM90, with FP32 accumulation and output.

## Visualization Dashboard

Open [`web/index.html`](web/index.html) directly in a browser to explore the
dark optimization worklog, source excerpts, stage-level principle diagrams,
S0-S7 throughput progression, and cuBLASLt comparison.

**Online demo:** [https://chaiduo.github.io/FT-GEMM/](https://chaiduo.github.io/FT-GEMM/)

## Curated Optimization Chain

Each stage is a standalone source file under `src/`:

| Stage | File | Key change |
|---|---|---|
| S0 | `src/s0_tiled_abft_fp8_gemm.cu` | Native tiled GEMM with tile-local ABFT |
| S1 | `src/s1_pipeline_abft_fp8_gemm.cu` | Batched multi-stream GEMM/checksum/ABFT |
| S2 | `src/s2_wgmma_pipeline_abft_fp8_gemm.cu` | WGMMA and three-stream tile pipeline |
| S3 | `src/s3_fused_online_abft_fp8_gemm.cu` | TMA-WGMMA fused online ABFT |
| S4 | `src/s4_cached_input_abft_fp8_gemm.cu` | Cross-CTA A/B checksum cache |
| S5 | `src/s5_expected_cache_abft_fp8_gemm.cu` | Full expected-checksum cache |
| S6 | `src/s6_warpspecialized_abft_fp8_gemm.cu` | `128x128` warp-specialized mainloop |
| S7 | `src/s7_cluster_multicast_abft_fp8_gemm.cu` | K128 stages and `2x1` TMA multicast |

ABFT detection is enabled from S0 onward. GEMM-only implementations remain
available as comparison baselines but are not part of this primary chain.

## GEMM + Softmax Cross-Nonlinearity ABFT

`src/gemm_softmax_abft.cu` is a correctness-first prototype that detects a
single corrupted GEMM logit after row-wise Softmax. It is separate from S0-S7
because it demonstrates a nonlinear invariant, not a faster GEMM stage.

For `C = scale * A * B^T`, use the unique zero-sum position weight:

```text
w_j = 2*j - (N - 1)
sum_j w_j = 0
```

The FP8 inputs independently produce the expected logit projections:

```text
Expected0_i = scale * sum_k A_ik * (sum_j B_jk)
Expected1_i = scale * sum_k A_ik * (sum_j w_j * B_jk)
```

After Softmax, stable log-softmax values and the row `logZ` produce:

```text
Actual0_i = sum_j log(P_ij) + N * logZ_i
Actual1_i = sum_j w_j * log(P_ij)
```

For one pre-Softmax fault `C_if += delta`:

```text
Residual0 = Actual0_i - Expected0_i = delta
Residual1 = Actual1_i - Expected1_i = w_f * delta

w_f       = Residual1 / Residual0
fault_col = (w_f + N - 1) / 2
```

The prototype corrects the located logit by `-Residual0`, reruns Softmax, and
compares the recovered probability row with a clean reference. It also checks
the stored FP32 probability row sum. Stable log-softmax values are used
directly, so valid probability underflow is not treated as a fault.

Build and run:

```bash
make gemm_softmax_abft
CUDA_VISIBLE_DEVICES=1 ./build/gemm_softmax_abft \
  --m 128 --n 128 --k 128 \
  --fault-row 31 --fault-col 73 --fault-value 1
```

Run the clean false-positive check:

```bash
CUDA_VISIBLE_DEVICES=1 ./build/gemm_softmax_abft \
  --m 1024 --n 1024 --k 1024 --no-fault
```

An injected-fault run returns failure if the fault is missed, cannot be
located, or the recovered probabilities differ from the clean reference.
Thresholds are configurable with `--abs-tol`, `--rel-tol`,
`--row-sum-tol`, and `--location-tol`.

The covered fault boundary is the GEMM/logit path before Softmax. A fault in
A or B that identically affects both GEMM and the independent projection is a
common-mode error and requires protected inputs or another independent path.
A fault after the final probability store requires readback or a downstream
checksum.

Build all stages:

```bash
make key_stages -j
```

Run the final stage on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 \
  ./build/stage7_cluster_multicast_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 \
  --warmup 20 --repeat 50 --rel-tol 0.00002
```

Run an S7 random single-fault campaign:

```bash
make s7_fault_campaign \
  FAULT_TRIALS=1000 \
  FAULT_MIN_VALUE=1 \
  FAULT_MAX_VALUE=256 \
  FAULT_SEED=12345
```

The standalone campaign program is `src/s7_random_fault_campaign.cu`; it does
not change the benchmark behavior of `src/s7_cluster_multicast_abft_fp8_gemm.cu`.
Each trial selects a random `64x64` ABFT subtile, row, column, fault magnitude,
and sign. Magnitudes are sampled uniformly from
`[FAULT_MIN_VALUE, FAULT_MAX_VALUE]`.
The report includes detection rate, correction-attempt rate, verified
correction rate, missed faults, detected-but-uncorrected faults,
miscorrections, maximum post-correction residual, and per-magnitude-bin rates.

The current non-segmented S7 threshold is `abs_tol=0.01` and
`rel_tol=2e-5`. At `4096^3`, `rel_tol=2e-5` produces no false-positive tiles
for the deterministic benchmark input, while `rel_tol=1e-5` produces 710.
With 1,000 uniformly distributed random faults in `[1,256]`, the selected
threshold detects 100% and verifies correction for 98.1%. Exact-magnitude
`+/-1` faults remain below the reliable sensitivity range: detection is 11.1%
and verified correction is 0.5%.

Detailed optimization experiments are recorded in
`doc/NATIVE_OPTIMIZATION_PLAN.md`.

## Stage Performance

Benchmark environment:

```text
GPU: physical GPU 1, NVIDIA H20
Shape: M=N=K=4096
Warmup: 3
Repeat: 10
ABFT relative tolerance: 0.002
Execution: serial
```

S2 uses its separately calibrated legacy threshold `rel_tol=0.05`. The table
records the original S3-S7 measurements at `rel_tol=0.002`; the current S7
default is `rel_tol=2e-5`, which does not change the kernel instruction path.

| Stage | ABFT mode | Kernel/GEMM | Metadata | First call | 10-use amortized | Steady TFLOPS | Amortized TFLOPS |
|---|---|---:|---:|---:|---:|---:|---:|
| S0 Tiled ABFT | Online, separate | 50.189 ms GEMM + 875.315 ms ABFT | None | 925.504 ms | 925.504 ms | 0.15 | 0.15 |
| S1 Multi-stream ABFT | Online, overlapped | 543.968 ms | None | 543.968 ms | 543.968 ms | 0.25 | 0.25 |
| S2 WGMMA pipeline ABFT | Cached input metadata | 35.240 ms | 20.411 ms | 55.651 ms | 37.281 ms | 3.90 | 3.69 |
| S3 Fused online ABFT | Fully online | 4.139 ms | None | 4.139 ms | 4.139 ms | 33.21 | 33.21 |
| S4 Cross-CTA cache | Cached input metadata | 3.454 ms | 0.135 ms | 3.425 ms | 3.467 ms | 39.79 | 39.64 |
| S5 Expected cache | Cached | 1.035 ms | 1.982 ms | 3.015 ms | 1.233 ms | 132.85 | 111.49 |
| S6 Warp-specialized | Cached | 0.587 ms | 1.934 ms | 2.522 ms | 0.781 ms | 234.08 | 176.08 |
| S7 Cluster multicast | Cached | 0.532 ms | 1.944 ms | 2.476 ms | 0.726 ms | 258.52 | 189.29 |

Every stage performs ABFT detection and correction. S3 is the fully online
result when A and B change on every call. S4-S7 cache metadata derived from A
and B, so their steady-state results require unchanged inputs; use first-call
or amortized time when metadata preparation must be included. S7 reaches about
95% of the measured 271.3 TFLOPS cuBLASLt FP8 reference in steady state.

Run the first ABFT baseline:

```bash
make stage0
./build/stage0_tiled_abft_fp8_gemm --m 1024 --n 1024 --k 1024
```

The V3 native pipeline can be built and run independently:

```bash
make native_v3
./build/v3_native_pipeline_abft_fp8_gemm \
  --m 256 --n 256 --k 256 --batch-tiles 128
```

Build the independent Hopper WGMMA baseline:

```bash
make wgmma_v4
./build/v4_wgmma_fp8_gemm \
  --m 4096 --n 4096 --k 4096 \
  --warmup 2 --repeat 10
```

V4 uses one `64x64x32` FP8 WGMMA atom per 128-thread warpgroup and FP32
accumulation. It contains no checksum, fault injection, multi-stream pipeline,
or ABFT workspace, so it is the performance baseline for later fused versions.
It requires `M % 64 == 0` and `K % 32 == 0`; N may contain a partial final
tile.

Build the TMA double-buffered version:

```bash
make wgmma_v5
./build/v5_wgmma_tma_fp8_gemm \
  --m 4096 --n 4096 --k 4096 \
  --warmup 3 --repeat 20
```

V5 overlaps the next K tile's TMA load with the current tile's WGMMA
computation using two shared-memory stages and SM90 transaction barriers. On
the current H20, the serial `4096^3` comparison measured 18.45 TFLOPS for V4
and 130.64 TFLOPS for V5. V5 remains GEMM-only; ABFT fusion belongs to V6.

Build the fully fused online ABFT version:

```bash
make wgmma_v6
./build/v6_wgmma_fused_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 \
  --warmup 3 --repeat 20 --rel-tol 0.002
```

V6 derives expected checksums from each resident TMA A/B stage, computes actual checksums from the shared C tile, and performs detection and single-point correction before the only global C store. It has no separate metadata pass. At `4096^3` it measured 4.832 ms and 28.44 TFLOPS, a 357.6% online overhead over V5. The remaining bottleneck is scalar checksum work.

Build the optimized checksum version:

```bash
make wgmma_v7
./build/v7_wgmma_optimized_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 \
  --warmup 20 --repeat 50 --rel-tol 0.002
```

V7 uses four-thread shuffle subgroups for A/B column reductions, assigns row
and column checksums to separate halves of the warpgroup, and keeps expected
checksums in registers. Across three independent runs, the `4096^3` median
fell from 6.304 ms for V6 to 4.608 ms for V7, a 26.9% reduction.

Build the cross-CTA cached checksum version:

```bash
make wgmma_v8
./build/v8_wgmma_cached_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 \
  --warmup 20 --repeat 50 --rel-tol 0.002
```

V8 computes A metadata once per `64xK` tile and B metadata once per `64xK`
tile, then reuses it across every output CTA. The cache contains sums and
absolute sums and occupies 4 MiB at `4096^3`. Its median steady-state result
was 3.1235 ms (44.00 TFLOPS), 32.2% less time than the 4.608 ms V7 median.
Metadata preparation took about 0.097 ms, so a warm-clock first call was about
3.199 ms. The program reports metadata, first-call, steady-state, and amortized
times separately.

The cache is valid only while A and B remain unchanged. A workload that
changes either input on every call must include metadata preparation in its
end-to-end cost; V8 therefore has a fused detection/correction path but is not
the same fully online input-checksum design as V7.

Build the persistent three-stage version:

```bash
make wgmma_v9
./build/v9_wgmma_three_stage_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 \
  --warmup 20 --repeat 50 --rel-tol 0.002
```

V9 uses one 128-thread WGMMA warpgroup and one 32-thread verification warp in
each persistent CTA. Two shared C slots connect the compute and verification
groups with full/empty mbarriers. While one output tile is checked, corrected,
and stored, the WGMMA group advances through the next tile with TMA double
buffering and cached expected checksums.

At `4096^3`, three-process median steady-state time was 2.955 ms
(46.52 TFLOPS), 5.4% below V8. The CTA uses 43,776 bytes of shared memory,
71 registers per thread, and has five active blocks per SM on the test H20.
Small matrices do not contain enough tiles to fill the persistent pipeline and
can be slower than V8.

Build the K128 cluster-multicast version:

```bash
make wgmma_v13
CUDA_VISIBLE_DEVICES=1 \
  ./build/v13_wgmma_k128_cluster_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 \
  --warmup 20 --repeat 50 --rel-tol 0.002
```

V13 groups four `m64n128k32` operations into each `K=128` TMA stage and uses
a `2x1` CUDA cluster. The two CTAs share A through TMA multicast while loading
their own B tiles. Mainloop A/B storage and the C epilogue storage share the
same lifetime-aliased buffer, allowing three K128 stages while retaining two
active CTAs per SM.

On physical GPU 1, V13's three-run median was 0.5318 ms and 258.45 TFLOPS at
`4096^3`. This is 11.2% faster than V12 and 95.3% of the 271.3 TFLOPS
cuBLASLt baseline. Fifty-use throughput including metadata preparation was
about 240.9 TFLOPS.

Build the padded shared-C version:

```bash
make wgmma_v10
./build/v10_wgmma_padded_pipeline_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 \
  --warmup 20 --repeat 50 --rel-tol 0.002
```

V10 changes the shared C leading dimension from 64 to 65 to break the
32-way bank mapping in column checksum reads. Paired alternating runs measured
3.315 ms for V9 and 3.289 ms for V10, a 0.8% reduction. Nsight Compute measured
shared-load bank conflicts falling from 205.4 million to 201.7 million. Most
remaining conflicts come from scalar expected-checksum access to the
WGMMA-swizzled A/B stages.

CTA-count experiments also showed that the maximum resident grid of 390 CTAs
is faster than reducing the grid to 373 to balance the final tile wave. V10
therefore selects `SM count * active blocks per SM`, bounded by tile count.

Build the full expected-cache version:

```bash
make wgmma_v11
CUDA_VISIBLE_DEVICES=1 ./build/v11_wgmma_expected_cache_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 \
  --warmup 20 --repeat 50 --rel-tol 0.002
```

V11 precomputes complete row/column expected checksums and magnitude scales.
The preparation kernel assigns one warp to each checksum and reduces K across
32 lanes. The persistent WGMMA kernel therefore contains no expected-checksum
arithmetic or input-metadata synchronization; actual checksums, detection,
correction, and the final store remain active in the verification warp.

On physical GPU 1 at `4096^3`, paired runs measured 1.049-1.050 ms for V5
without ABFT and 1.038 ms for the V11 steady-state kernel. V11 therefore has
no measurable steady-state overhead in this test. Metadata preparation takes
about 1.933 ms and occupies 8 MiB. Including that cost, the first call is about
2.973 ms (46.2 TFLOPS); amortizing it across 50 uses gives about 1.076 ms
(127.7 TFLOPS), approximately 2.5% over V5. These reuse and first-call
measurements must not be conflated.

Build the wide warp-specialized version:

```bash
make wgmma_v12
CUDA_VISIBLE_DEVICES=1 \
  ./build/v12_wgmma_wide_warpspecialized_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 \
  --warmup 20 --repeat 50 --rel-tol 0.002
```

V12 applies 2D CTA tiling and warptiling to Hopper WGMMA. Each `128x128` CTA
uses a dedicated TMA producer warp, two `m64n128k32` consumer warpgroups,
three A/B shared-memory stages, and four verifier warps for its four `64x64`
ABFT subtiles. This increases input reuse and WGMMA issue density while
retaining tile-local detection and correction.

On physical GPU 1, V12's three-run median was 0.591 ms and 232.5 TFLOPS at
`4096^3`, 75.6% faster than V11 and 85.7% of the 271.3 TFLOPS cuBLASLt result. A tested
`128x256` configuration fell to 173.5 TFLOPS because 154 registers and
158 KiB shared memory limited it to one CTA per SM. Three TMA stages were
faster than two; four stages did not improve performance.

## cuBLASLt FP8 Reference

The standalone cuBLASLt benchmark provides a library-level FP8 baseline with
FP8 E4M3 inputs, FP32 accumulation, FP32 output, algorithm autotuning, and
sampled verification:

```bash
make cublaslt_fp8
./build/cublaslt_fp8_gemm_benchmark \
  --device 1 --m 4096 --n 4096 --k 4096 \
  --warmup 20 --repeat 50 --autotune-repeat 5
```

On physical GPU 1, three runs selected algorithm ID 66 and produced a median
of 0.5066 ms, 271.3 TFLOPS, or 91.7% of the configured 296 TFLOPS dense FP8
peak. `--fast-accum` reached 272.7 TFLOPS but increased sampled numerical
error, so the default comparison keeps standard FP32 accumulation.

This shows that the custom V5/V11 `64x64x32` one-warpgroup kernel is close to
its own implementation baseline, but still reaches only about 45% of the H20
dense FP8 peak. Closing that gap requires a larger production GEMM tile and
warp-specialized mainloop, not further ABFT epilogue tuning alone.

## What This Version Does

- Stores the main `A` and `B` matrices as CUDA FP8 E4M3 (`__nv_fp8_e4m3`).
- Extends them logically as `A'((M+1)xK)` and `B'(Kx(N+1))`.
- Computes `C' = A' * B'` with FP32 accumulation and FP32 output.
- The appended checksum row/column are FP32 metadata. Keeping checksums in FP32 is required; quantizing the sums back to FP8 would make `C'`'s checksum row/column differ from the sum of the main result even with no fault.
- Runs ABFT validation directly on the augmented result:
  - `C'[i,N]` is compared with `sum_j C'[i,j]`, for `i < M`.
  - `C'[M,j]` is compared with `sum_i C'[i,j]`, for `j < N`.
- Reports suspicious rows and columns. For a single output corruption, the first bad row and first bad column locate the corrupted element.
- Supports `--inject-fault` to perturb one output element and verify that ABFT detects it.

This is a correctness-oriented baseline. The GEMM kernel uses FP8 storage but scalar FP32 arithmetic after conversion. A production Hopper version should replace the GEMM body with cuBLASLt FP8 or CUTLASS/WGMMA while keeping the checksum flow as the validation layer.

## Build

```bash
make
```

The Makefile targets `sm_90`.

Build the CUTLASS/WGMMA version:

```bash
make cutlass
```

The CUTLASS target uses `sm_90a`, which is required by Hopper WGMMA FP8
instructions. Override `CUTLASS_ROOT` if CUTLASS is installed elsewhere.

Run it with:

```bash
./build/cutlass_fp8_gemm --m 1024 --n 1024 --k 1024 --abft-tile 128
```

The CUTLASS path partitions the output into independent ABFT tiles. `A` and
`B` are generated directly on the GPU with a deterministic counter-based
generator and quantized to FP8 there. Host copies retained after initialization
are only used for the sampled numerical reference check.

## Run

```bash
./build/abft_fp8_gemm
```

Example with a smaller shape:

```bash
./build/abft_fp8_gemm --m 256 --n 256 --k 256
```

Fault-injection check:

```bash
./build/abft_fp8_gemm --inject-fault
```

The normal run should print `abft_verification: PASS`. The fault-injection run should report at least one bad row and one bad column, then return a non-zero exit code.

The CUTLASS/WGMMA path uses a default checksum tolerance of `0.25`. Its independent
checksum reduction has a different accumulation order from the Tensor Core FP8 GEMM,
so its fault-free numerical residual is larger than the scalar baseline. Calibrate this
value on the target shape and input distribution before using the detector experimentally.

## Raw CUDA + CUTE WGMMA

Build the independent fused prototype:

```bash
make raw_wgmma
./build/raw_wgmma_fp8_abft --m 256 --n 192 --k 256
./build/raw_wgmma_fp8_abft --m 256 --n 192 --k 256 --inject-fault
```

`src/wgmma_fp8_abft.cu` does not use the CUTLASS GEMM or epilogue. It uses CUTE
only for the SM90 GMMA instruction, descriptor/layout construction, and register
fragment mapping. Each 128-thread CTA computes one `64x64` output tile with
`m64n64k32.f32.e4m3.e4m3` WGMMA stages. GEMM and expected-checksum generation
run as a three-stage batched pipeline on three nonblocking CUDA streams:

```text
GEMM batch i
    -> optimized tile checksum batch i
        -> tile-local ABFT detection/correction batch i
```

The next batch can enter GEMM while the previous batch is in checksum or ABFT. The batch size is controlled by `--batch-tiles` and defaults to 128. CUDA Graph replay is not currently exposed because the initial cross-stream capture topology could not be closed cleanly.
Input-side A/B tile sums and absolute sums are prepared once and reused by all
checksum batches. GPU-generated sparse random-sign metadata adds a second,
probabilistic checksum channel. The GEMM epilogue computes ordinary and random
actual row/column sums from its shared-memory C tile; the checksum batch only
subtracts these from cached expected values, so it no longer rereads C from
global memory. The random channel is calibrated separately because large
WGMMA numerical residuals can still mask a very small fault. The
output reports the one-time metadata preparation, each stream's kernel time,
and the overlapped end-to-end time. Detection uses an input-magnitude weighted
threshold:

```text
|actual - expected| > abs_tol + rel_tol * max(1, input_scale)
```

For a row check, `input_scale` is
`sum(k,j) |A[row,k] * B[k,j]|`; for a column check it is
`sum(k,i) |A[i,k] * B[k,col]|`. Configure the two terms with `--abs-tol` and `--rel-tol`. The sparse random
channel uses `--random-abs-tol` and `--random-rel-tol`; it is probabilistic and
can still miss a very small fault at 4096 scale after no-fault calibration.

This first raw prototype requires:

```text
M % 64 == 0
K % 32 == 0
```

The N dimension may have a final partial tile; it is zero-padded internally.
The current correctness baseline is `M=256, N=192, K=256`. Large shapes such
as `4096x4096x4096` execute successfully and report timing/TFLOPS. For that
shape, `--rel-tol 0.05` is the current no-fault calibration point; it suppresses
normal numerical false positives but can hide very small injected errors.

## Options

```text
--device N
--m N
--n N
--k N
--abft-tile N
--warmup N
--repeat N
--no-abft
--abs-tol X
--rel-tol X
--random-abs-tol X
--random-rel-tol X
--checksum-tol X
--inject-fault
```

`--no-abft` runs the same WGMMA GEMM without checksum or ABFT stages and prints
the GEMM-only baseline time.

## Current limitation

The remaining correctness gap is the large-shape numerical residual: a plus-one fault at 4096 cubed can still be hidden by the calibrated checksum threshold. The next optimization should use a higher-precision or independently computed validation path rather than lowering the threshold and reintroducing normal false positives.
