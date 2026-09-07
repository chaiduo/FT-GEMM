# Hopper FP8 ABFT GEMM

This project implements a five-stage FP8 E4M3 GEMM optimization chain for
NVIDIA Hopper/SM90. A and B use FP8 storage, accumulation and output use FP32,
and every retained stage performs tile-local ABFT detection and single-point
correction.

The primary chain is intentionally limited to stages with a measured
steady-state throughput increase over the previous stage.

## Online Demo

GitHub Pages: [https://chaiduo.github.io/FT-GEMM/](https://chaiduo.github.io/FT-GEMM/)

The online page presents the S0-S4 optimization chain, ABFT workflow, source
excerpts, architecture diagrams, and measured performance. It is deployed
automatically from `master` when `web/**` or the Pages workflow changes.

To open the same page locally:

```bash
xdg-open web/index.html
```

## Optimization Chain

| Stage | Source | Main optimization | 4096^3 TFLOPS |
|---|---|---|---:|
| S0 | `src/s0_tiled_abft_fp8_gemm.cu` | Native 16x16 tiled GEMM + tile-local ABFT | 3.847 |
| S1 | `src/s1_tma_wgmma_abft_fp8_gemm.cu` | TMA double buffering + WGMMA | 128.738 |
| S2 | `src/s2_expected_cache_abft_fp8_gemm.cu` | Full expected-checksum cache | 132.524 |
| S3 | `src/s3_warpspecialized_abft_fp8_gemm.cu` | 128x128 CTA + Warp Specialization | 232.608 |
| S4 | `src/s4_cluster_multicast_abft_fp8_gemm.cu` | K128 stages + 2x1 CTA Cluster Multicast | 256.520 |

The progression is:

```text
Native tiled
    -> TMA/WGMMA
        -> expected checksum cache
            -> warp-specialized 128x128 mainloop
                -> K128 TMA + CTA Cluster Multicast
```

The former Native multi-stream, WGMMA batch-pipeline, and cross-CTA input-cache
experiments were removed from the primary chain because their measured
throughput was lower than the preceding retained stage.

## Benchmark

The latest unified measurement uses:

```text
GPU: physical GPU 1, NVIDIA H20
Shape: M=N=K=4096
Warmup: 3
Repeat: 10
Accumulation/output: FP32
```

| Stage | Expected preparation | Steady time | Steady TFLOPS |
|---|---:|---:|---:|
| S0 | 160.517 ms | 35.726 ms | 3.847 |
| S1 | 89.342 ms | 1.068 ms | 128.738 |
| S2 | 1.932 ms | 1.037 ms | 132.524 |
| S3 | 1.937 ms | 0.591 ms | 232.608 |
| S4 | 1.934 ms | 0.536 ms | 256.520 |

The steady-state column excludes one-time expected-checksum preparation.
When A/B change between calls, report first-call or amortized throughput too.
S4 reaches approximately 94.6%-95.3% of the measured 271.3 TFLOPS cuBLASLt
FP8 reference, depending on run-to-run GPU variation.

## ABFT Protocol

The implementation follows the strict ABFT ordering:

```text
1. Prepare expected row/column checksums from A and B.
2. Compute C = A x B.
3. Compute actual row/column checksums from C.
4. Compare expected and actual checksums.
5. Locate the row/column intersection and correct one corrupted element.
```

For one output tile:

```text
expected_row[r] = sum_k A[r,k] * sum_c B[c,k]
expected_col[c] = sum_k sum_r A[r,k] * B[c,k]
actual_row[r]   = sum_c C[r,c]
actual_col[c]   = sum_r C[r,c]
```

Only the case of exactly one abnormal row and one abnormal column is corrected.
Multiple abnormal rows or columns are reported but are not silently corrected.

## Build

Build all five retained stages:

```bash
make key_stages -j2
```

Build one stage:

```bash
make stage0
make stage1
make stage2
make stage3
make stage4
```

The CUTE/WGMMA stages require the configured CUTLASS header path in
`CUTLASS_ROOT` and compile for `sm_90a`.

## Run

Use physical GPU 1:

```bash
export CUDA_VISIBLE_DEVICES=1
```

Run the complete primary chain manually:

```bash
./build/stage0_tiled_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 --warmup 3 --repeat 10

./build/stage1_tma_wgmma_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 --warmup 3 --repeat 10

./build/stage2_expected_cache_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 --warmup 3 --repeat 10

./build/stage3_warpspecialized_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 --warmup 3 --repeat 10

./build/stage4_cluster_multicast_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 --warmup 3 --repeat 10
```

Run the final-stage random fault campaign:

```bash
make s4_fault_campaign \
  FAULT_TRIALS=1000 \
  FAULT_MIN_VALUE=1 \
  FAULT_MAX_VALUE=256
```

The campaign is implemented in
`src/s4_random_fault_campaign.cu` and uses the final S4 kernel path.

## Stage Details

### S0: Native Tiled ABFT

S0 is the correctness baseline. A 16x16 CUDA thread block loads A and B tiles
into shared memory, performs scalar FP32 accumulation, and runs tile-local
ABFT as separate kernels. It supports general positive shapes.

### S1: TMA-WGMMA

S1 replaces scalar multiply-adds with the Hopper
`m64n64k32.f32.e4m3.e4m3` WGMMA atom. TMA asynchronously fills double-buffered
shared-memory stages while the warpgroup issues matrix operations. Expected
checksums are prepared before the GEMM kernel, while actual checksum and
correction remain in the output path.

Shape requirements: `M % 64 == 0`, `K % 32 == 0`; N may have a boundary tile.

### S2: Full Expected-Checksum Cache

S2 removes expected row/column checksum generation from the hot GEMM path.
Input-side metadata and full expected checksum metadata are prepared once and
reused across steady-state calls with unchanged A/B.

Shape requirements: `M % 64 == 0`, `K % 32 == 0`; N may have a boundary tile.

### S3: Warp Specialization

S3 expands the CTA tile to 128x128 and assigns different roles to the CTA:

```text
producer warp       -> TMA loads
consumer warpgroups -> WGMMA
verifier warps      -> ABFT subtile reductions
```

This increases input reuse and WGMMA issue density while keeping verification
parallel with the mainloop.

Shape requirements: `M % 128 == 0`, `K % 32 == 0`; N may have a boundary tile.

### S4: K128 and CTA Cluster Multicast

S4 loads K=128 per TMA stage and executes four K32 WGMMA operations before
rotating the stage. A 2x1 CTA cluster multicasts the same A tile to two CTAs,
while each CTA consumes its own B tile. Shared-memory union storage keeps the
mainloop and epilogue within the occupancy budget.

Shape requirements: `M % 128 == 0`, `N % 256 == 0`, `K % 128 == 0`.

## Auxiliary Tools

Build the cuBLASLt FP8 reference:

```bash
make cublaslt_fp8
CUDA_VISIBLE_DEVICES=1 ./build/cublaslt_fp8_gemm_benchmark \
  --m 4096 --n 4096 --k 4096 --warmup 3 --repeat 10
```

`src/gemm_softmax_abft.cu` is a separate correctness prototype for a
GEMM-plus-Softmax invariant and is not part of the five-stage performance
chain.
