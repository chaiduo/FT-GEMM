# Hopper FP8 ABFT GEMM: S0-S4 Optimization Plan

This document describes the retained five-stage chain in
`/data01/cd_workspace/FT-GEMM`. The stage numbers now reflect the final
monotonic performance chain rather than the chronological order of discarded
experiments.

## Source Map

| Stage | Source | Main change |
|---|---|---|
| S0 | `src/s0_tiled_abft_fp8_gemm.cu` | Native tiled GEMM + tile-local ABFT |
| S1 | `src/s1_tma_wgmma_abft_fp8_gemm.cu` | TMA double buffering + WGMMA |
| S2 | `src/s2_expected_cache_abft_fp8_gemm.cu` | Full expected-checksum cache |
| S3 | `src/s3_warpspecialized_abft_fp8_gemm.cu` | 128x128 CTA + Warp Specialization |
| S4 | `src/s4_cluster_multicast_abft_fp8_gemm.cu` | K128 TMA + 2x1 CTA multicast |

The former Native multi-stream, WGMMA batch-pipeline, and cross-CTA input-cache
implementations are removed. They remain part of the design history only when
needed to explain why they were not retained.

## Common ABFT Contract

Every retained stage follows:

```text
prepare expected checksum
    -> compute GEMM
        -> compute actual checksum
            -> compare, locate, and correct
```

For one output tile:

```text
expected_row[r] = sum_k A[r,k] * sum_c B[c,k]
expected_col[c] = sum_k sum_r A[r,k] * B[c,k]
actual_row[r]   = sum_c C[r,c]
actual_col[c]   = sum_r C[r,c]
```

The correction path is enabled only when exactly one row and one column fail.
Their intersection identifies the corrupted element:

```text
C[bad_row,bad_col] += 0.5 * (row_delta + col_delta)
```

## Performance Target

Unified measurement:

```text
GPU: physical GPU 1, NVIDIA H20
Shape: M=N=K=4096
Warmup: 3
Repeat: 10
```

| Stage | Expected preparation | Steady time | Steady TFLOPS | Increment |
|---|---:|---:|---:|---:|
| S0 | 160.517 ms | 35.726 ms | 3.847 | baseline |
| S1 | 89.342 ms | 1.068 ms | 128.738 | +3246% |
| S2 | 1.932 ms | 1.037 ms | 132.524 | +2.94% |
| S3 | 1.937 ms | 0.591 ms | 232.608 | +75.52% |
| S4 | 1.934 ms | 0.536 ms | 256.520 | +10.28% |

Steady-state time excludes one-time expected metadata preparation. When A/B
change between calls, first-call and amortized values must also be reported.

## S0: Native Tiled ABFT

S0 provides the correctness and cost baseline:

```text
one CTA -> one 16x16 output tile
shared-memory A/B tiles
scalar FP32 multiply-add
separate actual checksum and correction kernels
```

This implementation keeps the control flow simple and supports general
positive shapes. Its limitations are low data reuse, scalar arithmetic, and
repeated global-memory traffic for checksum work.

## S1: TMA-WGMMA

S1 changes the compute and data movement primitives:

```text
TMA -> shared-memory stage
WGMMA -> FP8 matrix multiply
FP32 accumulator -> C tile
actual checksum + ABFT -> output path
```

The WGMMA atom is `m64n64k32.f32.e4m3.e4m3`. TMA asynchronously moves A/B
tiles while the warpgroup issues matrix instructions. The two-stage shared
buffer hides part of global-memory latency.

Requirements: `M % 64 == 0`, `K % 32 == 0`; N may have a boundary tile.

## S2: Full Expected-Checksum Cache

S2 moves all expected-checksum work out of the repeated GEMM path:

```text
input metadata
    -> expected row/column checksums and scales
        -> repeated TMA-WGMMA + actual checksum + ABFT
```

This stage is useful when A and B remain unchanged across repeated GEMM calls.
The preparation cost is reported separately rather than hidden inside the
steady-state number.

Requirements: `M % 64 == 0`, `K % 32 == 0`; N may have a boundary tile.

## S3: Warp Specialization

S3 uses a 128x128 CTA tile and separates responsibilities:

```text
producer warp       -> TMA load and stage management
consumer warpgroup  -> WGMMA accumulation
consumer warpgroup  -> WGMMA accumulation
verifier warps      -> parallel ABFT subtile reductions
```

The larger CTA tile increases A/B reuse. Multiple consumer warpgroups improve
WGMMA issue density, while independent verifier warps prevent one serial
verification warp from stalling the epilogue.

Requirements: `M % 128 == 0`, `K % 32 == 0`; N may have a boundary tile.

## S4: K128 and CTA Cluster Multicast

S4 is the final architecture-specific version:

```text
CTA tile: 128x128
TMA stage: K=128
WGMMA: four K32 operations per TMA stage
cluster: 2x1
A tile: TMA multicast
B tile: independently loaded by each CTA
```

The K128 stage amortizes TMA and barrier overhead. CTA multicast prevents both
CTAs from independently loading the same A tile. A shared-memory union between
the mainloop and epilogue keeps the allocation within the occupancy budget.

Requirements: `M % 128 == 0`, `N % 256 == 0`, `K % 128 == 0`.

## Validation

Build the retained chain:

```bash
make key_stages -j2
```

Run the final random fault campaign:

```bash
make s4_fault_campaign FAULT_TRIALS=1000
```

Use `CUDA_VISIBLE_DEVICES=1` so the process selects physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 ./build/stage4_cluster_multicast_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 --warmup 3 --repeat 10
```
