# Hopper FP8 ABFT GEMM Optimization History

This document records the design decisions behind the curated S0-S7
optimization chain. The project root `README.md` is the primary build and
benchmark reference; this file provides the technical history.

## Current Source Map

Every retained stage enables tile-local ABFT detection and single-point
correction.

| Stage | Source | Main change |
|---|---|---|
| S0 | `src/s0_tiled_abft_fp8_gemm.cu` | Native tiled FP8 GEMM and tile-local ABFT |
| S1 | `src/s1_pipeline_abft_fp8_gemm.cu` | Batched three-stream overlap |
| S2 | `src/s2_wgmma_pipeline_abft_fp8_gemm.cu` | CUTE/WGMMA mainloop |
| S3 | `src/s3_fused_online_abft_fp8_gemm.cu` | TMA-WGMMA fused online ABFT |
| S4 | `src/s4_cached_input_abft_fp8_gemm.cu` | Cross-CTA A/B checksum cache |
| S5 | `src/s5_expected_cache_abft_fp8_gemm.cu` | Full expected-checksum cache |
| S6 | `src/s6_warpspecialized_abft_fp8_gemm.cu` | `128x128` warp-specialized mainloop |
| S7 | `src/s7_cluster_multicast_abft_fp8_gemm.cu` | K128 stages and `2x1` TMA multicast |

Build the complete chain:

```bash
make key_stages -j
```

Run a stage on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 \
  ./build/stage7_cluster_multicast_abft_fp8_gemm \
  --m 4096 --n 4096 --k 4096 \
  --warmup 20 --repeat 50 --rel-tol 0.00002
```

## Common Contract

- A and B use FP8 E4M3 storage.
- Accumulation and C output use FP32.
- Inputs are generated deterministically on the GPU.
- Each output tile owns row and column checksums.
- Exactly one abnormal row and column identify a correctable output element.
- Benchmarks report kernel time, metadata time where applicable, and TFLOPS.
- Formal measurements use physical GPU 1 through `CUDA_VISIBLE_DEVICES=1`.

The WGMMA stages store A as row-major `[M,K]`, B as row-major `[N,K]` while
using it logically as transposed, and C as column-major `[N,M]`.

## ABFT Invariant

For an output element:

```text
C[r,c] = sum(k) A[r,k] * B[c,k]
```

Actual checksums are reduced from C:

```text
actual_row[r] = sum(c) C[r,c]
actual_col[c] = sum(r) C[r,c]
```

Expected checksums are derived from A and B:

```text
expected_row[r] = sum(k) A[r,k] * sum(c) B[c,k]
expected_col[c] = sum(k) sum(r) A[r,k] * B[c,k]
```

Detection uses a magnitude-weighted tolerance:

```text
abs(expected - actual)
    > abs_tol + rel_tol * max(1, scale)
```

If one row and one column fail, their intersection is corrected with:

```text
C[bad_row,bad_col] += 0.5 * (row_delta + col_delta)
```

## Optimization Progression

### S0: Native Tiled ABFT

S0 establishes the correctness baseline with a `16x16` shared-memory tiled
CUDA GEMM. GEMM and ABFT execute separately. Expected row and column checksums
are recomputed from A and B for each output tile, making checksum work the
dominant cost.

### S1: Batched Multi-Stream Pipeline

S1 partitions output tiles into batches and connects three nonblocking streams
with CUDA events:

```text
GEMM(batch i)
    -> checksum(batch i)
        -> detect/correct(batch i)
```

Independent batches allow:

```text
GEMM(i+1) || checksum(i) || ABFT(i-1)
```

The batch size remains configurable because overly small batches are dominated
by launch and event overhead.

### S2: WGMMA Pipeline

S2 replaces scalar CUDA multiply-adds with the Hopper
`m64n64k32.f32.e4m3.e4m3` WGMMA instruction through CUTE. It retains the
batched stream pipeline and caches input-side checksum metadata.

This stage demonstrates the Tensor Core throughput gain, but synchronous
global-to-shared movement and separate checksum scheduling still limit
performance.

### S3: Fully Online TMA-WGMMA ABFT

S3 introduces TMA double buffering and performs all ABFT work in the output
CTA:

```text
TMA load A/B
    -> WGMMA
       || expected checksum
    -> actual checksum from shared C
    -> detect/correct
    -> store C
```

It has no reusable metadata preparation pass and is therefore the fair
end-to-end result when A and B change for every GEMM call. Scalar expected
checksum arithmetic becomes the primary bottleneck after the WGMMA mainloop is
accelerated.

### S4: Cross-CTA Input Checksum Cache

S4 computes A tile sums/magnitudes and B tile sums/magnitudes once, then
reuses them across all output CTAs in the orthogonal matrix dimension. The
main kernel still forms complete expected row and column checksums.

The cache is valid only while A and B remain unchanged. A changing-input
workload must include metadata preparation in end-to-end time.

### S5: Full Expected-Checksum Cache

S5 moves complete expected row/column checksums and magnitude scales out of
the WGMMA K-loop. The metadata kernel assigns one warp to each checksum and
parallelizes the K reduction across 32 lanes.

The main kernel retains actual checksum reduction, threshold checks,
correction, and the final C store. Removing scalar A/B scans and associated
barriers brings its steady-state kernel into the no-ABFT mainloop performance
range.

### S6: Wide Warp-Specialized Mainloop

S6 expands the CTA tile to `128x128x32` and separates roles:

```text
producer:     1 TMA warp
consumers:    2 WGMMA warpgroups
verification: 4 warps for four 64x64 ABFT subtiles
TMA stages:   3
```

The retained `128x128` design uses 90 registers per thread and 93,440 bytes of
shared memory, allowing two active CTAs per SM. A tested `128x256` tile used
154 registers and 158 KiB of shared memory, reduced occupancy to one CTA per
SM, and was slower.

### S7: K128 and Cluster Multicast

S7 expands each TMA stage to K=128 and executes four K32 WGMMA operations per
stage. A `2x1` CTA cluster shares A through TMA multicast while each CTA loads
its own B tile.

Cluster-wide empty-barrier accounting prevents either CTA from overwriting a
multicast A stage while the peer CTA still consumes it. Mainloop A/B storage
and C epilogue storage use a union because their lifetimes do not overlap,
reducing shared memory from 167,168 to 100,608 bytes and restoring two active
CTAs per SM.

S7 requires:

```text
M % 128 == 0
N % 256 == 0
K % 128 == 0
```

## Verified Performance

Unified benchmark:

```text
GPU: physical GPU 1, NVIDIA H20
Shape: M=N=K=4096
Warmup: 3
Repeat: 10
Execution: serial
```

S2 uses its legacy calibrated threshold `rel_tol=0.05`. These historical
measurements used `rel_tol=0.002` for S3-S7; the current S7 default is
`rel_tol=2e-5` after the later calibration described below.

| Stage | Steady time | Metadata | First call | Steady TFLOPS |
|---|---:|---:|---:|---:|
| S0 | 925.504 ms | None | 925.504 ms | 0.1485 |
| S1 | 543.968 ms | None | 543.968 ms | 0.2527 |
| S2 | 35.240 ms | 20.411 ms | 55.651 ms | 3.90 |
| S3 | 4.139 ms | None | 4.139 ms | 33.21 |
| S4 | 3.454 ms | 0.135 ms | 3.425 ms | 39.79 |
| S5 | 1.035 ms | 1.982 ms | 3.015 ms | 132.85 |
| S6 | 0.587 ms | 1.934 ms | 2.522 ms | 234.08 |
| S7 | 0.532 ms | 1.944 ms | 2.476 ms | 258.52 |

S4-S7 steady-state results require reusable A/B-derived metadata. S3 is the
fully online comparison for changing inputs. First-call and amortized results
must include metadata preparation.

The independent cuBLASLt reference reaches 271.3 TFLOPS at the same shape.
S7 reaches approximately 95.3% of that measured library throughput.

## Non-Segmented Threshold Calibration

A K-segmented prototype was evaluated but not retained. Although K32, K64,
and K128 segments could detect an injected `+1` error, the best K128 variant
reached only 53.2 TFLOPS because every segment materialized and reloaded C,
shared storage grew to 167,168 bytes, and occupancy fell to one CTA/SM.

The retained S7 instead keeps full-K ABFT and lowers `rel_tol` from `2e-3` to
`2e-5`. On the deterministic `4096^3` input, thresholds from `2e-5` upward
produce zero false-positive tiles, while `1e-5` produces 710. The safety
margin at `2e-5` preserves S7's 258 TFLOPS performance.

For 1,000 random single faults with uniformly sampled magnitude `[1,256]`,
random sign, and seed 12345, the selected threshold detects 100% and verifies
correction for 98.1%. A separate fixed-magnitude `+/-1` campaign detects 11.1%
and verifies correction for 0.5%, so unit-magnitude errors remain below the
reliable sensitivity range without K segmentation.

## Historical Experiments

The original V0-V13 development sequence included intermediate files that are
not retained in the curated source tree. Their conclusions are preserved
here:

- A four-thread shuffle subgroup reduced online checksum overhead, but was
  superseded by cached input metadata.
- A 160-thread persistent CTA overlapped WGMMA and verification, but small
  matrices could not fill the pipeline efficiently.
- Padding shared C from stride 64 to 65 reduced bank conflicts by only about
  0.8%; most conflicts came from scalar access to WGMMA-swizzled A/B storage.
- Reducing the persistent grid from 390 to 373 CTAs lowered active warps and
  regressed performance.
- Using `warpgroup_wait<1>()` regressed performance because accumulator
  dependencies and delayed stage release blocked the producer.
- A rank-0-only multicast launch deadlocked because the peer CTA barrier never
  completed. Both cluster ranks must participate in their TMA copy slices.
- A CTA-local empty barrier allowed a peer producer to overwrite multicast A.
  Remote arrivals from both CTAs fixed the lifetime violation.

## Validation

The retained stages have been checked with:

- fault-free execution;
- single injected output faults;
- correction count and bad-tile count validation;
- boundary shapes supported by each stage;
- Compute Sanitizer on representative WGMMA/TMA configurations.

At `4096^3`, very small injected errors such as `+1` can fall below a
magnitude-scaled numerical tolerance. Fault injection is considered successful
only when exactly one bad tile is detected and corrected; an undetected
injection is not reported as PASS.
