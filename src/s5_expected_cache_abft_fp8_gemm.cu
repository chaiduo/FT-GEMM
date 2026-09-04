#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cute/algorithm/cooperative_gemm.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/arch/mma_sm90_gmma_ext.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/device_kernel.h"

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(error));                            \
      std::exit(EXIT_FAILURE);                                              \
    }                                                                      \
  } while (0)

using Element = cutlass::float_e4m3_t;
using Accumulator = float;
using TileM = cute::_64;
using TileN = cute::_64;
using TileK = cute::_32;
using TileShape = cute::Shape<TileM, TileN, TileK>;
using GmmaOp = decltype(cute::GMMA::ss_op_selector<
    Element, Element, Accumulator, TileShape, cute::GMMA::Major::K,
    cute::GMMA::Major::K>());
using TiledMma = decltype(cute::make_tiled_mma(GmmaOp{}));
using SmemLayoutA = decltype(cute::tile_to_shape(
    cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
    cute::Shape<TileM, TileK>{}));
using SmemLayoutB = decltype(cute::tile_to_shape(
    cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
    cute::Shape<TileN, TileK>{}));
constexpr int kStages = 2;
constexpr int kOutputStages = 2;
using SmemLayoutAStages = decltype(cute::tile_to_shape(
    cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
    cute::Shape<TileM, TileK, cute::Int<kStages>>{},
    cute::Step<cute::_1, cute::_2, cute::_3>{}));
using SmemLayoutBStages = decltype(cute::tile_to_shape(
    cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
    cute::Shape<TileN, TileK, cute::Int<kStages>>{},
    cute::Step<cute::_1, cute::_2, cute::_3>{}));
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

constexpr int kComputeThreads = 128;
constexpr int kVerifyThreads = 32;
constexpr int kPipelineThreads = kComputeThreads + kVerifyThreads;
constexpr int kM = 64;
constexpr int kN = 64;
constexpr int kK = 32;
constexpr int kCStride = kM + 1;
constexpr uint32_t kTmaTransactionBytes =
    (kM * kK + kN * kK) * sizeof(Element);

struct PipelineSharedStorage {
  alignas(8) uint64_t tma_full_barrier[kStages];
  alignas(8) uint64_t output_full_barrier[kOutputStages];
  alignas(8) uint64_t output_empty_barrier[kOutputStages];
  alignas(128) Element a[cute::cosize_v<SmemLayoutAStages>];
  alignas(128) Element b[cute::cosize_v<SmemLayoutBStages>];
  alignas(128) float c[kOutputStages][kCStride * kN];
  float deltas[kOutputStages][kM + kN];
  int tile_ids[kOutputStages];
  int valid_cols[kOutputStages];
  int bad_row_count[kOutputStages];
  int bad_col_count[kOutputStages];
  int bad_row[kOutputStages];
  int bad_col[kOutputStages];
};

struct Options {
  int m = 256;
  int n = 256;
  int k = 256;
  int warmup = 2;
  int repeat = 10;
  int verify_samples = 8;
  float verify_abs_tolerance = 2.5e-1f;
  float verify_rel_tolerance = 1.0e-2f;
  float abft_abs_tolerance = 1.0e-2f;
  float abft_rel_tolerance = 1.0e-5f;
  float fault_value = 1.0f;
  bool inject_fault = false;
};

static int div_up(int x, int y) {
  return (x + y - 1) / y;
}

static int get_int(int argc, char** argv, const char* name, int value) {
  size_t length = std::strlen(name);
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
      return std::atoi(argv[i + 1]);
    }
    if (std::strncmp(argv[i], name, length) == 0 &&
        argv[i][length] == '=') {
      return std::atoi(argv[i] + length + 1);
    }
  }
  return value;
}

static float get_float(int argc, char** argv, const char* name, float value) {
  size_t length = std::strlen(name);
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
      return std::strtof(argv[i + 1], nullptr);
    }
    if (std::strncmp(argv[i], name, length) == 0 &&
        argv[i][length] == '=') {
      return std::strtof(argv[i] + length + 1, nullptr);
    }
  }
  return value;
}

static Options parse_options(int argc, char** argv) {
  Options options;
  options.m = get_int(argc, argv, "--m", options.m);
  options.n = get_int(argc, argv, "--n", options.n);
  options.k = get_int(argc, argv, "--k", options.k);
  options.warmup = get_int(argc, argv, "--warmup", options.warmup);
  options.repeat = get_int(argc, argv, "--repeat", options.repeat);
  options.verify_samples =
      get_int(argc, argv, "--verify-samples", options.verify_samples);
  options.verify_abs_tolerance = get_float(
      argc, argv, "--verify-abs-tol", options.verify_abs_tolerance);
  options.verify_rel_tolerance = get_float(
      argc, argv, "--verify-rel-tol", options.verify_rel_tolerance);
  options.abft_abs_tolerance =
      get_float(argc, argv, "--abs-tol", options.abft_abs_tolerance);
  options.abft_rel_tolerance =
      get_float(argc, argv, "--rel-tol", options.abft_rel_tolerance);
  options.fault_value =
      get_float(argc, argv, "--fault-value", options.fault_value);

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--inject-fault") == 0) {
      options.inject_fault = true;
    }
    if (std::strcmp(argv[i], "--help") == 0) {
      std::printf(
          "Usage: %s [--m N --n N --k N --warmup N --repeat N]\n"
          "       [--verify-samples N --verify-abs-tol X "
          "--verify-rel-tol X]\n"
          "       [--abs-tol X --rel-tol X --fault-value X "
          "--inject-fault]\n"
          "Constraints: M %% 64 == 0 and K %% 32 == 0; "
          "N may have a boundary tile.\n",
          argv[0]);
      std::exit(EXIT_SUCCESS);
    }
  }

  if (options.m <= 0 || options.n <= 0 || options.k <= 0 ||
      options.m % kM != 0 || options.k % kK != 0 ||
      options.warmup < 0 || options.repeat <= 0 ||
      options.verify_samples < 0 || options.verify_abs_tolerance < 0.0f ||
      options.verify_rel_tolerance < 0.0f ||
      options.abft_abs_tolerance < 0.0f ||
      options.abft_rel_tolerance < 0.0f || options.fault_value <= 0.0f) {
    std::fprintf(
        stderr,
        "Invalid options. Require M%%64==0, K%%32==0, and positive shape.\n");
    std::exit(EXIT_FAILURE);
  }
  return options;
}

__device__ __forceinline__ uint32_t mix_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

__global__ void generate_fp8_kernel(Element* output, size_t count,
                                    uint32_t seed) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  uint32_t bits = mix_u32(seed + static_cast<uint32_t>(index));
  float unit = static_cast<float>(bits & 0x00ffffffU) / 16777216.0f;
  output[index] = Element(2.0f * unit - 1.0f);
}

__global__ void cache_input_tile_checksums_kernel(
    const Element* a, const Element* b, float* a_sums,
    float* a_magnitudes, float* b_sums, float* b_magnitudes, int n, int k,
    int tile_rows, int tile_cols) {
  int tile_id = blockIdx.x;
  if (tile_id < tile_rows) {
    int row_begin = tile_id * kM;
    for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
      float sum = 0.0f;
      float magnitude = 0.0f;
#pragma unroll 8
      for (int row = 0; row < kM; ++row) {
        float value = static_cast<float>(a[(row_begin + row) * k + kk]);
        sum += value;
        magnitude += fabsf(value);
      }
      size_t index = static_cast<size_t>(tile_id) * k + kk;
      a_sums[index] = sum;
      a_magnitudes[index] = magnitude;
    }
  } else {
    int tile_col = tile_id - tile_rows;
    if (tile_col >= tile_cols) {
      return;
    }
    int col_begin = tile_col * kN;
    int valid_cols = min(kN, n - col_begin);
    for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
      float sum = 0.0f;
      float magnitude = 0.0f;
#pragma unroll 8
      for (int col = 0; col < kN; ++col) {
        if (col < valid_cols) {
          float value =
              static_cast<float>(b[(col_begin + col) * k + kk]);
          sum += value;
          magnitude += fabsf(value);
        }
      }
      size_t index = static_cast<size_t>(tile_col) * k + kk;
      b_sums[index] = sum;
      b_magnitudes[index] = magnitude;
    }
  }
}

__global__ void cache_expected_tile_checksums_kernel(
    const Element* a, const Element* b, const float* a_sums,
    const float* a_magnitudes, const float* b_sums,
    const float* b_magnitudes, float* expected_rows,
    float* expected_cols, float* row_scales, float* col_scales,
    int n, int k, int tile_rows, int tile_cols) {
  constexpr int kWarpSize = 32;
  int warp = threadIdx.x / kWarpSize;
  int lane = threadIdx.x % kWarpSize;
  int warps_per_block = blockDim.x / kWarpSize;
  int check_id = blockIdx.x * warps_per_block + warp;
  int checks_per_tile = kM + kN;
  int total_checks = tile_rows * tile_cols * checks_per_tile;
  if (check_id >= total_checks) {
    return;
  }
  int tile_id = check_id / checks_per_tile;
  int check_index = check_id % checks_per_tile;
  int tile_row = tile_id / tile_cols;
  int tile_col = tile_id % tile_cols;
  int valid_cols = min(kN, n - tile_col * kN);
  bool is_row = check_index < kM;
  int row_or_col = is_row ? check_index : check_index - kM;
  float expected = 0.0f;
  float scale = 0.0f;

  if (is_row) {
    int row = row_or_col;
    int global_row = tile_row * kM + row;
    for (int kk = lane; kk < k; kk += kWarpSize) {
      float avalue =
          static_cast<float>(a[global_row * k + kk]);
      expected += avalue * b_sums[tile_col * k + kk];
      scale += fabsf(avalue) *
               b_magnitudes[tile_col * k + kk];
    }
  } else if (row_or_col < valid_cols) {
    int global_col = tile_col * kN + row_or_col;
    for (int kk = lane; kk < k; kk += kWarpSize) {
      float bvalue =
          static_cast<float>(b[global_col * k + kk]);
      expected += a_sums[tile_row * k + kk] * bvalue;
      scale += a_magnitudes[tile_row * k + kk] * fabsf(bvalue);
    }
  }

#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    expected += __shfl_down_sync(0xffffffffU, expected, offset);
    scale += __shfl_down_sync(0xffffffffU, scale, offset);
  }

  if (lane == 0) {
    if (is_row) {
      expected_rows[tile_id * kM + row_or_col] = expected;
      row_scales[tile_id * kM + row_or_col] = scale;
    } else {
      size_t output_index =
          static_cast<size_t>(tile_id) * kN + row_or_col;
      expected_cols[output_index] = expected;
      col_scales[output_index] = scale;
    }
  }
}

template <class Layout>
__device__ auto make_smem_tensor(Element* pointer, Layout layout) {
  return cute::make_tensor(cute::make_smem_ptr(pointer), layout);
}

template <class TmaA, class TmaB>
__global__ void __cluster_dims__(1, 1, 1)
wgmma_tma_expected_cache_abft_kernel(
    CUTLASS_GRID_CONSTANT TmaA const tma_a,
    CUTLASS_GRID_CONSTANT TmaB const tma_b,
    const float* expected_rows, const float* expected_cols,
    const float* row_scales, const float* col_scales, float* c,
    int* bad_tiles, int* corrected_tiles, int m, int n, int k,
    int tile_rows, int tile_cols,
    float abs_tolerance, float rel_tolerance, float fault_value,
    bool inject_fault) {
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  PipelineSharedStorage& storage =
      *reinterpret_cast<PipelineSharedStorage*>(shared_bytes);

  int local_thread =
      threadIdx.x < kComputeThreads
          ? threadIdx.x
          : threadIdx.x - kComputeThreads;
  bool is_compute_group = threadIdx.x < kComputeThreads;
  bool is_group_leader = local_thread == 0;
  int tile_count = tile_rows * tile_cols;
  int k_tile_count = k / kK;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int stage = 0; stage < kOutputStages; ++stage) {
      cutlass::arch::ClusterBarrier::init(
          &storage.output_full_barrier[stage], 1);
      cutlass::arch::ClusterBarrier::init(
          &storage.output_empty_barrier[stage], 1);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int stage = 0; stage < kOutputStages; ++stage) {
      cutlass::arch::ClusterBarrier::arrive(
          &storage.output_empty_barrier[stage]);
    }
  }
  __syncthreads();

  if (is_compute_group) {
    // Producer/compute group: TMA(k+1) overlaps WGMMA(k).
    bool is_tma_leader =
        local_thread < 32 && static_cast<bool>(cute::elect_one_sync());
    if (is_tma_leader) {
      cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
      cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
    }

    auto sA = cute::make_tensor(
        cute::make_smem_ptr(storage.a), SmemLayoutAStages{});
    auto sB = cute::make_tensor(
        cute::make_smem_ptr(storage.b), SmemLayoutBStages{});
    using CLayout = decltype(cute::make_layout(
        cute::make_shape(cute::Int<kM>{}, cute::Int<kN>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<kCStride>{})));
    auto mA = tma_a.get_tma_tensor(cute::make_shape(m, k, cute::_1{}));
    auto mB = tma_b.get_tma_tensor(cute::make_shape(n, k, cute::_1{}));
    auto gA = cute::local_tile(
        mA, TileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<cute::_1, cute::X, cute::_1>{});
    auto gB = cute::local_tile(
        mB, TileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<cute::X, cute::_1, cute::_1>{});
    auto block_tma_a = tma_a.get_slice(0);
    auto block_tma_b = tma_b.get_slice(0);
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(local_thread);

    for (int task = 0, tile_id = blockIdx.x;
         tile_id < tile_count;
         ++task, tile_id += gridDim.x) {
      int output_stage = task % kOutputStages;
      uint32_t output_phase = (task / kOutputStages) & 1;
      cutlass::arch::ClusterBarrier::wait(
          &storage.output_empty_barrier[output_stage], output_phase);
      cutlass::arch::NamedBarrier::sync(kComputeThreads, 0);

      int tile_row = tile_id / tile_cols;
      int tile_col = tile_id % tile_cols;
      int valid_cols = min(kN, n - tile_col * kN);

      if (is_tma_leader) {
#pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
          cutlass::arch::ClusterTransactionBarrier::init(
              &storage.tma_full_barrier[stage], 1);
        }
        cutlass::arch::fence_barrier_init();
      }
      cutlass::arch::NamedBarrier::sync(kComputeThreads, 0);

      auto gA_tile = gA(
          cute::_, cute::_, tile_row, cute::_, cute::Int<0>{});
      auto gB_tile = gB(
          cute::_, cute::_, tile_col, cute::_, cute::Int<0>{});
      auto tAgA = block_tma_a.partition_S(gA_tile);
      auto tAsA = block_tma_a.partition_D(sA);
      auto tBgB = block_tma_b.partition_S(gB_tile);
      auto tBsB = block_tma_b.partition_D(sB);

      if (is_tma_leader) {
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            &storage.tma_full_barrier[0], kTmaTransactionBytes);
        cute::copy(
            tma_a.with(storage.tma_full_barrier[0]),
            tAgA(cute::_, cute::_, cute::_, 0),
            tAsA(cute::_, cute::_, cute::_, 0));
        cute::copy(
            tma_b.with(storage.tma_full_barrier[0]),
            tBgB(cute::_, cute::_, cute::_, 0),
            tBsB(cute::_, cute::_, cute::_, 0));
      }

      auto accum = cute::partition_fragment_C(
          tiled_mma, cute::make_shape(cute::_64{}, cute::_64{}));
      cute::clear(accum);

      for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        int read_stage = k_tile % kStages;
        uint32_t read_phase = (k_tile / kStages) & 1;
        cutlass::arch::ClusterTransactionBarrier::wait(
            &storage.tma_full_barrier[read_stage], read_phase);

        int next_k_tile = k_tile + 1;
        if (next_k_tile < k_tile_count && is_tma_leader) {
          int write_stage = next_k_tile % kStages;
          cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
              &storage.tma_full_barrier[write_stage],
              kTmaTransactionBytes);
          cute::copy(
              tma_a.with(storage.tma_full_barrier[write_stage]),
              tAgA(cute::_, cute::_, cute::_, next_k_tile),
              tAsA(cute::_, cute::_, cute::_, write_stage));
          cute::copy(
              tma_b.with(storage.tma_full_barrier[write_stage]),
              tBgB(cute::_, cute::_, cute::_, next_k_tile),
              tBsB(cute::_, cute::_, cute::_, write_stage));
        }

        auto sA_stage = sA(cute::_, cute::_, read_stage);
        auto sB_stage = sB(cute::_, cute::_, read_stage);
        auto tCsA = thread_mma.partition_A(sA_stage);
        auto tCsB = thread_mma.partition_B(sB_stage);
        auto tCrA = thread_mma.make_fragment_A(tCsA);
        auto tCrB = thread_mma.make_fragment_B(tCsB);
        cute::copy(tCsA, tCrA);
        cute::copy(tCsB, tCrB);

        tiled_mma.accumulate_ =
            k_tile == 0 ? cute::GMMA::ScaleOut::Zero
                        : cute::GMMA::ScaleOut::One;
        cute::warpgroup_fence_operand(accum);
        cute::warpgroup_arrive();
        cute::gemm(tiled_mma, tCrA, tCrB, accum);
        cute::warpgroup_commit_batch();

        cute::warpgroup_wait<0>();
        cute::warpgroup_fence_operand(accum);
        cutlass::arch::NamedBarrier::sync(kComputeThreads, 0);
      }

      auto sC = cute::make_tensor(
          cute::make_smem_ptr(storage.c[output_stage]), CLayout{});
      auto tCsC = thread_mma.partition_C(sC);
      cute::copy(accum, tCsC);
      if (is_group_leader) {
        storage.tile_ids[output_stage] = tile_id;
        storage.valid_cols[output_stage] = valid_cols;
      }
      cutlass::arch::NamedBarrier::sync(kComputeThreads, 0);
      if (is_group_leader) {
        cutlass::arch::ClusterBarrier::arrive(
            &storage.output_full_barrier[output_stage]);
      }
    }
  } else {
    // Verify group: consume the previous output slot while compute advances.
    for (int task = 0, tile_id = blockIdx.x;
         tile_id < tile_count;
         ++task, tile_id += gridDim.x) {
      int output_stage = task % kOutputStages;
      uint32_t output_phase = (task / kOutputStages) & 1;
      cutlass::arch::ClusterBarrier::wait(
          &storage.output_full_barrier[output_stage], output_phase);

      tile_id = storage.tile_ids[output_stage];
      int tile_row = tile_id / tile_cols;
      int tile_col = tile_id % tile_cols;
      int valid_cols = storage.valid_cols[output_stage];

      if (is_group_leader) {
        storage.bad_row_count[output_stage] = 0;
        storage.bad_col_count[output_stage] = 0;
        storage.bad_row[output_stage] = -1;
        storage.bad_col[output_stage] = -1;
        if (inject_fault && tile_id == tile_count / 2) {
          int row = kM / 2;
          int col = min(kN / 2, n - tile_col * kN - 1);
          if (col >= 0) {
            storage.c[output_stage][col * kCStride + row] +=
                fault_value;
          }
        }
      }
      cutlass::arch::NamedBarrier::sync(kVerifyThreads, 1);

      for (int checksum_index = local_thread;
           checksum_index < kM;
           checksum_index += kVerifyThreads) {
        float row_actual = 0.0f;
#pragma unroll
        for (int col = 0; col < kN; ++col) {
          if (col < valid_cols) {
            row_actual +=
                storage.c[output_stage]
                         [col * kCStride + checksum_index];
          }
        }
        float row_delta =
            expected_rows[tile_id * kM + checksum_index] - row_actual;
        storage.deltas[output_stage][checksum_index] = row_delta;
        if (fabsf(row_delta) >
            abs_tolerance +
                rel_tolerance *
                    fmaxf(1.0f,
                          row_scales[tile_id * kM + checksum_index])) {
          atomicAdd(&storage.bad_row_count[output_stage], 1);
          atomicCAS(&storage.bad_row[output_stage], -1, checksum_index);
        }

        float col_delta = 0.0f;
        if (checksum_index < valid_cols) {
          float col_actual = 0.0f;
#pragma unroll
          for (int row = 0; row < kM; ++row) {
            col_actual +=
                storage.c[output_stage]
                         [checksum_index * kCStride + row];
          }
          col_delta =
              expected_cols[tile_id * kN + checksum_index] -
              col_actual;
          if (fabsf(col_delta) >
              abs_tolerance +
                  rel_tolerance *
                      fmaxf(1.0f,
                            col_scales[tile_id * kN + checksum_index])) {
            atomicAdd(&storage.bad_col_count[output_stage], 1);
            atomicCAS(
                &storage.bad_col[output_stage], -1, checksum_index);
          }
        }
        storage.deltas[output_stage][kM + checksum_index] =
            col_delta;
      }
      cutlass::arch::NamedBarrier::sync(kVerifyThreads, 1);

      if (is_group_leader) {
        if (storage.bad_row_count[output_stage] > 0 ||
            storage.bad_col_count[output_stage] > 0) {
          atomicAdd(bad_tiles, 1);
        }
        if (storage.bad_row_count[output_stage] == 1 &&
            storage.bad_col_count[output_stage] == 1) {
          int row = storage.bad_row[output_stage];
          int col = storage.bad_col[output_stage];
          storage.c[output_stage][col * kCStride + row] +=
              0.5f *
              (storage.deltas[output_stage][row] +
               storage.deltas[output_stage][kM + col]);
          atomicAdd(corrected_tiles, 1);
        }
      }
      cutlass::arch::NamedBarrier::sync(kVerifyThreads, 1);

      for (int index = local_thread; index < kM * kN;
           index += kVerifyThreads) {
        int row = index % kM;
        int col = index / kM;
        int global_col = tile_col * kN + col;
        if (global_col < n) {
          c[global_col * m + tile_row * kM + row] =
              storage.c[output_stage][col * kCStride + row];
        }
      }
      cutlass::arch::NamedBarrier::sync(kVerifyThreads, 1);
      if (is_group_leader) {
        cutlass::arch::ClusterBarrier::arrive(
            &storage.output_empty_barrier[output_stage]);
      }
    }
  }
}

struct VerificationResult {
  float max_absolute_error = 0.0f;
  float max_tolerance_ratio = 0.0f;
  int checked = 0;
};

static VerificationResult verify_samples(
    const std::vector<Element>& a, const std::vector<Element>& b,
    const std::vector<float>& c, const Options& options) {
  VerificationResult result;
  if (options.verify_samples == 0) {
    return result;
  }

  int row_step = std::max(1, options.m / options.verify_samples);
  int col_step = std::max(1, options.n / options.verify_samples);
  int checked_rows = 0;
  for (int row = 0; row < options.m && checked_rows < options.verify_samples;
       row += row_step, ++checked_rows) {
    int checked_cols = 0;
    for (int col = 0;
         col < options.n && checked_cols < options.verify_samples;
         col += col_step, ++checked_cols) {
      double reference = 0.0;
      for (int kk = 0; kk < options.k; ++kk) {
        reference += static_cast<double>(
                         static_cast<float>(a[row * options.k + kk])) *
                     static_cast<double>(
                         static_cast<float>(b[col * options.k + kk]));
      }
      float absolute_error = static_cast<float>(
          std::fabs(c[col * options.m + row] - reference));
      float tolerance =
          options.verify_abs_tolerance +
          options.verify_rel_tolerance *
              std::max(1.0f, static_cast<float>(std::fabs(reference)));
      result.max_absolute_error =
          std::max(result.max_absolute_error, absolute_error);
      result.max_tolerance_ratio =
          std::max(result.max_tolerance_ratio, absolute_error / tolerance);
      ++result.checked;
    }
  }
  return result;
}

int main(int argc, char** argv) {
  Options options = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(0));

  size_t a_count = static_cast<size_t>(options.m) * options.k;
  size_t b_count = static_cast<size_t>(options.n) * options.k;
  size_t c_count = static_cast<size_t>(options.m) * options.n;
  int tile_rows = options.m / kM;
  int tile_cols = div_up(options.n, kN);
  int tile_count = tile_rows * tile_cols;

  Element* d_a = nullptr;
  Element* d_b = nullptr;
  float* d_c = nullptr;
  float* d_a_sums = nullptr;
  float* d_a_magnitudes = nullptr;
  float* d_b_sums = nullptr;
  float* d_b_magnitudes = nullptr;
  float* d_expected_rows = nullptr;
  float* d_expected_cols = nullptr;
  float* d_row_scales = nullptr;
  float* d_col_scales = nullptr;
  int* d_bad_tiles = nullptr;
  int* d_corrected_tiles = nullptr;
  size_t a_metadata_count = static_cast<size_t>(tile_rows) * options.k;
  size_t b_metadata_count = static_cast<size_t>(tile_cols) * options.k;
  size_t row_metadata_count = static_cast<size_t>(tile_count) * kM;
  size_t col_metadata_count = static_cast<size_t>(tile_count) * kN;
  CUDA_CHECK(cudaMalloc(&d_a, a_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_b, b_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_a_sums, a_metadata_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_a_magnitudes,
                        a_metadata_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b_sums, b_metadata_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b_magnitudes,
                        b_metadata_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &d_expected_rows, row_metadata_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &d_expected_cols, col_metadata_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_row_scales, row_metadata_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_col_scales, col_metadata_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bad_tiles, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_corrected_tiles, sizeof(int)));

  constexpr int generator_threads = 256;
  generate_fp8_kernel<<<
      static_cast<int>((a_count + generator_threads - 1) / generator_threads),
      generator_threads>>>(d_a, a_count, 0x12345678U);
  generate_fp8_kernel<<<
      static_cast<int>((b_count + generator_threads - 1) / generator_threads),
      generator_threads>>>(d_b, b_count, 0x9abcdef0U);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  auto tensor_a = cute::make_tensor(
      d_a,
      cute::make_layout(
          cute::make_shape(options.m, options.k, cute::_1{}),
          cute::make_stride(options.k, cute::_1{}, cute::_0{})));
  auto tensor_b = cute::make_tensor(
      d_b,
      cute::make_layout(
          cute::make_shape(options.n, options.k, cute::_1{}),
          cute::make_stride(options.k, cute::_1{}, cute::_0{})));
  auto tma_a = cute::make_tma_copy_A_sm90(
      cute::SM90_TMA_LOAD{}, tensor_a,
      SmemLayoutAStages{}(cute::_, cute::_, cute::Int<0>{}),
      TileShape{}, ClusterShape{});
  auto tma_b = cute::make_tma_copy_B_sm90(
      cute::SM90_TMA_LOAD{}, tensor_b,
      SmemLayoutBStages{}(cute::_, cute::_, cute::Int<0>{}),
      TileShape{}, ClusterShape{});

  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  auto kernel =
      wgmma_tma_expected_cache_abft_kernel<
          decltype(tma_a), decltype(tma_b)>;
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(PipelineSharedStorage)));

  int multiprocessor_count = 0;
  int active_blocks_per_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &multiprocessor_count, cudaDevAttrMultiProcessorCount, 0));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, kernel, kPipelineThreads,
      sizeof(PipelineSharedStorage)));
  int max_persistent_ctas = std::min(
      tile_count, multiprocessor_count * active_blocks_per_sm);
  int persistent_ctas = max_persistent_ctas;
  int tiles_per_persistent_cta =
      div_up(tile_count, persistent_ctas);

  CUDA_CHECK(cudaEventRecord(start, stream));
  cache_input_tile_checksums_kernel<<<tile_rows + tile_cols, 256, 0, stream>>>(
      d_a, d_b, d_a_sums, d_a_magnitudes, d_b_sums, d_b_magnitudes,
      options.n, options.k, tile_rows, tile_cols);
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float input_metadata_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&input_metadata_ms, start, stop));

  constexpr int metadata_threads = 256;
  constexpr int metadata_warps = metadata_threads / 32;
  int expected_metadata_blocks =
      div_up(tile_count * (kM + kN), metadata_warps);
  CUDA_CHECK(cudaEventRecord(start, stream));
  cache_expected_tile_checksums_kernel<<<expected_metadata_blocks,
                                         metadata_threads, 0, stream>>>(
      d_a, d_b, d_a_sums, d_a_magnitudes, d_b_sums, d_b_magnitudes,
      d_expected_rows, d_expected_cols, d_row_scales, d_col_scales,
      options.n, options.k, tile_rows, tile_cols);
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float expected_metadata_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&expected_metadata_ms, start, stop));
  float metadata_prepare_ms = input_metadata_ms + expected_metadata_ms;

  auto enqueue_gemm = [&]() {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = 1;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(persistent_ctas, 1, 1);
    config.blockDim = dim3(kPipelineThreads, 1, 1);
    config.dynamicSmemBytes = sizeof(PipelineSharedStorage);
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(
        &config, kernel,
        tma_a, tma_b, d_expected_rows, d_expected_cols, d_row_scales,
        d_col_scales, d_c, d_bad_tiles, d_corrected_tiles, options.m,
        options.n, options.k, tile_rows, tile_cols,
        options.abft_abs_tolerance,
        options.abft_rel_tolerance, options.fault_value,
        options.inject_fault));
  };

  CUDA_CHECK(cudaMemsetAsync(d_bad_tiles, 0, sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_corrected_tiles, 0, sizeof(int), stream));
  CUDA_CHECK(cudaEventRecord(start, stream));
  enqueue_gemm();
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float first_gemm_abft_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&first_gemm_abft_ms, start, stop));

  for (int i = 0; i < options.warmup; ++i) {
    CUDA_CHECK(cudaMemsetAsync(d_bad_tiles, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_corrected_tiles, 0, sizeof(int), stream));
    enqueue_gemm();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float total_ms = 0.0f;
  for (int i = 0; i < options.repeat; ++i) {
    CUDA_CHECK(cudaMemsetAsync(d_bad_tiles, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_corrected_tiles, 0, sizeof(int), stream));
    CUDA_CHECK(cudaEventRecord(start, stream));
    enqueue_gemm();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float iteration_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&iteration_ms, start, stop));
    total_ms += iteration_ms;
  }
  CUDA_CHECK(cudaGetLastError());

  std::vector<Element> h_a(a_count);
  std::vector<Element> h_b(b_count);
  std::vector<float> h_c(c_count);
  CUDA_CHECK(cudaMemcpy(h_a.data(), d_a, a_count * sizeof(Element),
                       cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, b_count * sizeof(Element),
                       cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, c_count * sizeof(float),
                       cudaMemcpyDeviceToHost));
  int bad_tiles = 0;
  int corrected_tiles = 0;
  CUDA_CHECK(cudaMemcpy(&bad_tiles, d_bad_tiles, sizeof(int),
                       cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&corrected_tiles, d_corrected_tiles, sizeof(int),
                       cudaMemcpyDeviceToHost));

  float average_ms = total_ms / options.repeat;
  float first_call_total_ms = metadata_prepare_ms + first_gemm_abft_ms;
  float amortized_ms =
      average_ms + metadata_prepare_ms / options.repeat;
  double operations =
      2.0 * static_cast<double>(options.m) * options.n * options.k;
  double tflops = operations / (average_ms * 1.0e9);
  double first_call_tflops =
      operations / (first_call_total_ms * 1.0e9);
  double amortized_tflops =
      operations / (amortized_ms * 1.0e9);
  VerificationResult verification =
      verify_samples(h_a, h_b, h_c, options);
  bool numerical_pass =
      options.verify_samples == 0 || verification.max_tolerance_ratio <= 1.0f;
  bool abft_pass = options.inject_fault
                       ? bad_tiles == 1 && corrected_tiles == 1
                       : bad_tiles == 0;
  bool passed = numerical_pass && abft_pass;

  size_t metadata_bytes =
      (2 * (a_metadata_count + b_metadata_count) +
       2 * (row_metadata_count + col_metadata_count)) *
      sizeof(float);
  std::printf(
      "version: S5 full expected-cache persistent TMA-WGMMA ABFT\n");
  std::printf(
      "instruction_tile: 64x64x32, input_stages: %d, output_stages: %d\n",
      kStages, kOutputStages);
  std::printf(
      "threads_per_cta: %d, compute_threads: %d, verify_threads: %d\n",
      kPipelineThreads, kComputeThreads, kVerifyThreads);
  std::printf(
      "pipeline: cached expected -> TMA+WGMMA -> "
      "actual checksum+ABFT+store\n");
  std::printf("shared_storage_bytes: %zu\n", sizeof(PipelineSharedStorage));
  std::printf("layout: A=[M,K] row-major, B=[N,K] column-major, "
              "C=[N,M] column-major\n");
  std::printf("shape: M=%d N=%d K=%d, output_tiles=%d\n", options.m,
              options.n, options.k, tile_count);
  std::printf(
      "shared_c_stride: %d, persistent_ctas: %d, max_persistent_ctas: %d\n",
      kCStride, persistent_ctas, max_persistent_ctas);
  std::printf(
      "tiles_per_persistent_cta: %d, active_blocks_per_sm: %d\n",
      tiles_per_persistent_cta, active_blocks_per_sm);
  std::printf("metadata_cache_bytes: %zu\n", metadata_bytes);
  std::printf("input_metadata_time_ms: %.6f\n", input_metadata_ms);
  std::printf("expected_metadata_time_ms: %.6f\n", expected_metadata_ms);
  std::printf("metadata_prepare_time_ms: %.6f\n", metadata_prepare_ms);
  std::printf("first_fused_gemm_abft_time_ms: %.6f\n",
              first_gemm_abft_ms);
  std::printf("first_call_total_ms: %.6f\n", first_call_total_ms);
  std::printf("first_call_tflops: %.6f\n", first_call_tflops);
  std::printf("avg_fused_gemm_abft_time_ms: %.6f\n", average_ms);
  std::printf("amortized_time_ms: %.6f\n", amortized_ms);
  std::printf("amortized_tflops: %.6f\n", amortized_tflops);
  std::printf("fused_gemm_tflops: %.6f\n", tflops);
  std::printf("abft_tolerance: abs=%.6e rel=%.6e\n",
              options.abft_abs_tolerance, options.abft_rel_tolerance);
  std::printf("bad_tiles: %d\n", bad_tiles);
  std::printf("corrected_tiles: %d\n", corrected_tiles);
  std::printf("verification_samples: %d\n", verification.checked);
  std::printf("sampled_max_absolute_error: %.6e\n",
              verification.max_absolute_error);
  std::printf("sampled_max_tolerance_ratio: %.6e\n",
              verification.max_tolerance_ratio);
  std::printf("verification: %s\n", passed ? "PASS" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_a_sums));
  CUDA_CHECK(cudaFree(d_a_magnitudes));
  CUDA_CHECK(cudaFree(d_b_sums));
  CUDA_CHECK(cudaFree(d_b_magnitudes));
  CUDA_CHECK(cudaFree(d_expected_rows));
  CUDA_CHECK(cudaFree(d_expected_cols));
  CUDA_CHECK(cudaFree(d_row_scales));
  CUDA_CHECK(cudaFree(d_col_scales));
  CUDA_CHECK(cudaFree(d_bad_tiles));
  CUDA_CHECK(cudaFree(d_corrected_tiles));
  return passed ? 0 : 1;
}
