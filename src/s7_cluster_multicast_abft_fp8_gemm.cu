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
using TileN = cute::_128;
using TileK = cute::_32;
using StageK = cute::_128;
using MmaTileShape = cute::Shape<TileM, TileN, TileK>;
using TmaTileShape = cute::Shape<TileM, TileN, StageK>;
using GmmaOp = decltype(cute::GMMA::ss_op_selector<
    Element, Element, Accumulator, MmaTileShape, cute::GMMA::Major::K,
    cute::GMMA::Major::K>());
using TiledMma = decltype(cute::make_tiled_mma(GmmaOp{}));
using SmemLayoutA = decltype(cute::tile_to_shape(
    cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
    cute::Shape<TileM, StageK>{}));
using SmemLayoutB = decltype(cute::tile_to_shape(
    cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
    cute::Shape<TileN, StageK>{}));
constexpr int kStages = 3;
using SmemLayoutAStages = decltype(cute::tile_to_shape(
    cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
    cute::Shape<TileM, StageK, cute::Int<kStages>>{},
    cute::Step<cute::_1, cute::_2, cute::_3>{}));
using SmemLayoutBStages = decltype(cute::tile_to_shape(
    cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
    cute::Shape<TileN, StageK, cute::Int<kStages>>{},
    cute::Step<cute::_1, cute::_2, cute::_3>{}));
using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

constexpr int kM = 64;
constexpr int kN = 64;
constexpr int kK = 32;
constexpr int kStageK = 128;
constexpr int kMmaPerStage = kStageK / kK;
constexpr int kConsumerWarpgroups = 2;
constexpr int kConsumerThreads = kConsumerWarpgroups * 128;
constexpr int kProducerThreads = 32;
constexpr int kThreads = kConsumerThreads + kProducerThreads;
constexpr int kCtaM = kConsumerWarpgroups * kM;
constexpr int kCtaN = 128;
constexpr int kCStride = kM + 1;
constexpr int kAbftSubtiles = (kCtaM / kM) * (kCtaN / kN);
constexpr uint32_t kTmaTransactionBytes =
    (kCtaM * kStageK + kCtaN * kStageK) * sizeof(Element);

struct InputSharedStorage {
  alignas(128) Element
      a[kConsumerWarpgroups][cute::cosize_v<SmemLayoutAStages>];
  alignas(128) Element b[cute::cosize_v<SmemLayoutBStages>];
};

union MainloopEpilogueStorage {
  InputSharedStorage input;
  alignas(128) float c[kConsumerWarpgroups][kCStride * kCtaN];
};

struct PipelineSharedStorage {
  alignas(8) uint64_t tma_full_barrier[kStages];
  alignas(8) uint64_t tma_empty_barrier[kStages];
  MainloopEpilogueStorage buffers;
  float deltas[kAbftSubtiles][kM + kN];
  int bad_row_count[kAbftSubtiles];
  int bad_col_count[kAbftSubtiles];
  int bad_row[kAbftSubtiles];
  int bad_col[kAbftSubtiles];
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
          "Constraints: M %% 128 == 0, N %% 256 == 0, "
          "and K %% 128 == 0.\n",
          argv[0]);
      std::exit(EXIT_SUCCESS);
    }
  }

  if (options.m <= 0 || options.n <= 0 || options.k <= 0 ||
      options.m % kCtaM != 0 || options.n % (2 * kCtaN) != 0 ||
      options.k % kStageK != 0 ||
      options.warmup < 0 || options.repeat <= 0 ||
      options.verify_samples < 0 || options.verify_abs_tolerance < 0.0f ||
      options.verify_rel_tolerance < 0.0f ||
      options.abft_abs_tolerance < 0.0f ||
      options.abft_rel_tolerance < 0.0f || options.fault_value <= 0.0f) {
    std::fprintf(
        stderr,
        "Invalid options. Require M%%128==0, N%%256==0, "
        "K%%128==0, and positive shape.\n");
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
__global__ void __cluster_dims__(2, 1, 1)
wgmma_tma_k128_cluster_abft_kernel(
    CUTLASS_GRID_CONSTANT TmaA const tma_a,
    CUTLASS_GRID_CONSTANT TmaB const tma_b,
    const float* expected_rows, const float* expected_cols,
    const float* row_scales, const float* col_scales, float* c,
    int* bad_tiles, int* corrected_tiles, int m, int n, int k,
    int micro_tile_rows, int micro_tile_cols,
    int cta_rows, int cta_cols,
    float abs_tolerance, float rel_tolerance, float fault_value,
    bool inject_fault) {
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  PipelineSharedStorage& storage =
      *reinterpret_cast<PipelineSharedStorage*>(shared_bytes);

  int cluster_rank = cute::block_rank_in_cluster();
  int cluster_cols = cta_cols / 2;
  int cluster_id = blockIdx.x / 2;
  int cta_row = cluster_id / cluster_cols;
  int cta_col = (cluster_id % cluster_cols) * 2 + cluster_rank;
  int k_stage_count = k / kStageK;
  bool is_consumer = threadIdx.x < kConsumerThreads;
  bool is_producer = !is_consumer;
  bool is_producer_leader = threadIdx.x == kConsumerThreads;

  if (is_producer_leader) {
#pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
      cutlass::arch::ClusterTransactionBarrier::init(
          &storage.tma_full_barrier[stage], 1);
      cutlass::arch::ClusterBarrier::init(
          &storage.tma_empty_barrier[stage],
          kConsumerWarpgroups * 2);
    }
    cutlass::arch::fence_barrier_init();
  }
  cute::cluster_arrive_relaxed();
  cute::cluster_wait();
  int producer_lane = threadIdx.x - kConsumerThreads;
  if (is_producer &&
      producer_lane < kConsumerWarpgroups * 2) {
#pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
      cutlass::arch::ClusterBarrier::arrive(
          &storage.tma_empty_barrier[stage]);
    }
  }
  if (is_producer_leader) {
    cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
  }
  __syncthreads();

  if (is_producer) {
    auto sA0 = cute::make_tensor(
        cute::make_smem_ptr(storage.buffers.input.a[0]), SmemLayoutAStages{});
    auto sA1 = cute::make_tensor(
        cute::make_smem_ptr(storage.buffers.input.a[1]), SmemLayoutAStages{});
    auto sB = cute::make_tensor(
        cute::make_smem_ptr(storage.buffers.input.b), SmemLayoutBStages{});
    auto mA = tma_a.get_tma_tensor(cute::make_shape(m, k, cute::_1{}));
    auto mB = tma_b.get_tma_tensor(cute::make_shape(n, k, cute::_1{}));
    auto gA = cute::local_tile(
        mA, TmaTileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<cute::_1, cute::X, cute::_1>{});
    auto gB = cute::local_tile(
        mB, TmaTileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<cute::X, cute::_1, cute::_1>{});
    auto gA0 = gA(
        cute::_, cute::_, cta_row * kConsumerWarpgroups,
        cute::_, cute::Int<0>{});
    auto gA1 = gA(
        cute::_, cute::_, cta_row * kConsumerWarpgroups + 1,
        cute::_, cute::Int<0>{});
    auto gB_tile = gB(
        cute::_, cute::_, cta_col, cute::_, cute::Int<0>{});
    auto block_tma_a = tma_a.get_slice(cluster_rank);
    auto block_tma_b = tma_b.get_slice(0);
    auto tAgA0 = block_tma_a.partition_S(gA0);
    auto tAgA1 = block_tma_a.partition_S(gA1);
    auto tAsA0 = block_tma_a.partition_D(sA0);
    auto tAsA1 = block_tma_a.partition_D(sA1);
    auto tBgB = block_tma_b.partition_S(gB_tile);
    auto tBsB = block_tma_b.partition_D(sB);

    if (is_producer_leader) {
      constexpr uint16_t a_multicast_mask = 0x3;
      for (int k_stage = 0; k_stage < k_stage_count; ++k_stage) {
        int stage = k_stage % kStages;
        uint32_t phase = (k_stage / kStages) & 1;
        cutlass::arch::ClusterBarrier::wait(
            &storage.tma_empty_barrier[stage], phase);
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            &storage.tma_full_barrier[stage], kTmaTransactionBytes);
        cute::copy(
            tma_a.with(
                storage.tma_full_barrier[stage],
                a_multicast_mask),
            tAgA0(cute::_, cute::_, cute::_, k_stage),
            tAsA0(cute::_, cute::_, cute::_, stage));
        cute::copy(
            tma_a.with(
                storage.tma_full_barrier[stage],
                a_multicast_mask),
            tAgA1(cute::_, cute::_, cute::_, k_stage),
            tAsA1(cute::_, cute::_, cute::_, stage));
        cute::copy(
            tma_b.with(storage.tma_full_barrier[stage]),
            tBgB(cute::_, cute::_, cute::_, k_stage),
            tBsB(cute::_, cute::_, cute::_, stage));
      }
    }
    cutlass::arch::NamedBarrier::sync(kThreads, 2);
  } else {
    int consumer_group = threadIdx.x / 128;
    int consumer_thread = threadIdx.x % 128;
    auto sA = cute::make_tensor(
        cute::make_smem_ptr(storage.buffers.input.a[consumer_group]),
        SmemLayoutAStages{});
    auto sB = cute::make_tensor(
        cute::make_smem_ptr(storage.buffers.input.b), SmemLayoutBStages{});
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(consumer_thread);
    auto accum = cute::partition_fragment_C(
        tiled_mma,
        cute::make_shape(cute::Int<kM>{}, cute::Int<kCtaN>{}));
    cute::clear(accum);

    for (int k_stage = 0; k_stage < k_stage_count; ++k_stage) {
      int stage = k_stage % kStages;
      uint32_t phase = (k_stage / kStages) & 1;
      cutlass::arch::ClusterTransactionBarrier::wait(
          &storage.tma_full_barrier[stage], phase);

      auto sA_stage = sA(cute::_, cute::_, stage);
      auto sB_stage = sB(cute::_, cute::_, stage);
      cute::warpgroup_fence_operand(accum);
      cute::warpgroup_arrive();
#pragma unroll
      for (int k_subtile = 0; k_subtile < kMmaPerStage; ++k_subtile) {
        auto sA_subtile = cute::local_tile(
            sA_stage, cute::Shape<TileM, TileK>{},
            cute::make_coord(0, k_subtile));
        auto sB_subtile = cute::local_tile(
            sB_stage, cute::Shape<TileN, TileK>{},
            cute::make_coord(0, k_subtile));
        auto tCsA = thread_mma.partition_A(sA_subtile);
        auto tCsB = thread_mma.partition_B(sB_subtile);
        auto tCrA = thread_mma.make_fragment_A(tCsA);
        auto tCrB = thread_mma.make_fragment_B(tCsB);
        cute::copy(tCsA, tCrA);
        cute::copy(tCsB, tCrB);
        tiled_mma.accumulate_ =
            (k_stage == 0 && k_subtile == 0)
                ? cute::GMMA::ScaleOut::Zero
                : cute::GMMA::ScaleOut::One;
        cute::gemm(tiled_mma, tCrA, tCrB, accum);
      }
      cute::warpgroup_commit_batch();
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(accum);

      cutlass::arch::NamedBarrier::sync(128, consumer_group);
      if (consumer_thread == 0) {
        cutlass::arch::ClusterBarrier::arrive(
            &storage.tma_empty_barrier[stage], 0, 1);
        cutlass::arch::ClusterBarrier::arrive(
            &storage.tma_empty_barrier[stage], 1, 1);
      }
    }
    cutlass::arch::NamedBarrier::sync(kThreads, 2);
    using CLayout = decltype(cute::make_layout(
        cute::make_shape(cute::Int<kM>{}, cute::Int<kCtaN>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<kCStride>{})));
    auto sC = cute::make_tensor(
        cute::make_smem_ptr(storage.buffers.c[consumer_group]), CLayout{});
    auto tCsC = thread_mma.partition_C(sC);
    cute::copy(accum, tCsC);
    cutlass::arch::NamedBarrier::sync(128, consumer_group);
  }
  __syncthreads();

  if (threadIdx.x < kAbftSubtiles * 32) {
    int subtile = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int micro_tile_count = micro_tile_rows * micro_tile_cols;
    int col_groups = kCtaN / kN;
    int row_group = subtile / col_groups;
    int col_group = subtile % col_groups;
    int micro_row = cta_row * kConsumerWarpgroups + row_group;
    int micro_col = cta_col * col_groups + col_group;
    int valid_cols = min(kN, n - micro_col * kN);
    if (valid_cols > 0) {
      int micro_tile_id = micro_row * micro_tile_cols + micro_col;

      if (lane == 0) {
        storage.bad_row_count[subtile] = 0;
        storage.bad_col_count[subtile] = 0;
        storage.bad_row[subtile] = -1;
        storage.bad_col[subtile] = -1;
        if (inject_fault && micro_tile_id == micro_tile_count / 2) {
          int row = kM / 2;
          int col = min(kN / 2, valid_cols - 1);
          storage.buffers.c[row_group]
                   [(col_group * kN + col) * kCStride + row] +=
              fault_value;
        }
      }
      __syncwarp();

      for (int checksum_index = lane;
           checksum_index < kM;
           checksum_index += 32) {
        float row_actual = 0.0f;
#pragma unroll
        for (int col = 0; col < kN; ++col) {
          if (col < valid_cols) {
            row_actual +=
                storage.buffers.c[row_group]
                         [(col_group * kN + col) * kCStride +
                          checksum_index];
          }
        }
        float row_delta =
            expected_rows[micro_tile_id * kM + checksum_index] -
            row_actual;
        storage.deltas[subtile][checksum_index] = row_delta;
        if (fabsf(row_delta) >
            abs_tolerance +
                rel_tolerance *
                    fmaxf(
                        1.0f,
                        row_scales[micro_tile_id * kM + checksum_index])) {
          atomicAdd(&storage.bad_row_count[subtile], 1);
          atomicCAS(
              &storage.bad_row[subtile], -1, checksum_index);
        }

        float col_delta = 0.0f;
        if (checksum_index < valid_cols) {
          float col_actual = 0.0f;
#pragma unroll
          for (int row = 0; row < kM; ++row) {
            col_actual +=
                storage.buffers.c[row_group]
                         [(col_group * kN + checksum_index) *
                              kCStride +
                          row];
          }
          col_delta =
              expected_cols[micro_tile_id * kN + checksum_index] -
              col_actual;
          if (fabsf(col_delta) >
              abs_tolerance +
                  rel_tolerance *
                      fmaxf(
                          1.0f,
                          col_scales[
                              micro_tile_id * kN + checksum_index])) {
            atomicAdd(&storage.bad_col_count[subtile], 1);
            atomicCAS(
                &storage.bad_col[subtile], -1, checksum_index);
          }
        }
        storage.deltas[subtile][kM + checksum_index] = col_delta;
      }
      __syncwarp();

      if (lane == 0) {
        if (storage.bad_row_count[subtile] > 0 ||
            storage.bad_col_count[subtile] > 0) {
          atomicAdd(bad_tiles, 1);
        }
        if (storage.bad_row_count[subtile] == 1 &&
            storage.bad_col_count[subtile] == 1) {
          int row = storage.bad_row[subtile];
          int col = storage.bad_col[subtile];
          storage.buffers.c[row_group]
                   [(col_group * kN + col) * kCStride + row] +=
              0.5f *
              (storage.deltas[subtile][row] +
               storage.deltas[subtile][kM + col]);
          atomicAdd(corrected_tiles, 1);
        }
      }
      __syncwarp();
    }
  }
  __syncthreads();

  for (int index = threadIdx.x;
       index < kCtaM * kCtaN;
       index += kThreads) {
    int row = index % kCtaM;
    int col = index / kCtaM;
    int global_row = cta_row * kCtaM + row;
    int global_col = cta_col * kCtaN + col;
    if (global_row < m && global_col < n) {
      int row_group = row / kM;
      int local_row = row % kM;
      c[global_col * m + global_row] =
          storage.buffers.c[row_group][col * kCStride + local_row];
    }
  }
  cute::cluster_arrive();
  cute::cluster_wait();
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
  int cta_rows = options.m / kCtaM;
  int cta_cols = div_up(options.n, kCtaN);
  int cta_count = cta_rows * cta_cols;
  int cluster_count = cta_rows * (cta_cols / 2);

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
      cute::SM90_TMA_LOAD_MULTICAST{}, tensor_a,
      SmemLayoutAStages{}(cute::_, cute::_, cute::Int<0>{}),
      TmaTileShape{}, ClusterShape{});
  auto tma_b = cute::make_tma_copy_B_sm90(
      cute::SM90_TMA_LOAD{}, tensor_b,
      SmemLayoutBStages{}(cute::_, cute::_, cute::Int<0>{}),
      TmaTileShape{}, ClusterShape{});

  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  auto kernel =
      wgmma_tma_k128_cluster_abft_kernel<
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
      &active_blocks_per_sm, kernel, kThreads,
      sizeof(PipelineSharedStorage)));

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
    attribute.val.clusterDim.x = 2;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(cluster_count * 2, 1, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = sizeof(PipelineSharedStorage);
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(
        &config, kernel,
        tma_a, tma_b, d_expected_rows, d_expected_cols, d_row_scales,
        d_col_scales, d_c, d_bad_tiles, d_corrected_tiles, options.m,
        options.n, options.k, tile_rows, tile_cols, cta_rows, cta_cols,
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
      "version: S7 K128 2x1 cluster-multicast TMA-WGMMA ABFT\n");
  std::printf(
      "instruction_tile: 64x128x32, cta_tile: %dx%dx%d, "
      "tma_stage_k: %d, stages: %d\n",
      kCtaM, kCtaN, kK, kStageK, kStages);
  std::printf(
      "threads_per_cta: %d, consumer_warpgroups: %d, "
      "producer_threads: %d, verifier_warps: %d\n",
      kThreads, kConsumerWarpgroups, kProducerThreads, kAbftSubtiles);
  std::printf(
      "pipeline: dedicated TMA producer || 2 WGMMA consumers; "
      "then %d ABFT subtiles\n",
      kAbftSubtiles);
  std::printf("shared_storage_bytes: %zu\n", sizeof(PipelineSharedStorage));
  std::printf("layout: A=[M,K] row-major, B=[N,K] column-major, "
              "C=[N,M] column-major\n");
  std::printf(
      "shape: M=%d N=%d K=%d, abft_tiles=%d, cta_tiles=%d\n",
      options.m, options.n, options.k, tile_count, cta_count);
  std::printf("cluster_shape: 2x1x1, clusters: %d\n", cluster_count);
  std::printf(
      "shared_c_stride: %d, active_blocks_per_sm: %d\n",
      kCStride, active_blocks_per_sm);
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
