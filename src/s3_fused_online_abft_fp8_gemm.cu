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
using SmemLayoutAStages = decltype(cute::tile_to_shape(
    cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
    cute::Shape<TileM, TileK, cute::Int<kStages>>{},
    cute::Step<cute::_1, cute::_2, cute::_3>{}));
using SmemLayoutBStages = decltype(cute::tile_to_shape(
    cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
    cute::Shape<TileN, TileK, cute::Int<kStages>>{},
    cute::Step<cute::_1, cute::_2, cute::_3>{}));
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

constexpr int kThreads = 128;
constexpr int kM = 64;
constexpr int kN = 64;
constexpr int kK = 32;
constexpr uint32_t kTmaTransactionBytes =
    (kM * kK + kN * kK) * sizeof(Element);

struct SharedStorage {
  alignas(8) uint64_t full_barrier[kStages];
  alignas(128) Element a[cute::cosize_v<SmemLayoutAStages>];
  alignas(128) Element b[cute::cosize_v<SmemLayoutBStages>];
  alignas(16) float c[kM * kN];
  float stage_a_sums[kK];
  float stage_a_magnitudes[kK];
  float stage_b_sums[kK];
  float stage_b_magnitudes[kK];
  float expected_rows[kM];
  float expected_cols[kN];
  float row_scales[kM];
  float col_scales[kN];
  float row_deltas[kM];
  float col_deltas[kN];
  int bad_row_count;
  int bad_col_count;
  int bad_row;
  int bad_col;
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

template <class Layout>
__device__ auto make_smem_tensor(Element* pointer, Layout layout) {
  return cute::make_tensor(cute::make_smem_ptr(pointer), layout);
}

// One 128-thread warpgroup is both the TMA producer and WGMMA consumer.
// Stage i+1 is loaded by TMA while WGMMA consumes stage i.
template <class TmaA, class TmaB>
__global__ void __cluster_dims__(1, 1, 1) wgmma_tma_fused_abft_kernel(
    CUTLASS_GRID_CONSTANT TmaA const tma_a,
    CUTLASS_GRID_CONSTANT TmaB const tma_b,
    float* c, int* bad_tiles, int* corrected_tiles, int m, int n, int k, int tile_rows, int tile_cols,
    float abs_tolerance, float rel_tolerance, float fault_value,
    bool inject_fault) {
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  SharedStorage& storage =
      *reinterpret_cast<SharedStorage*>(shared_bytes);

  int tile_id = blockIdx.x;
  if (tile_id >= tile_rows * tile_cols) {
    return;
  }
  int tile_row = tile_id / tile_cols;
  int tile_col = tile_id % tile_cols;
  int k_tile_count = k / kK;
  int valid_cols = min(kN, n - tile_col * kN);
  bool is_tma_leader =
      (threadIdx.x < 32) && static_cast<bool>(cute::elect_one_sync());

  auto sA = cute::make_tensor(
      cute::make_smem_ptr(storage.a), SmemLayoutAStages{});
  auto sB = cute::make_tensor(
      cute::make_smem_ptr(storage.b), SmemLayoutBStages{});
  using CLayout = decltype(cute::make_layout(
      cute::make_shape(cute::Int<kM>{}, cute::Int<kN>{}),
      cute::make_stride(cute::Int<1>{}, cute::Int<kM>{})));
  auto sC =
      cute::make_tensor(cute::make_smem_ptr(storage.c), CLayout{});
  if (threadIdx.x < kM) {
    storage.expected_rows[threadIdx.x] = 0.0f;
    storage.row_scales[threadIdx.x] = 0.0f;
    storage.expected_cols[threadIdx.x] = 0.0f;
    storage.col_scales[threadIdx.x] = 0.0f;
  }

  if (is_tma_leader) {
    for (int stage = 0; stage < kStages; ++stage) {
      cutlass::arch::ClusterTransactionBarrier::init(
          &storage.full_barrier[stage], 1);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  if (is_tma_leader) {
    cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
  }

  auto mA = tma_a.get_tma_tensor(cute::make_shape(m, k, cute::_1{}));
  auto mB = tma_b.get_tma_tensor(cute::make_shape(n, k, cute::_1{}));
  auto gA = cute::local_tile(
      mA, TileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
      cute::Step<cute::_1, cute::X, cute::_1>{});
  auto gB = cute::local_tile(
      mB, TileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
      cute::Step<cute::X, cute::_1, cute::_1>{});
  auto gA_tile = gA(
      cute::_, cute::_, tile_row, cute::_, cute::Int<0>{});
  auto gB_tile = gB(
      cute::_, cute::_, tile_col, cute::_, cute::Int<0>{});
  auto block_tma_a = tma_a.get_slice(0);
  auto block_tma_b = tma_b.get_slice(0);
  auto tAgA = block_tma_a.partition_S(gA_tile);
  auto tAsA = block_tma_a.partition_D(sA);
  auto tBgB = block_tma_b.partition_S(gB_tile);
  auto tBsB = block_tma_b.partition_D(sB);

  // TMA prologue: fill stage 0.
  if (is_tma_leader) {
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
        &storage.full_barrier[0], kTmaTransactionBytes);
    cute::copy(
        tma_a.with(storage.full_barrier[0]),
        tAgA(cute::_, cute::_, cute::_, 0),
        tAsA(cute::_, cute::_, cute::_, 0));
    cute::copy(
        tma_b.with(storage.full_barrier[0]),
        tBgB(cute::_, cute::_, cute::_, 0),
        tBsB(cute::_, cute::_, cute::_, 0));
  }

  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_slice(threadIdx.x);
  auto accum = cute::partition_fragment_C(
      tiled_mma, cute::make_shape(cute::_64{}, cute::_64{}));
  cute::clear(accum);

  for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
    int read_stage = k_tile % kStages;
    uint32_t read_phase = (k_tile / kStages) & 1;
    cutlass::arch::ClusterTransactionBarrier::wait(
        &storage.full_barrier[read_stage], read_phase);

    // Prefetch the next tile into the alternate stage before issuing WGMMA.
    int next_k_tile = k_tile + 1;
    if (next_k_tile < k_tile_count) {
      if (is_tma_leader) {
        int write_stage = next_k_tile % kStages;
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            &storage.full_barrier[write_stage], kTmaTransactionBytes);
        cute::copy(
            tma_a.with(storage.full_barrier[write_stage]),
            tAgA(cute::_, cute::_, cute::_, next_k_tile),
            tAsA(cute::_, cute::_, cute::_, write_stage));
        cute::copy(
            tma_b.with(storage.full_barrier[write_stage]),
            tBgB(cute::_, cute::_, cute::_, next_k_tile),
            tBsB(cute::_, cute::_, cute::_, write_stage));
      }
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

    if (threadIdx.x < kK) {
      float a_sum = 0.0f;
      float a_magnitude = 0.0f;
      float b_sum = 0.0f;
      float b_magnitude = 0.0f;
      for (int row = 0; row < kM; ++row) {
        float value = static_cast<float>(sA_stage(row, threadIdx.x));
        a_sum += value;
        a_magnitude += fabsf(value);
      }
      for (int col = 0; col < valid_cols; ++col) {
        float value = static_cast<float>(sB_stage(col, threadIdx.x));
        b_sum += value;
        b_magnitude += fabsf(value);
      }
      storage.stage_a_sums[threadIdx.x] = a_sum;
      storage.stage_a_magnitudes[threadIdx.x] = a_magnitude;
      storage.stage_b_sums[threadIdx.x] = b_sum;
      storage.stage_b_magnitudes[threadIdx.x] = b_magnitude;
    }
    __syncthreads();

    if (threadIdx.x < kM) {
      float row_expected = 0.0f;
      float row_scale = 0.0f;
      float col_expected = 0.0f;
      float col_scale = 0.0f;
      for (int kk = 0; kk < kK; ++kk) {
        float avalue = static_cast<float>(sA_stage(threadIdx.x, kk));
        row_expected += avalue * storage.stage_b_sums[kk];
        row_scale += fabsf(avalue) * storage.stage_b_magnitudes[kk];
        if (threadIdx.x < valid_cols) {
          float bvalue = static_cast<float>(sB_stage(threadIdx.x, kk));
          col_expected += storage.stage_a_sums[kk] * bvalue;
          col_scale += storage.stage_a_magnitudes[kk] * fabsf(bvalue);
        }
      }
      storage.expected_rows[threadIdx.x] += row_expected;
      storage.row_scales[threadIdx.x] += row_scale;
      if (threadIdx.x < valid_cols) {
        storage.expected_cols[threadIdx.x] += col_expected;
        storage.col_scales[threadIdx.x] += col_scale;
      }
    }
    __syncthreads();

    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(accum);
    __syncthreads();
  }

  auto tCsC = thread_mma.partition_C(sC);
  cute::copy(accum, tCsC);
  __syncthreads();

  if (threadIdx.x == 0) {
    storage.bad_row_count = 0;
    storage.bad_col_count = 0;
    storage.bad_row = -1;
    storage.bad_col = -1;
    if (inject_fault && tile_id == (tile_rows * tile_cols) / 2) {
      int row = kM / 2;
      int col = min(kN / 2, n - tile_col * kN - 1);
      if (col >= 0) {
        storage.c[col * kM + row] += fault_value;
      }
    }
  }
  __syncthreads();

  if (threadIdx.x < kM) {
    float actual = 0.0f;
    for (int col = 0; col < valid_cols; ++col) {
      actual += storage.c[col * kM + threadIdx.x];
    }
    float expected = storage.expected_rows[threadIdx.x];
    float delta = expected - actual;
    float scale = fmaxf(1.0f, storage.row_scales[threadIdx.x]);
    storage.row_deltas[threadIdx.x] = delta;
    if (fabsf(delta) > abs_tolerance + rel_tolerance * scale) {
      atomicAdd(&storage.bad_row_count, 1);
      atomicCAS(&storage.bad_row, -1, threadIdx.x);
    }
  }

  if (threadIdx.x < kN) {
    float delta = 0.0f;
    if (threadIdx.x < valid_cols) {
      float actual = 0.0f;
      for (int row = 0; row < kM; ++row) {
        actual += storage.c[threadIdx.x * kM + row];
      }
      float expected = storage.expected_cols[threadIdx.x];
      delta = expected - actual;
      float scale = fmaxf(1.0f, storage.col_scales[threadIdx.x]);
      if (fabsf(delta) > abs_tolerance + rel_tolerance * scale) {
        atomicAdd(&storage.bad_col_count, 1);
        atomicCAS(&storage.bad_col, -1, threadIdx.x);
      }
    }
    storage.col_deltas[threadIdx.x] = delta;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (storage.bad_row_count > 0 || storage.bad_col_count > 0) {
      atomicAdd(bad_tiles, 1);
    }
    if (storage.bad_row_count == 1 && storage.bad_col_count == 1) {
      storage.c[storage.bad_col * kM + storage.bad_row] +=
          0.5f * (storage.row_deltas[storage.bad_row] +
                  storage.col_deltas[storage.bad_col]);
      atomicAdd(corrected_tiles, 1);
    }
  }
  __syncthreads();

  for (int index = threadIdx.x; index < kM * kN; index += kThreads) {
    int row = index % kM;
    int col = index / kM;
    int global_col = tile_col * kN + col;
    if (global_col < n) {
      c[global_col * m + tile_row * kM + row] = storage.c[index];
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
  int* d_bad_tiles = nullptr;
  int* d_corrected_tiles = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_b, b_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(float)));
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
      wgmma_tma_fused_abft_kernel<decltype(tma_a), decltype(tma_b)>;
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  auto enqueue_gemm = [&]() {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = 1;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(tile_count, 1, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(
        &config, kernel,
        tma_a, tma_b, d_c, d_bad_tiles, d_corrected_tiles, options.m, options.n,
        options.k, tile_rows, tile_cols, options.abft_abs_tolerance,
        options.abft_rel_tolerance, options.fault_value, options.inject_fault));
  };

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
  double tflops =
      (2.0 * static_cast<double>(options.m) * options.n * options.k) /
      (average_ms * 1.0e9);
  VerificationResult verification =
      verify_samples(h_a, h_b, h_c, options);
  bool numerical_pass =
      options.verify_samples == 0 || verification.max_tolerance_ratio <= 1.0f;
  bool abft_pass = options.inject_fault
                       ? bad_tiles == 1 && corrected_tiles == 1
                       : bad_tiles == 0;
  bool passed = numerical_pass && abft_pass;

  std::printf("version: S3 Hopper TMA-WGMMA fused online ABFT\n");
  std::printf(
      "instruction_tile: 64x64x32, stages: %d, threads_per_cta: %d\n",
      kStages, kThreads);
  std::printf("shared_storage_bytes: %zu\n", sizeof(SharedStorage));
  std::printf("layout: A=[M,K] row-major, B=[N,K] column-major, "
              "C=[N,M] column-major\n");
  std::printf("shape: M=%d N=%d K=%d, output_tiles=%d\n", options.m,
              options.n, options.k, tile_count);
  std::printf("avg_fused_gemm_abft_time_ms: %.6f\n", average_ms);
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
  CUDA_CHECK(cudaFree(d_bad_tiles));
  CUDA_CHECK(cudaFree(d_corrected_tiles));
  return passed ? 0 : 1;
}
