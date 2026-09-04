#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cute/tensor.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/cooperative_gemm.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/arch/mma_sm90_gmma_ext.hpp"

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error = (call);                                                 \
    if (error != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(error));                                 \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

// This raw kernel uses one 64x64x32 GMMA atom per 128-thread CTA.
// CUTE is used only for the SM90 GMMA instruction, descriptor, and fragment
// mappings. No CUTLASS GEMM or CUTLASS epilogue is used.
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

constexpr int kThreads = 128;
constexpr int kM = 64;
constexpr int kN = 64;
constexpr int kK = 32;
constexpr int kDefaultPipelineBatchTiles = 128;
constexpr int kTileBytesA = kM * kK * sizeof(Element);
constexpr int kTileBytesB = kN * kK * sizeof(Element);
constexpr int kTileBytesC = kM * kN * sizeof(float);
constexpr int kGemmSmemBytes = kTileBytesA + kTileBytesB + kTileBytesC;

struct Options {
  int m = 256;
  int n = 192;
  int k = 256;
  int warmup = 2;
  int repeat = 10;
  float abs_tolerance = 1.0e-2f;
  float rel_tolerance = 2.0e-3f;
  float random_abs_tolerance = 5.0e-1f;
  float random_rel_tolerance = 1.0e-8f;
  int batch_tiles = kDefaultPipelineBatchTiles;
  bool inject_fault = false;
  bool no_abft = false;
  bool random_abs_tolerance_explicit = false;
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
  options.abs_tolerance =
      get_float(argc, argv, "--abs-tol", options.abs_tolerance);
  options.abs_tolerance =
      get_float(argc, argv, "--checksum-tol", options.abs_tolerance);
  options.rel_tolerance =
      get_float(argc, argv, "--rel-tol", options.rel_tolerance);
  options.random_abs_tolerance = get_float(
      argc, argv, "--random-abs-tol", options.random_abs_tolerance);
  options.random_rel_tolerance = get_float(
      argc, argv, "--random-rel-tol", options.random_rel_tolerance);
  options.batch_tiles =
      get_int(argc, argv, "--batch-tiles", options.batch_tiles);
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--inject-fault") == 0) {
      options.inject_fault = true;
    }
    if (std::strcmp(argv[i], "--no-abft") == 0) {
      options.no_abft = true;
    }
    if (std::strcmp(argv[i], "--random-abs-tol") == 0 ||
        std::strncmp(argv[i], "--random-abs-tol=", 17) == 0) {
      options.random_abs_tolerance_explicit = true;
    }
    if (std::strcmp(argv[i], "--help") == 0) {
      std::printf(
          "Usage: %s [--m N --n N --k N --warmup N --repeat N]\n"
          "       [--abs-tol X --rel-tol X --random-abs-tol X]\n"
          "       [--random-rel-tol X --batch-tiles N]\n"
          "       [--inject-fault --no-abft]\n"
          "Constraints: M %% 64 == 0, K %% 32 == 0; N may have a boundary tile.\n",
          argv[0]);
      std::exit(EXIT_SUCCESS);
    }
  }
  if (std::max(options.m, std::max(options.n, options.k)) > 1024 &&
      !options.random_abs_tolerance_explicit) {
    options.random_abs_tolerance = 4.8e1f;
  }
  if (options.m <= 0 || options.n <= 0 || options.k <= 0 ||
      options.m % kM != 0 || options.k % kK != 0 ||
      options.warmup < 0 || options.repeat <= 0 ||
      options.abs_tolerance < 0.0f ||
      options.rel_tolerance <= 0.0f || options.random_abs_tolerance < 0.0f ||
      options.random_rel_tolerance <= 0.0f || options.batch_tiles <= 0) {
    std::fprintf(stderr,
      "Invalid shape/options. Require M%%64==0 and K%%32==0.\n");
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

__global__ void generate_sign_weights_kernel(float* output, size_t count,
                                            uint32_t seed) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    uint32_t bits = mix_u32(seed + static_cast<uint32_t>(index));
    output[index] = (bits & 3U) == 0U
                        ? ((bits & 4U) ? 1.0f : -1.0f)
                        : 0.0f;
  }
}

__global__ void zero_b_padding_kernel(Element* b, int k, int n,
                                      int n_padded) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int padding_cols = n_padded - n;
  int total = padding_cols * k;
  if (index >= total) {
    return;
  }
  int col = n + index / k;
  int kk = index % k;
  b[col * k + kk] = Element(0.0f);
}

template <class Layout>
__device__ auto make_smem_tensor(Element* pointer, Layout layout) {
  return cute::make_tensor(cute::make_smem_ptr(pointer), layout);
}

__global__ void wgmma_gemm_kernel(
    const Element* a, const Element* b, float* c, float* actual_rows,
    float* actual_cols, float* actual_random_rows, float* actual_random_cols,
    const float* row_weights, const float* col_weights, int m, int n, int k,
    int tile_rows, int tile_cols,
    int tile_begin, int tile_count, bool inject_fault) {
  extern __shared__ __align__(16) unsigned char shared[];
  Element* a_smem = reinterpret_cast<Element*>(shared);
  Element* b_smem = reinterpret_cast<Element*>(shared + kTileBytesA);
  float* c_smem = reinterpret_cast<float*>(shared + kTileBytesA + kTileBytesB);

  int tile_id = tile_begin + blockIdx.x;
  int tile_row = tile_id / tile_cols;
  int tile_col = tile_id % tile_cols;
  if (blockIdx.x >= tile_count || tile_id >= tile_rows * tile_cols) {
    return;
  }

  auto sA = make_smem_tensor(a_smem, SmemLayoutA{});
  auto sB = make_smem_tensor(b_smem, SmemLayoutB{});
  using CLayout = decltype(cute::make_layout(
      cute::make_shape(cute::Int<kM>{}, cute::Int<kN>{}),
      cute::make_stride(cute::Int<1>{}, cute::Int<kM>{})));
  CLayout c_layout{};
  auto sC = cute::make_tensor(cute::make_smem_ptr(c_smem), c_layout);

  for (int index = threadIdx.x; index < kM * kN; index += kThreads) {
    c_smem[index] = 0.0f;
  }
  __syncthreads();

  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_slice(threadIdx.x);
  auto accum = cute::partition_fragment_C(tiled_mma,
                                           cute::make_shape(cute::_64{},
                                                            cute::_64{}));
  cute::clear(accum);

  for (int k_base = 0; k_base < k; k_base += kK) {
    for (int index = threadIdx.x; index < kM * kK; index += kThreads) {
      int row = index / kK;
      int kk = index % kK;
      sA(row, kk) =
          a[(tile_row * kM + row) * k + k_base + kk];
    }
    for (int index = threadIdx.x; index < kN * kK; index += kThreads) {
      int col = index / kK;
      int kk = index % kK;
      int global_col = tile_col * kN + col;
      sB(col, kk) = global_col < n
          ? b[global_col * k + k_base + kk]
          : Element(0.0f);
    }
    __syncthreads();

    auto tCsA = thread_mma.partition_A(sA);
    auto tCsB = thread_mma.partition_B(sB);
    auto tCrA = thread_mma.make_fragment_A(tCsA);
    auto tCrB = thread_mma.make_fragment_B(tCsB);
    cute::copy(tCsA, tCrA);
    cute::copy(tCsB, tCrB);

    tiled_mma.accumulate_ = k_base == 0
        ? cute::GMMA::ScaleOut::Zero
        : cute::GMMA::ScaleOut::One;
    cute::warpgroup_fence_operand(accum);
    cute::warpgroup_arrive();
    cute::gemm(tiled_mma, tCrA, tCrB, accum);
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(accum);
    __syncthreads();
  }

  auto tCsC = thread_mma.partition_C(sC);
  cute::copy(accum, tCsC);
  __syncthreads();

  if (inject_fault && tile_id == (tile_rows * tile_cols) / 2 &&
      threadIdx.x == 0) {
    int row = kM / 2;
    int col = kN / 2;
    if (actual_random_rows != nullptr && actual_random_cols != nullptr) {
      for (int candidate = 0; candidate < kM; ++candidate) {
        if (row_weights[tile_row * kM + candidate] != 0.0f) {
          row = candidate;
          break;
        }
      }
      for (int candidate = 0; candidate < kN; ++candidate) {
        if (tile_col * kN + candidate < n &&
            col_weights[tile_col * kN + candidate] != 0.0f) {
          col = candidate;
          break;
        }
      }
    }
    if (tile_col * kN + col < n) {
      c_smem[col * kM + row] += 1.0f;
    }
  }
  __syncthreads();

  if (actual_rows != nullptr && threadIdx.x < kM) {
    float actual = 0.0f;
    int valid_cols = min(kN, n - tile_col * kN);
    for (int col = 0; col < valid_cols; ++col) {
      actual += c_smem[col * kM + threadIdx.x];
    }
    actual_rows[tile_id * kM + threadIdx.x] = actual;
    if (actual_random_rows != nullptr) {
      float random_actual = 0.0f;
      for (int col = 0; col < valid_cols; ++col) {
        random_actual += c_smem[col * kM + threadIdx.x] *
                         col_weights[tile_col * kN + col];
      }
      actual_random_rows[tile_id * kM + threadIdx.x] = random_actual;
    }
  }
  if (actual_cols != nullptr && threadIdx.x < kN) {
    int valid_cols = min(kN, n - tile_col * kN);
    float actual = 0.0f;
    if (threadIdx.x < valid_cols) {
      for (int row = 0; row < kM; ++row) {
        actual += c_smem[threadIdx.x * kM + row];
      }
    }
    actual_cols[tile_id * kN + threadIdx.x] = actual;
    if (actual_random_cols != nullptr) {
      float random_actual = 0.0f;
      if (threadIdx.x < valid_cols) {
        for (int row = 0; row < kM; ++row) {
          random_actual += c_smem[threadIdx.x * kM + row] *
                           row_weights[tile_row * kM + row];
        }
      }
      actual_random_cols[tile_id * kN + threadIdx.x] = random_actual;
    }
  }

  for (int index = threadIdx.x; index < kM * kN; index += kThreads) {
    int row = index % kM;
    int col = index / kM;
    int global_col = tile_col * kN + col;
    if (global_col < n) {
      c[global_col * m + tile_row * kM + row] = c_smem[index];
    }
  }
}

__global__ void input_tile_sums_kernel(
    const Element* a, const Element* b, float* a_sums, float* a_magnitudes,
    float* b_sums, float* b_magnitudes, float* a_weighted_sums,
    float* b_weighted_sums, const float* row_weights, const float* col_weights,
    int m, int n, int n_padded, int k,
    int tile_rows, int tile_cols) {
  int tile_id = blockIdx.x;
  if (tile_id >= max(tile_rows, tile_cols)) {
    return;
  }

  if (tile_id < tile_rows) {
    for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
      float sum = 0.0f;
      float magnitude = 0.0f;
      float weighted_sum = 0.0f;
      for (int row = 0; row < kM; ++row) {
        float value = static_cast<float>(
            a[(tile_id * kM + row) * k + kk]);
        sum += value;
        magnitude += fabsf(value);
        weighted_sum += row_weights[tile_id * kM + row] * value;
      }
      a_sums[tile_id * k + kk] = sum;
      a_magnitudes[tile_id * k + kk] = magnitude;
      a_weighted_sums[tile_id * k + kk] = weighted_sum;
    }
  }

  if (tile_id < tile_cols) {
    int valid_cols = min(kN, n - tile_id * kN);
    for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
      float sum = 0.0f;
      float magnitude = 0.0f;
      float weighted_sum = 0.0f;
      for (int col = 0; col < valid_cols; ++col) {
        float value = static_cast<float>(
            b[(tile_id * kN + col) * k + kk]);
        sum += value;
        magnitude += fabsf(value);
        weighted_sum += col_weights[tile_id * kN + col] * value;
      }
      b_sums[tile_id * k + kk] = sum;
      b_magnitudes[tile_id * k + kk] = magnitude;
      b_weighted_sums[tile_id * k + kk] = weighted_sum;
    }
  }
}

__global__ void expected_tile_checksums_kernel(
    const Element* a, const Element* b, const float* a_sums,
    const float* a_magnitudes, const float* b_sums,
    const float* b_magnitudes, const float* a_weighted_sums,
    const float* b_weighted_sums, float* expected_rows, float* expected_cols,
    float* expected_random_rows, float* expected_random_cols,
    float* row_scales, float* col_scales, int m, int n, int k,
    int tile_rows, int tile_cols) {
  int tile_id = blockIdx.x;
  if (tile_id >= tile_rows * tile_cols) {
    return;
  }
  int tile_row = tile_id / tile_cols;
  int tile_col = tile_id % tile_cols;
  int valid_cols = min(kN, n - tile_col * kN);

  if (threadIdx.x < kM) {
    int global_row = tile_row * kM + threadIdx.x;
    float expected = 0.0f;
    float random_expected = 0.0f;
    float scale = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
      float avalue = static_cast<float>(a[global_row * k + kk]);
      expected += avalue * b_sums[tile_col * k + kk];
      random_expected += avalue * b_weighted_sums[tile_col * k + kk];
      scale += fabsf(avalue) * b_magnitudes[tile_col * k + kk];
    }
    expected_rows[tile_id * kM + threadIdx.x] = expected;
    expected_random_rows[tile_id * kM + threadIdx.x] = random_expected;
    row_scales[tile_id * kM + threadIdx.x] = scale;
  }

  if (threadIdx.x < kN) {
    int global_col = tile_col * kN + threadIdx.x;
    if (threadIdx.x < valid_cols) {
      float expected = 0.0f;
      float random_expected = 0.0f;
      float scale = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        float bvalue = static_cast<float>(b[global_col * k + kk]);
        expected += a_sums[tile_row * k + kk] * bvalue;
        random_expected += a_weighted_sums[tile_row * k + kk] * bvalue;
        scale += a_magnitudes[tile_row * k + kk] * fabsf(bvalue);
      }
      expected_cols[tile_id * kN + threadIdx.x] = expected;
      expected_random_cols[tile_id * kN + threadIdx.x] = random_expected;
      col_scales[tile_id * kN + threadIdx.x] = scale;
    } else {
      expected_cols[tile_id * kN + threadIdx.x] = 0.0f;
      expected_random_cols[tile_id * kN + threadIdx.x] = 0.0f;
      col_scales[tile_id * kN + threadIdx.x] = 0.0f;
    }
  }
}

__global__ void tile_checksum_kernel(
    const float* actual_rows, const float* actual_cols,
    const float* actual_random_rows, const float* actual_random_cols,
    const float* expected_rows, const float* expected_cols,
    const float* expected_random_rows, const float* expected_random_cols,
    float* row_deltas, float* col_deltas, float* random_row_deltas,
    float* random_col_deltas, int n, int tile_rows, int tile_cols, int tile_begin,
    int tile_count) {
  int tile_id = tile_begin + blockIdx.x;
  int tile_col = tile_id % tile_cols;
  if (blockIdx.x >= tile_count || tile_id >= tile_rows * tile_cols) {
    return;
  }

  int valid_cols = min(kN, n - tile_col * kN);
  if (threadIdx.x < kM) {
    row_deltas[tile_id * kM + threadIdx.x] =
        expected_rows[tile_id * kM + threadIdx.x] -
        actual_rows[tile_id * kM + threadIdx.x];
    random_row_deltas[tile_id * kM + threadIdx.x] =
        expected_random_rows[tile_id * kM + threadIdx.x] -
        actual_random_rows[tile_id * kM + threadIdx.x];
  }
  if (threadIdx.x < kN) {
    col_deltas[tile_id * kN + threadIdx.x] =
        threadIdx.x < valid_cols
            ? expected_cols[tile_id * kN + threadIdx.x] -
                  actual_cols[tile_id * kN + threadIdx.x]
            : 0.0f;
    random_col_deltas[tile_id * kN + threadIdx.x] =
        threadIdx.x < valid_cols
            ? expected_random_cols[tile_id * kN + threadIdx.x] -
                  actual_random_cols[tile_id * kN + threadIdx.x]
            : 0.0f;
  }
}

__device__ __forceinline__ void atomic_max_float(float* address, float value) {
  int* address_as_int = reinterpret_cast<int*>(address);
  int old = *address_as_int;
  while (value > __int_as_float(old)) {
    int assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    if (old == assumed) {
      break;
    }
  }
}

__global__ void tile_abft_check_kernel(
    float* c, const float* row_deltas, const float* col_deltas,
    const float* random_row_deltas, const float* random_col_deltas,
    const float* row_scales, const float* col_scales, int m, int n,
    int tile_rows, int tile_cols, int tile_begin, int tile_count,
    float abs_tolerance, float rel_tolerance, float random_abs_tolerance,
    float random_rel_tolerance, int* bad_tiles,
    int* corrected_tiles, float* max_normalized_error,
    float* max_absolute_error, float* max_random_absolute_error) {
  int tile_id = tile_begin + blockIdx.x;
  int tile_row = tile_id / tile_cols;
  int tile_col = tile_id % tile_cols;
  if (blockIdx.x >= tile_count || tile_id >= tile_rows * tile_cols) {
    return;
  }

  __shared__ int bad_row_count;
  __shared__ int bad_col_count;
  __shared__ int bad_row;
  __shared__ int bad_col;
  if (threadIdx.x == 0) {
    bad_row_count = 0;
    bad_col_count = 0;
    bad_row = -1;
    bad_col = -1;
  }
  __syncthreads();

  for (int row = threadIdx.x; row < kM; row += kThreads) {
    float diff = fabsf(row_deltas[tile_id * kM + row]);
    float random_diff = fabsf(random_row_deltas[tile_id * kM + row]);
    float scale = fmaxf(1.0f, row_scales[tile_id * kM + row]);
    atomic_max_float(max_absolute_error, diff);
    atomic_max_float(max_random_absolute_error, random_diff);
    atomic_max_float(max_normalized_error, diff / scale);
    bool bad = diff > abs_tolerance + rel_tolerance * scale ||
               random_diff > random_abs_tolerance + random_rel_tolerance * scale;
    if (bad) {
      atomicAdd(&bad_row_count, 1);
      atomicCAS(&bad_row, -1, row);
    }
  }

  int valid_cols = min(kN, n - tile_col * kN);
  for (int col = threadIdx.x; col < valid_cols; col += kThreads) {
    float diff = fabsf(col_deltas[tile_id * kN + col]);
    float random_diff = fabsf(random_col_deltas[tile_id * kN + col]);
    float scale = fmaxf(1.0f, col_scales[tile_id * kN + col]);
    atomic_max_float(max_absolute_error, diff);
    atomic_max_float(max_random_absolute_error, random_diff);
    atomic_max_float(max_normalized_error, diff / scale);
    bool bad = diff > abs_tolerance + rel_tolerance * scale ||
               random_diff > random_abs_tolerance + random_rel_tolerance * scale;
    if (bad) {
      atomicAdd(&bad_col_count, 1);
      atomicCAS(&bad_col, -1, col);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && bad_row_count > 0 && bad_col_count > 0) {
    atomicAdd(bad_tiles, 1);
  }
  if (bad_row_count == 1 && bad_col_count == 1 && threadIdx.x == 0) {
    int global_row = tile_row * kM + bad_row;
    int global_col = tile_col * kN + bad_col;
    float row_delta = row_deltas[tile_id * kM + bad_row];
    float col_delta = col_deltas[tile_id * kN + bad_col];
    c[global_col * m + global_row] += 0.5f * (row_delta + col_delta);
    atomicAdd(corrected_tiles, 1);
  }
}

static float max_reference_error(const std::vector<Element>& a,
                                 const std::vector<Element>& b,
                                 const std::vector<float>& c, int m, int n,
                                 int k) {
  float maximum = 0.0f;
  for (int row = 0; row < m; row += std::max(1, m / 8)) {
    for (int col = 0; col < n; col += std::max(1, n / 8)) {
      double reference = 0.0;
      for (int kk = 0; kk < k; ++kk) {
        reference += static_cast<double>(
                         static_cast<float>(a[row * k + kk])) *
                     static_cast<double>(
                         static_cast<float>(b[col * k + kk]));
      }
      maximum = std::max(
          maximum, static_cast<float>(
                       std::fabs(c[col * m + row] - reference)));
    }
  }
  return maximum;
}

int main(int argc, char** argv) {
  Options options = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(0));
  size_t a_count = static_cast<size_t>(options.m) * options.k;
  int n_padded = div_up(options.n, kN) * kN;
  size_t b_count = static_cast<size_t>(options.k) * n_padded;
  size_t c_count = static_cast<size_t>(options.m) * options.n;
  int tile_rows = options.m / kM;
  int tile_cols = div_up(options.n, kN);
  int tile_count = tile_rows * tile_cols;

  Element* d_a = nullptr;
  Element* d_b = nullptr;
  float* d_c = nullptr;
  float* d_row_deltas = nullptr;
  float* d_col_deltas = nullptr;
  float* d_random_row_deltas = nullptr;
  float* d_random_col_deltas = nullptr;
  float* d_row_scales = nullptr;
  float* d_col_scales = nullptr;
  float* d_expected_rows = nullptr;
  float* d_expected_cols = nullptr;
  float* d_expected_random_rows = nullptr;
  float* d_expected_random_cols = nullptr;
  float* d_a_sums = nullptr;
  float* d_a_magnitudes = nullptr;
  float* d_b_sums = nullptr;
  float* d_b_magnitudes = nullptr;
  float* d_a_weighted_sums = nullptr;
  float* d_b_weighted_sums = nullptr;
  float* d_row_weights = nullptr;
  float* d_col_weights = nullptr;
  int* d_bad_tiles = nullptr;
  int* d_corrected_tiles = nullptr;
  float* d_max_normalized_error = nullptr;
  float* d_max_absolute_error = nullptr;
  float* d_max_random_absolute_error = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_b, b_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_row_deltas, tile_count * kM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_col_deltas, tile_count * kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_random_row_deltas, tile_count * kM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_random_col_deltas, tile_count * kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_row_scales, tile_count * kM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_col_scales, tile_count * kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_expected_rows, tile_count * kM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_expected_cols, tile_count * kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_expected_random_rows, tile_count * kM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_expected_random_cols, tile_count * kN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_a_sums, tile_rows * options.k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_a_magnitudes,
                        tile_rows * options.k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b_sums, tile_cols * options.k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b_magnitudes,
                        tile_cols * options.k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_a_weighted_sums,
                        tile_rows * options.k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b_weighted_sums,
                        tile_cols * options.k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_row_weights, options.m * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_col_weights, n_padded * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bad_tiles, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_corrected_tiles, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_max_normalized_error, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_max_absolute_error, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_max_random_absolute_error, sizeof(float)));

  constexpr int threads = 256;
  generate_fp8_kernel<<<static_cast<int>((a_count + threads - 1) / threads),
                        threads>>>(d_a, a_count, 0x12345678U);
  generate_fp8_kernel<<<static_cast<int>((b_count + threads - 1) / threads),
                        threads>>>(d_b, b_count, 0x9abcdef0U);
  generate_sign_weights_kernel<<<(options.m + threads - 1) / threads, threads>>>(
      d_row_weights, options.m, 0x13579bdfU);
  generate_sign_weights_kernel<<<(n_padded + threads - 1) / threads, threads>>>(
      d_col_weights, n_padded, 0x2468ace0U);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  if (n_padded > options.n) {
    int padding_count = (n_padded - options.n) * options.k;
    zero_b_padding_kernel<<<(padding_count + threads - 1) / threads, threads>>>(
        d_b, options.k, options.n, n_padded);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  if (options.no_abft) {
    cudaStream_t baseline_stream;
    cudaEvent_t baseline_start;
    cudaEvent_t baseline_stop;
    CUDA_CHECK(cudaStreamCreateWithFlags(&baseline_stream,
                                         cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&baseline_start));
    CUDA_CHECK(cudaEventCreate(&baseline_stop));
    auto enqueue_gemm = [&]() {
      wgmma_gemm_kernel<<<tile_count, kThreads, kGemmSmemBytes,
                          baseline_stream>>>(
          d_a, d_b, d_c, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, options.m, options.n, options.k,
          tile_rows, tile_cols, 0, tile_count, false);
    };
    for (int i = 0; i < options.warmup; ++i) {
      enqueue_gemm();
    }
    CUDA_CHECK(cudaStreamSynchronize(baseline_stream));
    float total_ms = 0.0f;
    for (int i = 0; i < options.repeat; ++i) {
      CUDA_CHECK(cudaEventRecord(baseline_start, baseline_stream));
      enqueue_gemm();
      CUDA_CHECK(cudaEventRecord(baseline_stop, baseline_stream));
      CUDA_CHECK(cudaEventSynchronize(baseline_stop));
      float iteration_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&iteration_ms, baseline_start,
                                      baseline_stop));
      total_ms += iteration_ms;
    }
    double average_ms = total_ms / options.repeat;
    double tflops =
        (2.0 * static_cast<double>(options.m) * options.n * options.k) /
        (average_ms * 1.0e9);
    std::printf("kernel: raw CUDA + CUTE WGMMA GEMM-only baseline\n");
    std::printf("shape: M=%d N=%d K=%d, tile=64x64x32\n", options.m,
                options.n, options.k);
    std::printf("avg_gemm_only_time_ms: %.6f\n", average_ms);
    std::printf("gemm_only_tflops: %.6f\n", tflops);
    CUDA_CHECK(cudaEventDestroy(baseline_start));
    CUDA_CHECK(cudaEventDestroy(baseline_stop));
    CUDA_CHECK(cudaStreamDestroy(baseline_stream));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_row_deltas));
    CUDA_CHECK(cudaFree(d_col_deltas));
    CUDA_CHECK(cudaFree(d_random_row_deltas));
    CUDA_CHECK(cudaFree(d_random_col_deltas));
    CUDA_CHECK(cudaFree(d_row_scales));
    CUDA_CHECK(cudaFree(d_col_scales));
    CUDA_CHECK(cudaFree(d_expected_rows));
    CUDA_CHECK(cudaFree(d_expected_cols));
    CUDA_CHECK(cudaFree(d_expected_random_rows));
    CUDA_CHECK(cudaFree(d_expected_random_cols));
    CUDA_CHECK(cudaFree(d_a_sums));
    CUDA_CHECK(cudaFree(d_a_magnitudes));
    CUDA_CHECK(cudaFree(d_b_sums));
    CUDA_CHECK(cudaFree(d_b_magnitudes));
    CUDA_CHECK(cudaFree(d_a_weighted_sums));
    CUDA_CHECK(cudaFree(d_b_weighted_sums));
    CUDA_CHECK(cudaFree(d_row_weights));
    CUDA_CHECK(cudaFree(d_col_weights));
    CUDA_CHECK(cudaFree(d_bad_tiles));
    CUDA_CHECK(cudaFree(d_corrected_tiles));
    CUDA_CHECK(cudaFree(d_max_normalized_error));
    CUDA_CHECK(cudaFree(d_max_absolute_error));
    CUDA_CHECK(cudaFree(d_max_random_absolute_error));
    return 0;
  }
  cudaStream_t gemm_stream;
  cudaStream_t checksum_stream;
  cudaStream_t abft_stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&gemm_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&checksum_stream,
                                      cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&abft_stream, cudaStreamNonBlocking));

  cudaEvent_t metadata_start, metadata_done;
  CUDA_CHECK(cudaEventCreate(&metadata_start));
  CUDA_CHECK(cudaEventCreate(&metadata_done));
  CUDA_CHECK(cudaEventRecord(metadata_start, checksum_stream));
  input_tile_sums_kernel<<<std::max(tile_rows, tile_cols), kThreads, 0,
                           checksum_stream>>>(
      d_a, d_b, d_a_sums, d_a_magnitudes, d_b_sums, d_b_magnitudes,
      d_a_weighted_sums, d_b_weighted_sums, d_row_weights, d_col_weights,
      options.m,
      options.n, n_padded, options.k, tile_rows, tile_cols);
  expected_tile_checksums_kernel<<<tile_count, kThreads, 0, checksum_stream>>>(
      d_a, d_b, d_a_sums, d_a_magnitudes, d_b_sums, d_b_magnitudes,
      d_a_weighted_sums, d_b_weighted_sums, d_expected_rows, d_expected_cols,
      d_expected_random_rows, d_expected_random_cols, d_row_scales,
      d_col_scales, options.m,
      options.n, options.k, tile_rows, tile_cols);
  CUDA_CHECK(cudaEventRecord(metadata_done, checksum_stream));
  CUDA_CHECK(cudaEventSynchronize(metadata_done));

  int batch_count = div_up(tile_count, options.batch_tiles);
  std::vector<cudaEvent_t> gemm_start(batch_count);
  std::vector<cudaEvent_t> gemm_done(batch_count);
  std::vector<cudaEvent_t> checksum_start(batch_count);
  std::vector<cudaEvent_t> checksum_done(batch_count);
  std::vector<cudaEvent_t> abft_start(batch_count);
  std::vector<cudaEvent_t> abft_done(batch_count);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int batch = 0; batch < batch_count; ++batch) {
    CUDA_CHECK(cudaEventCreate(&gemm_start[batch]));
    CUDA_CHECK(cudaEventCreate(&gemm_done[batch]));
    CUDA_CHECK(cudaEventCreate(&checksum_start[batch]));
    CUDA_CHECK(cudaEventCreate(&checksum_done[batch]));
    CUDA_CHECK(cudaEventCreate(&abft_start[batch]));
    CUDA_CHECK(cudaEventCreate(&abft_done[batch]));
  }

  auto enqueue_iteration = [&]() {
    for (int batch = 0; batch < batch_count; ++batch) {
      int tile_begin = batch * options.batch_tiles;
      int tiles_this_batch =
          std::min(options.batch_tiles, tile_count - tile_begin);
      CUDA_CHECK(cudaEventRecord(gemm_start[batch], gemm_stream));
      wgmma_gemm_kernel<<<tiles_this_batch, kThreads, kGemmSmemBytes,
                          gemm_stream>>>(
          d_a, d_b, d_c, d_row_deltas, d_col_deltas, d_random_row_deltas,
          d_random_col_deltas, d_row_weights, d_col_weights, options.m,
          options.n,
          options.k, tile_rows, tile_cols, tile_begin, tiles_this_batch,
          options.inject_fault);
      CUDA_CHECK(cudaEventRecord(gemm_done[batch], gemm_stream));

      CUDA_CHECK(cudaStreamWaitEvent(checksum_stream, gemm_done[batch], 0));
      CUDA_CHECK(cudaEventRecord(checksum_start[batch], checksum_stream));
      tile_checksum_kernel<<<tiles_this_batch, kThreads, 0, checksum_stream>>>(
          d_row_deltas, d_col_deltas, d_random_row_deltas, d_random_col_deltas,
          d_expected_rows, d_expected_cols, d_expected_random_rows,
          d_expected_random_cols, d_row_deltas, d_col_deltas,
          d_random_row_deltas, d_random_col_deltas, options.n, tile_rows, tile_cols,
          tile_begin, tiles_this_batch);
      CUDA_CHECK(cudaEventRecord(checksum_done[batch], checksum_stream));

      CUDA_CHECK(cudaStreamWaitEvent(abft_stream, checksum_done[batch], 0));
      if (batch == 0) {
        CUDA_CHECK(cudaMemsetAsync(d_bad_tiles, 0, sizeof(int), abft_stream));
        CUDA_CHECK(cudaMemsetAsync(d_corrected_tiles, 0, sizeof(int),
                                   abft_stream));
        CUDA_CHECK(cudaMemsetAsync(d_max_normalized_error, 0, sizeof(float),
                                   abft_stream));
        CUDA_CHECK(cudaMemsetAsync(d_max_absolute_error, 0, sizeof(float),
                                   abft_stream));
        CUDA_CHECK(cudaMemsetAsync(d_max_random_absolute_error, 0,
                                   sizeof(float), abft_stream));
      }
      CUDA_CHECK(cudaEventRecord(abft_start[batch], abft_stream));
      tile_abft_check_kernel<<<tiles_this_batch, kThreads, 0, abft_stream>>>(
          d_c, d_row_deltas, d_col_deltas, d_random_row_deltas,
          d_random_col_deltas, d_row_scales, d_col_scales, options.m,
          options.n, tile_rows, tile_cols, tile_begin,
          tiles_this_batch, options.abs_tolerance, options.rel_tolerance,
          options.random_abs_tolerance, options.random_rel_tolerance,
          d_bad_tiles, d_corrected_tiles, d_max_normalized_error,
          d_max_absolute_error, d_max_random_absolute_error);
      CUDA_CHECK(cudaEventRecord(abft_done[batch], abft_stream));
    }
  };

  for (int i = 0; i < options.warmup; ++i) {
    enqueue_iteration();
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  float total_elapsed_ms = 0.0f;
  float gemm_elapsed_ms = 0.0f;
  float checksum_elapsed_ms = 0.0f;
  float abft_elapsed_ms = 0.0f;
  for (int i = 0; i < options.repeat; ++i) {
    CUDA_CHECK(cudaEventRecord(start, gemm_stream));
    enqueue_iteration();
    CUDA_CHECK(cudaEventRecord(stop, abft_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float iteration_ms = 0.0f;
    float iteration_gemm_ms = 0.0f;
    float iteration_checksum_ms = 0.0f;
    float iteration_abft_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&iteration_ms, start, stop));
    for (int batch = 0; batch < batch_count; ++batch) {
      float batch_gemm_ms = 0.0f;
      float batch_checksum_ms = 0.0f;
      float batch_abft_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&batch_gemm_ms, gemm_start[batch],
                                      gemm_done[batch]));
      CUDA_CHECK(cudaEventElapsedTime(&batch_checksum_ms,
                                      checksum_start[batch],
                                      checksum_done[batch]));
      CUDA_CHECK(cudaEventElapsedTime(&batch_abft_ms, abft_start[batch],
                                      abft_done[batch]));
      iteration_gemm_ms += batch_gemm_ms;
      iteration_checksum_ms += batch_checksum_ms;
      iteration_abft_ms += batch_abft_ms;
    }
    total_elapsed_ms += iteration_ms;
    gemm_elapsed_ms += iteration_gemm_ms;
    checksum_elapsed_ms += iteration_checksum_ms;
    abft_elapsed_ms += iteration_abft_ms;
  }
  float elapsed_ms = total_elapsed_ms;
  float metadata_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&metadata_ms, metadata_start,
                                  metadata_done));

  std::vector<Element> h_a(a_count);
  std::vector<Element> h_b(b_count);
  std::vector<float> h_c(c_count);
  CUDA_CHECK(cudaMemcpy(h_a.data(), d_a, a_count * sizeof(Element),
                       cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, b_count * sizeof(Element),
                       cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, c_count * sizeof(float),
                       cudaMemcpyDeviceToHost));
  float error = max_reference_error(h_a, h_b, h_c, options.m, options.n,
                                    options.k);

  int bad_tiles = 0;
  int corrected_tiles = 0;
  float max_normalized_error = 0.0f;
  float max_absolute_error = 0.0f;
  float max_random_absolute_error = 0.0f;
  CUDA_CHECK(cudaMemcpy(&bad_tiles, d_bad_tiles, sizeof(int),
                       cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&corrected_tiles, d_corrected_tiles, sizeof(int),
                       cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&max_normalized_error, d_max_normalized_error,
                        sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&max_absolute_error, d_max_absolute_error,
                        sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&max_random_absolute_error,
                        d_max_random_absolute_error, sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::printf("version: S2 raw CUDA + CUTE WGMMA pipeline ABFT\n");
  std::printf("kernel: raw CUDA + CUTE WGMMA 3-stage tiled ABFT\n");
  std::printf("shape: M=%d N=%d K=%d, tile=64x64x32\n", options.m, options.n,
              options.k);
  std::printf("tile_grid: %dx%d, tile_count=%d\n", tile_rows, tile_cols,
              tile_count);
  std::printf("pipeline: 3-stage batches, batch_tiles=%d, batch_count=%d\n",
              options.batch_tiles, batch_count);
  std::printf("input_checksum_metadata_prepare_time_ms: %.6f\n",
              metadata_ms);
  std::printf("overlap: GEMM -> optimized checksum -> tile-local ABFT; "
              "successive batches overlap across streams\n");
  double average_ms = elapsed_ms / options.repeat;
  double average_gemm_ms = gemm_elapsed_ms / options.repeat;
  double average_checksum_ms = checksum_elapsed_ms / options.repeat;
  double average_abft_ms = abft_elapsed_ms / options.repeat;
  std::printf("avg_overlapped_gemm_checksum_abft_time_ms: %.6f\n",
              average_ms);
  std::printf("avg_gemm_stream_kernel_time_ms: %.6f\n", average_gemm_ms);
  std::printf("avg_checksum_stream_time_ms: %.6f\n", average_checksum_ms);
  std::printf("avg_abft_stream_kernel_time_ms: %.6f\n", average_abft_ms);
  std::printf("checksum_tolerance: abs=%.6e rel=%.6e\n",
              options.abs_tolerance, options.rel_tolerance);
  std::printf("max_input_scaled_checksum_error: %.6e\n",
              max_normalized_error);
  std::printf("max_absolute_checksum_error: %.6e\n", max_absolute_error);
  std::printf("random_checksum_tolerance: abs=%.6e rel=%.6e\n",
              options.random_abs_tolerance, options.random_rel_tolerance);
  std::printf("max_random_checksum_error: %.6e\n",
              max_random_absolute_error);
  double tflops =
      (2.0 * static_cast<double>(options.m) * options.n * options.k) /
      (average_ms * 1.0e9);
  std::printf("overlapped_gemm_tflops: %.6f\n", tflops);
  std::printf("sampled_max_error: %.6e\n", error);
  std::printf("bad_tiles: %d\n", bad_tiles);
  std::printf("corrected_tiles: %d\n", corrected_tiles);
  std::printf("abft_detection: %s\n",
              bad_tiles == 0 ? "CLEAN" : "FAULT_DETECTED");
  std::printf("abft_verification: %s\n",
              bad_tiles == 0
                  ? "PASS"
                  : (options.inject_fault && bad_tiles == 1 &&
                             corrected_tiles == 1
                         ? "PASS_AFTER_SINGLE_CORRECTION"
                         : "FAIL"));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(metadata_start));
  CUDA_CHECK(cudaEventDestroy(metadata_done));
  for (int batch = 0; batch < batch_count; ++batch) {
    CUDA_CHECK(cudaEventDestroy(gemm_start[batch]));
    CUDA_CHECK(cudaEventDestroy(gemm_done[batch]));
    CUDA_CHECK(cudaEventDestroy(checksum_start[batch]));
    CUDA_CHECK(cudaEventDestroy(checksum_done[batch]));
    CUDA_CHECK(cudaEventDestroy(abft_start[batch]));
    CUDA_CHECK(cudaEventDestroy(abft_done[batch]));
  }
  CUDA_CHECK(cudaStreamDestroy(gemm_stream));
  CUDA_CHECK(cudaStreamDestroy(checksum_stream));
  CUDA_CHECK(cudaStreamDestroy(abft_stream));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_row_deltas));
  CUDA_CHECK(cudaFree(d_col_deltas));
  CUDA_CHECK(cudaFree(d_row_scales));
  CUDA_CHECK(cudaFree(d_col_scales));
  CUDA_CHECK(cudaFree(d_expected_rows));
  CUDA_CHECK(cudaFree(d_expected_cols));
  CUDA_CHECK(cudaFree(d_a_sums));
  CUDA_CHECK(cudaFree(d_a_magnitudes));
  CUDA_CHECK(cudaFree(d_b_sums));
  CUDA_CHECK(cudaFree(d_b_magnitudes));
  CUDA_CHECK(cudaFree(d_max_normalized_error));
  CUDA_CHECK(cudaFree(d_max_absolute_error));
  CUDA_CHECK(cudaFree(d_bad_tiles));
  CUDA_CHECK(cudaFree(d_corrected_tiles));
  return 0;
}
