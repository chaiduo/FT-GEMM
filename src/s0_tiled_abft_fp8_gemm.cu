#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                                             \
  do {                                                                                               \
    cudaError_t error = (call);                                                                      \
    if (error != cudaSuccess) {                                                                      \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      std::exit(EXIT_FAILURE);                                                                       \
    }                                                                                                \
  } while (0)

using Element = __nv_fp8_e4m3;
constexpr int kTile = 16;

struct Options {
  int m = 256;
  int n = 256;
  int k = 256;
  int warmup = 2;
  int repeat = 10;
  int verify_samples = 8;
  float abs_tolerance = 5.0e-2f;
  float rel_tolerance = 1.0e-5f;
  bool inject_fault = false;
};

static int get_int(int argc, char** argv, const char* name, int value) {
  size_t length = std::strlen(name);
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
      return std::atoi(argv[i + 1]);
    }
    if (std::strncmp(argv[i], name, length) == 0 && argv[i][length] == '=') {
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
    if (std::strncmp(argv[i], name, length) == 0 && argv[i][length] == '=') {
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
  options.verify_samples = get_int(argc, argv, "--verify-samples", options.verify_samples);
  options.abs_tolerance = get_float(argc, argv, "--abs-tol", options.abs_tolerance);
  options.rel_tolerance = get_float(argc, argv, "--rel-tol", options.rel_tolerance);
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--inject-fault") == 0) {
      options.inject_fault = true;
    }
    if (std::strcmp(argv[i], "--help") == 0) {
      std::printf(
          "Usage: %s [--m N --n N --k N --warmup N --repeat N]\n"
          "       [--abs-tol X --rel-tol X --verify-samples N]\n"
          "       [--inject-fault]\n",
          argv[0]);
      std::exit(EXIT_SUCCESS);
    }
  }
  if (options.m <= 0 || options.n <= 0 || options.k <= 0 || options.warmup < 0 || options.repeat <= 0 || options.verify_samples < 0 ||
      options.abs_tolerance < 0.0f || options.rel_tolerance <= 0.0f) {
    std::fprintf(stderr, "Invalid shape, tolerance, or iteration options.\n");
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

__global__ void generate_fp8_kernel(Element* output, size_t count, uint32_t seed) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  uint32_t bits = mix_u32(seed + static_cast<uint32_t>(index));
  float unit = static_cast<float>(bits & 0x00ffffffU) / 16777216.0f;
  output[index] = Element(2.0f * unit - 1.0f);
}

__global__ void native_tiled_fp8_gemm_kernel(const Element* a, const Element* b, float* c, int m, int n, int k) {
  __shared__ Element a_tile[kTile][kTile];
  __shared__ Element b_tile[kTile][kTile];
  int row = blockIdx.y * kTile + threadIdx.y;
  int col = blockIdx.x * kTile + threadIdx.x;
  float accumulator = 0.0f;

  for (int k_base = 0; k_base < k; k_base += kTile) {
    int a_col = k_base + threadIdx.x;
    int b_row = k_base + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] = row < m && a_col < k ? a[row * k + a_col] : Element(0.0f);
    b_tile[threadIdx.y][threadIdx.x] = b_row < k && col < n ? b[b_row * n + col] : Element(0.0f);
    __syncthreads();
    for (int kk = 0; kk < kTile; ++kk) {
      accumulator += static_cast<float>(a_tile[threadIdx.y][kk]) * static_cast<float>(b_tile[kk][threadIdx.x]);
    }
    __syncthreads();
  }
  if (row < m && col < n) {
    c[row * n + col] = accumulator;
  }
}

__global__ void prepare_expected_checksum_kernel(const Element* a, const Element* b, float* expected_rows, float* expected_cols,
                                                 float* row_scales, float* col_scales, int m, int n, int k, int tile_rows, int tile_cols) {
  int tile_id = blockIdx.x;
  if (tile_id >= tile_rows * tile_cols) {
    return;
  }
  int tile_row = tile_id / tile_cols;
  int tile_col = tile_id % tile_cols;
  int valid_rows = min(kTile, m - tile_row * kTile);
  int valid_cols = min(kTile, n - tile_col * kTile);

  if (threadIdx.x < valid_rows) {
    int row = tile_row * kTile + threadIdx.x;
    float expected = 0.0f;
    float scale = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
      float b_sum = 0.0f;
      float b_abs_sum = 0.0f;
      for (int local_col = 0; local_col < valid_cols; ++local_col) {
        float value = static_cast<float>(b[kk * n + tile_col * kTile + local_col]);
        b_sum += value;
        b_abs_sum += fabsf(value);
      }
      float avalue = static_cast<float>(a[row * k + kk]);
      expected += avalue * b_sum;
      scale += fabsf(avalue) * b_abs_sum;
    }
    expected_rows[tile_id * kTile + threadIdx.x] = expected;
    row_scales[tile_id * kTile + threadIdx.x] = scale;
  }

  if (threadIdx.x < valid_cols) {
    int col = tile_col * kTile + threadIdx.x;
    float expected = 0.0f;
    float scale = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
      float a_sum = 0.0f;
      float a_abs_sum = 0.0f;
      for (int local_row = 0; local_row < valid_rows; ++local_row) {
        float value = static_cast<float>(a[(tile_row * kTile + local_row) * k + kk]);
        a_sum += value;
        a_abs_sum += fabsf(value);
      }
      float bvalue = static_cast<float>(b[kk * n + col]);
      expected += a_sum * bvalue;
      scale += a_abs_sum * fabsf(bvalue);
    }
    expected_cols[tile_id * kTile + threadIdx.x] = expected;
    col_scales[tile_id * kTile + threadIdx.x] = scale;
  }
}

__device__ __forceinline__ bool checksum_bad(float actual, float expected, float scale, float abs_tolerance, float rel_tolerance) {
  return fabsf(actual - expected) > abs_tolerance + rel_tolerance * fmaxf(1.0f, scale);
}

// One CTA owns one C tile. Expected checksums are prepared before GEMM;
// this kernel computes actual checksums, compares them, and corrects C.
__global__ void tile_abft_kernel(float* c, const float* expected_rows, const float* expected_cols, const float* row_scales,
                                 const float* col_scales, int m, int n, int tile_rows, int tile_cols, float abs_tolerance,
                                 float rel_tolerance, bool inject_fault, int* bad_tiles, int* corrected_tiles) {
  __shared__ float c_tile[kTile][kTile];
  __shared__ int bad_row_count;
  __shared__ int bad_col_count;
  __shared__ int bad_row;
  __shared__ int bad_col;
  __shared__ float row_deltas[kTile];
  __shared__ float col_deltas[kTile];

  int tile_id = blockIdx.x;
  int tile_row = tile_id / tile_cols;
  int tile_col = tile_id % tile_cols;
  if (tile_id >= tile_rows * tile_cols) {
    return;
  }
  int row = tile_row * kTile + threadIdx.y;
  int col = tile_col * kTile + threadIdx.x;
  int valid_rows = min(kTile, m - tile_row * kTile);
  int valid_cols = min(kTile, n - tile_col * kTile);

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    bad_row_count = 0;
    bad_col_count = 0;
    bad_row = -1;
    bad_col = -1;
    if (inject_fault && tile_id == (tile_rows * tile_cols) / 2) {
      int fault_row = tile_row * kTile + valid_rows / 2;
      int fault_col = tile_col * kTile + valid_cols / 2;
      if (fault_row < m && fault_col < n) {
        c[fault_row * n + fault_col] += 1.0f;
      }
    }
  }
  __syncthreads();

  if (threadIdx.y < valid_rows && threadIdx.x < valid_cols) {
    c_tile[threadIdx.y][threadIdx.x] = c[row * n + col];
  } else {
    c_tile[threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y < valid_rows) {
    float actual = 0.0f;
    for (int local_col = 0; local_col < valid_cols; ++local_col) {
      actual += c_tile[threadIdx.y][local_col];
    }
    float expected = expected_rows[tile_id * kTile + threadIdx.y];
    float scale = row_scales[tile_id * kTile + threadIdx.y];
    row_deltas[threadIdx.y] = expected - actual;
    if (checksum_bad(actual, expected, scale, abs_tolerance, rel_tolerance)) {
      atomicAdd(&bad_row_count, 1);
      atomicCAS(&bad_row, -1, threadIdx.y);
    }
  }

  if (threadIdx.y == 0 && threadIdx.x < valid_cols) {
    float actual = 0.0f;
    for (int local_row = 0; local_row < valid_rows; ++local_row) {
      actual += c_tile[local_row][threadIdx.x];
    }
    float expected = expected_cols[tile_id * kTile + threadIdx.x];
    float scale = col_scales[tile_id * kTile + threadIdx.x];
    col_deltas[threadIdx.x] = expected - actual;
    if (checksum_bad(actual, expected, scale, abs_tolerance, rel_tolerance)) {
      atomicAdd(&bad_col_count, 1);
      atomicCAS(&bad_col, -1, threadIdx.x);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (bad_row_count > 0 && bad_col_count > 0) {
      atomicAdd(bad_tiles, 1);
    }
    if (bad_row_count == 1 && bad_col_count == 1) {
      int fault_row = tile_row * kTile + bad_row;
      int fault_col = tile_col * kTile + bad_col;
      c[fault_row * n + fault_col] += 0.5f * (row_deltas[bad_row] + col_deltas[bad_col]);
      atomicAdd(corrected_tiles, 1);
    }
  }
}

static float verify_samples(const std::vector<Element>& a, const std::vector<Element>& b, const std::vector<float>& c,
                            const Options& options) {
  float maximum = 0.0f;
  int row_step = std::max(1, options.m / std::max(1, options.verify_samples));
  int col_step = std::max(1, options.n / std::max(1, options.verify_samples));
  int checked_rows = 0;
  for (int row = 0; row < options.m && checked_rows < options.verify_samples; row += row_step, ++checked_rows) {
    for (int col = 0; col < options.n; col += col_step) {
      double reference = 0.0;
      for (int kk = 0; kk < options.k; ++kk) {
        reference += static_cast<double>(static_cast<float>(a[row * options.k + kk])) *
                     static_cast<double>(static_cast<float>(b[kk * options.n + col]));
      }
      maximum = std::max(maximum, static_cast<float>(std::fabs(c[row * options.n + col] - reference)));
    }
  }
  return maximum;
}

int main(int argc, char** argv) {
  Options options = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(0));
  size_t a_count = static_cast<size_t>(options.m) * options.k;
  size_t b_count = static_cast<size_t>(options.k) * options.n;
  size_t c_count = static_cast<size_t>(options.m) * options.n;
  int tile_rows = (options.m + kTile - 1) / kTile;
  int tile_cols = (options.n + kTile - 1) / kTile;
  int tile_count = tile_rows * tile_cols;

  Element* d_a = nullptr;
  Element* d_b = nullptr;
  float* d_c = nullptr;
  float* d_expected_rows = nullptr;
  float* d_expected_cols = nullptr;
  float* d_row_scales = nullptr;
  float* d_col_scales = nullptr;
  int* d_bad_tiles = nullptr; // 检测到异常的 Tile 数
  int* d_corrected_tiles = nullptr; // 完成纠正的 Tile 数
  CUDA_CHECK(cudaMalloc(&d_a, a_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_b, b_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_expected_rows, tile_count * kTile * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_expected_cols, tile_count * kTile * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_row_scales, tile_count * kTile * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_col_scales, tile_count * kTile * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bad_tiles, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_corrected_tiles, sizeof(int)));

  constexpr int threads = 256;
  generate_fp8_kernel<<<static_cast<int>((a_count + threads - 1) / threads), threads>>>(d_a, a_count, 0x12345678U);
  generate_fp8_kernel<<<static_cast<int>((b_count + threads - 1) / threads), threads>>>(d_b, b_count, 0x9abcdef0U);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 block(kTile, kTile);
  dim3 prepare_block(threads);
  dim3 grid(tile_cols, tile_rows);
  cudaStream_t stream;
  cudaEvent_t metadata_start, metadata_stop, gemm_start, gemm_stop, total_start, total_stop;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&metadata_start));
  CUDA_CHECK(cudaEventCreate(&metadata_stop));
  CUDA_CHECK(cudaEventCreate(&gemm_start));
  CUDA_CHECK(cudaEventCreate(&gemm_stop));
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));

  CUDA_CHECK(cudaEventRecord(metadata_start, stream));
  prepare_expected_checksum_kernel<<<tile_count, prepare_block, 0, stream>>>(
      d_a, d_b, d_expected_rows, d_expected_cols, d_row_scales, d_col_scales, options.m, options.n, options.k, tile_rows, tile_cols);
  CUDA_CHECK(cudaEventRecord(metadata_stop, stream));
  CUDA_CHECK(cudaEventSynchronize(metadata_stop));
  float metadata_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&metadata_ms, metadata_start, metadata_stop));

  auto enqueue_gemm = [&]() { native_tiled_fp8_gemm_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, options.m, options.n, options.k); };
  auto enqueue_abft = [&]() {
    tile_abft_kernel<<<tile_count, block, 0, stream>>>(
        d_c, d_expected_rows, d_expected_cols, d_row_scales, d_col_scales, options.m, options.n, tile_rows, tile_cols, options.abs_tolerance,
        options.rel_tolerance, options.inject_fault, d_bad_tiles, d_corrected_tiles);
  };

  for (int i = 0; i < options.warmup; ++i) {
    CUDA_CHECK(cudaMemsetAsync(d_bad_tiles, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_corrected_tiles, 0, sizeof(int), stream));
    enqueue_gemm();
    enqueue_abft();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float total_ms = 0.0f;
  float gemm_ms = 0.0f;
  float abft_ms = 0.0f;
  for (int i = 0; i < options.repeat; ++i) {
    CUDA_CHECK(cudaMemsetAsync(d_bad_tiles, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_corrected_tiles, 0, sizeof(int), stream));
    CUDA_CHECK(cudaEventRecord(total_start, stream));
    CUDA_CHECK(cudaEventRecord(gemm_start, stream));
    enqueue_gemm();
    CUDA_CHECK(cudaEventRecord(gemm_stop, stream));
    enqueue_abft();
    CUDA_CHECK(cudaEventRecord(total_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(total_stop));

    float iteration_gemm = 0.0f;
    float iteration_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&iteration_gemm, gemm_start, gemm_stop));
    CUDA_CHECK(cudaEventElapsedTime(&iteration_total, total_start, total_stop));
    total_ms += iteration_total;
    gemm_ms += iteration_gemm;
    abft_ms += iteration_total - iteration_gemm;
  }

  std::vector<Element> h_a(a_count);
  std::vector<Element> h_b(b_count);
  std::vector<float> h_c(c_count);
  CUDA_CHECK(cudaMemcpy(h_a.data(), d_a, a_count * sizeof(Element), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_b.data(), d_b, b_count * sizeof(Element), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, c_count * sizeof(float), cudaMemcpyDeviceToHost));
  int bad_tiles = 0;
  int corrected_tiles = 0;
  CUDA_CHECK(cudaMemcpy(&bad_tiles, d_bad_tiles, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&corrected_tiles, d_corrected_tiles, sizeof(int), cudaMemcpyDeviceToHost));

  float average_total_ms = total_ms / options.repeat;
  float average_gemm_ms = gemm_ms / options.repeat;
  float average_abft_ms = abft_ms / options.repeat;
  double tflops = (2.0 * static_cast<double>(options.m) * options.n * options.k) / (average_total_ms * 1.0e9);
  float error = verify_samples(h_a, h_b, h_c, options);
  bool verification = options.inject_fault ? bad_tiles == 1 && corrected_tiles == 1 : bad_tiles == 0 && error < 1.0e-3f;

  std::printf("version: S0 native tiled FP8 GEMM + tile-local ABFT\n");
  std::printf("tile: %dx%d, shape: M=%d N=%d K=%d\n", kTile, kTile, options.m, options.n, options.k);
  std::printf("avg_gemm_time_ms: %.6f\n", average_gemm_ms);
  std::printf("avg_abft_time_ms: %.6f\n", average_abft_ms);
  std::printf("expected_prepare_time_ms: %.6f\n", metadata_ms);
  std::printf("avg_steady_state_end_to_end_time_ms: %.6f\n", average_total_ms);
  std::printf("steady_state_tflops: %.6f\n", tflops);
  std::printf("first_call_approx_time_ms: %.6f\n", metadata_ms + average_total_ms);
  std::printf("sampled_max_error: %.6e\n", error);
  std::printf("bad_tiles: %d\n", bad_tiles);
  std::printf("corrected_tiles: %d\n", corrected_tiles);
  std::printf("verification: %s\n", verification ? "PASS" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(metadata_start));
  CUDA_CHECK(cudaEventDestroy(metadata_stop));
  CUDA_CHECK(cudaEventDestroy(gemm_start));
  CUDA_CHECK(cudaEventDestroy(gemm_stop));
  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_expected_rows));
  CUDA_CHECK(cudaFree(d_expected_cols));
  CUDA_CHECK(cudaFree(d_row_scales));
  CUDA_CHECK(cudaFree(d_col_scales));
  CUDA_CHECK(cudaFree(d_bad_tiles));
  CUDA_CHECK(cudaFree(d_corrected_tiles));
  return verification ? 0 : 1;
}
