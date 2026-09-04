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
constexpr int kThreads = kTile * kTile;
constexpr int kDefaultBatchTiles = 128;

struct Options {
  int m = 256;
  int n = 256;
  int k = 256;
  int warmup = 2;
  int repeat = 10;
  int verify_samples = 8;
  int batch_tiles = kDefaultBatchTiles;
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
  options.batch_tiles = get_int(argc, argv, "--batch-tiles", options.batch_tiles);
  options.abs_tolerance = get_float(argc, argv, "--abs-tol", options.abs_tolerance);
  options.rel_tolerance = get_float(argc, argv, "--rel-tol", options.rel_tolerance);
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--inject-fault") == 0) {
      options.inject_fault = true;
    }
    if (std::strcmp(argv[i], "--help") == 0) {
      std::printf(
          "Usage: %s [--m N --n N --k N --warmup N --repeat N]\n"
          "       [--batch-tiles N --abs-tol X --rel-tol X]\n"
          "       [--verify-samples N --inject-fault]\n",
          argv[0]);
      std::exit(EXIT_SUCCESS);
    }
  }
  if (options.m <= 0 || options.n <= 0 || options.k <= 0 || options.warmup < 0 || options.repeat <= 0 || options.verify_samples < 0 ||
      options.batch_tiles <= 0 || options.abs_tolerance < 0.0f || options.rel_tolerance <= 0.0f) {
    std::fprintf(stderr, "Invalid shape, batch, tolerance, or iteration options.\n");
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

__global__ void native_tiled_fp8_gemm_kernel(const Element* a, const Element* b, float* c, int m, int n, int k, int tile_begin,
                                             int tile_count, int tile_cols) {
  __shared__ Element a_tile[kTile][kTile];
  __shared__ Element b_tile[kTile][kTile];

  int tile_id = tile_begin + blockIdx.x;
  if (blockIdx.x >= tile_count) {
    return;
  }
  int tile_row = tile_id / tile_cols;
  int tile_col = tile_id % tile_cols;
  int row = tile_row * kTile + threadIdx.y;
  int col = tile_col * kTile + threadIdx.x;
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

// Metadata is computed once because A and B remain unchanged across repeats.
__global__ void input_tile_sums_kernel(const Element* a, const Element* b, float* a_sums, float* b_sums, int m, int n, int k, int tile_rows,
                                       int tile_cols) {
  int tile_id = blockIdx.x;
  if (tile_id >= max(tile_rows, tile_cols)) {
    return;
  }
  if (tile_id < tile_rows) {
    for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
      float sum = 0.0f;
      for (int row = 0; row < kTile; ++row) {
        sum += static_cast<float>(a[(tile_id * kTile + row) * k + kk]);
      }
      a_sums[tile_id * k + kk] = sum;
    }
  }
  if (tile_id < tile_cols) {
    for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
      float sum = 0.0f;
      int valid_cols = min(kTile, n - tile_id * kTile);
      for (int col = 0; col < valid_cols; ++col) {
        sum += static_cast<float>(b[kk * n + tile_id * kTile + col]);
      }
      b_sums[tile_id * k + kk] = sum;
    }
  }
}

__device__ __forceinline__ bool checksum_bad(float actual, float expected, float abs_tolerance, float rel_tolerance) {
  float scale = fmaxf(1.0f, fabsf(expected));
  return fabsf(actual - expected) > abs_tolerance + rel_tolerance * scale;
}

// This stage waits for one GEMM batch, loads C once, and writes row/column
// residuals. Expected checksums use reusable A/B metadata.
__global__ void tile_checksum_kernel(const Element* a, const Element* b, const float* a_sums, const float* b_sums, float* c,
                                     float* row_deltas, float* col_deltas, int m, int n, int k, int tile_rows, int tile_cols,
                                     int tile_begin, int tile_count, float abs_tolerance, float rel_tolerance, bool inject_fault) {
  extern __shared__ float c_tile[];
  int tile_id = tile_begin + blockIdx.x;
  if (blockIdx.x >= tile_count || tile_id >= tile_rows * tile_cols) {
    return;
  }
  int tile_row = tile_id / tile_cols;
  int tile_col = tile_id % tile_cols;
  int valid_rows = min(kTile, m - tile_row * kTile);
  int valid_cols = min(kTile, n - tile_col * kTile);

  if (inject_fault && threadIdx.x == 0 && tile_id == (tile_rows * tile_cols) / 2) {
    int fault_row = tile_row * kTile + valid_rows / 2;
    int fault_col = tile_col * kTile + valid_cols / 2;
    if (fault_row < m && fault_col < n) {
      c[fault_row * n + fault_col] += 1.0f;
    }
  }
  __syncthreads();

  for (int index = threadIdx.x; index < kTile * kTile; index += kThreads) {
    int local_row = index / kTile;
    int local_col = index % kTile;
    int global_row = tile_row * kTile + local_row;
    int global_col = tile_col * kTile + local_col;
    c_tile[index] = global_row < m && global_col < n ? c[global_row * n + global_col] : 0.0f;
  }
  __syncthreads();

  if (threadIdx.x < valid_rows) {
    int global_row = tile_row * kTile + threadIdx.x;
    float actual = 0.0f;
    float expected = 0.0f;
    for (int col = 0; col < valid_cols; ++col) {
      actual += c_tile[threadIdx.x * kTile + col];
    }
    for (int kk = 0; kk < k; ++kk) {
      expected += static_cast<float>(a[global_row * k + kk]) * b_sums[tile_col * k + kk];
    }
    row_deltas[tile_id * kTile + threadIdx.x] = expected - actual;
  }

  if (threadIdx.x < valid_cols) {
    int global_col = tile_col * kTile + threadIdx.x;
    float actual = 0.0f;
    float expected = 0.0f;
    for (int row = 0; row < valid_rows; ++row) {
      actual += c_tile[row * kTile + threadIdx.x];
    }
    for (int kk = 0; kk < k; ++kk) {
      expected += a_sums[tile_row * k + kk] * static_cast<float>(b[kk * n + global_col]);
    }
    col_deltas[tile_id * kTile + threadIdx.x] = expected - actual;
  }
}

__global__ void tile_abft_kernel(float* c, const float* row_deltas, const float* col_deltas, int m, int n, int tile_rows, int tile_cols,
                                 int tile_begin, int tile_count, float abs_tolerance, float rel_tolerance, int* bad_tiles,
                                 int* corrected_tiles) {
  __shared__ int bad_row_count;
  __shared__ int bad_col_count;
  __shared__ int bad_row;
  __shared__ int bad_col;
  int tile_id = tile_begin + blockIdx.x;
  if (blockIdx.x >= tile_count || tile_id >= tile_rows * tile_cols) {
    return;
  }
  int tile_row = tile_id / tile_cols;
  int tile_col = tile_id % tile_cols;
  int valid_rows = min(kTile, m - tile_row * kTile);
  int valid_cols = min(kTile, n - tile_col * kTile);
  if (threadIdx.x == 0) {
    bad_row_count = 0;
    bad_col_count = 0;
    bad_row = -1;
    bad_col = -1;
  }
  __syncthreads();

  if (threadIdx.x < valid_rows) {
    float delta = row_deltas[tile_id * kTile + threadIdx.x];
    if (checksum_bad(0.0f, delta, abs_tolerance, rel_tolerance)) {
      atomicAdd(&bad_row_count, 1);
      atomicCAS(&bad_row, -1, threadIdx.x);
    }
  }
  if (threadIdx.x < valid_cols) {
    float delta = col_deltas[tile_id * kTile + threadIdx.x];
    if (checksum_bad(0.0f, delta, abs_tolerance, rel_tolerance)) {
      atomicAdd(&bad_col_count, 1);
      atomicCAS(&bad_col, -1, threadIdx.x);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && bad_row_count > 0 && bad_col_count > 0) {
    atomicAdd(bad_tiles, 1);
  }
  if (threadIdx.x == 0 && bad_row_count == 1 && bad_col_count == 1) {
    int row = tile_row * kTile + bad_row;
    int col = tile_col * kTile + bad_col;
    c[row * n + col] += row_deltas[tile_id * kTile + bad_row] * 0.5f + col_deltas[tile_id * kTile + bad_col] * 0.5f;
    atomicAdd(corrected_tiles, 1);
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
  float* d_a_sums = nullptr;
  float* d_b_sums = nullptr;
  float* d_row_deltas = nullptr;
  float* d_col_deltas = nullptr;
  int* d_bad_tiles = nullptr;
  int* d_corrected_tiles = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_b, b_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_a_sums, tile_rows * options.k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b_sums, tile_cols * options.k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_row_deltas, tile_count * kTile * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_col_deltas, tile_count * kTile * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bad_tiles, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_corrected_tiles, sizeof(int)));

  constexpr int threads = 256;
  generate_fp8_kernel<<<static_cast<int>((a_count + threads - 1) / threads), threads>>>(d_a, a_count, 0x12345678U);
  generate_fp8_kernel<<<static_cast<int>((b_count + threads - 1) / threads), threads>>>(d_b, b_count, 0x9abcdef0U);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaStream_t gemm_stream;
  cudaStream_t checksum_stream;
  cudaStream_t abft_stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&gemm_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&checksum_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&abft_stream, cudaStreamNonBlocking));

  cudaEvent_t metadata_done;
  CUDA_CHECK(cudaEventCreate(&metadata_done));
  input_tile_sums_kernel<<<std::max(tile_rows, tile_cols), threads, 0, checksum_stream>>>(d_a, d_b, d_a_sums, d_b_sums, options.m,
                                                                                          options.n, options.k, tile_rows, tile_cols);
  CUDA_CHECK(cudaEventRecord(metadata_done, checksum_stream));
  CUDA_CHECK(cudaEventSynchronize(metadata_done));

  int batch_count = (tile_count + options.batch_tiles - 1) / options.batch_tiles;
  std::vector<cudaEvent_t> gemm_start(batch_count);
  std::vector<cudaEvent_t> gemm_done(batch_count);
  std::vector<cudaEvent_t> checksum_start(batch_count);
  std::vector<cudaEvent_t> checksum_done(batch_count);
  std::vector<cudaEvent_t> abft_start(batch_count);
  std::vector<cudaEvent_t> abft_done(batch_count);
  for (int batch = 0; batch < batch_count; ++batch) {
    CUDA_CHECK(cudaEventCreate(&gemm_start[batch]));
    CUDA_CHECK(cudaEventCreate(&gemm_done[batch]));
    CUDA_CHECK(cudaEventCreate(&checksum_start[batch]));
    CUDA_CHECK(cudaEventCreate(&checksum_done[batch]));
    CUDA_CHECK(cudaEventCreate(&abft_start[batch]));
    CUDA_CHECK(cudaEventCreate(&abft_done[batch]));
  }
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  dim3 gemm_block(kTile, kTile);

  auto enqueue_iteration = [&]() {
    for (int batch = 0; batch < batch_count; ++batch) {
      int tile_begin = batch * options.batch_tiles;
      int tiles_this_batch = std::min(options.batch_tiles, tile_count - tile_begin);
      CUDA_CHECK(cudaEventRecord(gemm_start[batch], gemm_stream));
      native_tiled_fp8_gemm_kernel<<<tiles_this_batch, gemm_block, 0, gemm_stream>>>(d_a, d_b, d_c, options.m, options.n, options.k,
                                                                                     tile_begin, tiles_this_batch, tile_cols);
      CUDA_CHECK(cudaEventRecord(gemm_done[batch], gemm_stream));

      CUDA_CHECK(cudaStreamWaitEvent(checksum_stream, gemm_done[batch], 0));
      CUDA_CHECK(cudaEventRecord(checksum_start[batch], checksum_stream));
      tile_checksum_kernel<<<tiles_this_batch, kThreads, kTile * kTile * sizeof(float), checksum_stream>>>(
          d_a, d_b, d_a_sums, d_b_sums, d_c, d_row_deltas, d_col_deltas, options.m, options.n, options.k, tile_rows, tile_cols, tile_begin,
          tiles_this_batch, options.abs_tolerance, options.rel_tolerance, options.inject_fault);
      CUDA_CHECK(cudaEventRecord(checksum_done[batch], checksum_stream));

      CUDA_CHECK(cudaStreamWaitEvent(abft_stream, checksum_done[batch], 0));
      if (batch == 0) {
        CUDA_CHECK(cudaMemsetAsync(d_bad_tiles, 0, sizeof(int), abft_stream));
        CUDA_CHECK(cudaMemsetAsync(d_corrected_tiles, 0, sizeof(int), abft_stream));
      }
      CUDA_CHECK(cudaEventRecord(abft_start[batch], abft_stream));
      tile_abft_kernel<<<tiles_this_batch, kThreads, 0, abft_stream>>>(d_c, d_row_deltas, d_col_deltas, options.m, options.n, tile_rows,
                                                                       tile_cols, tile_begin, tiles_this_batch, options.abs_tolerance,
                                                                       options.rel_tolerance, d_bad_tiles, d_corrected_tiles);
      CUDA_CHECK(cudaEventRecord(abft_done[batch], abft_stream));
    }
  };

  for (int i = 0; i < options.warmup; ++i) {
    enqueue_iteration();
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  float total_ms = 0.0f;
  float gemm_ms = 0.0f;
  float checksum_ms = 0.0f;
  float abft_ms = 0.0f;
  for (int i = 0; i < options.repeat; ++i) {
    CUDA_CHECK(cudaMemsetAsync(d_bad_tiles, 0, sizeof(int), abft_stream));
    CUDA_CHECK(cudaMemsetAsync(d_corrected_tiles, 0, sizeof(int), abft_stream));
    CUDA_CHECK(cudaEventRecord(start, gemm_stream));
    enqueue_iteration();
    CUDA_CHECK(cudaEventRecord(stop, abft_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float iteration_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&iteration_total, start, stop));
    total_ms += iteration_total;
    for (int batch = 0; batch < batch_count; ++batch) {
      float value = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&value, gemm_start[batch], gemm_done[batch]));
      gemm_ms += value;
      CUDA_CHECK(cudaEventElapsedTime(&value, checksum_start[batch], checksum_done[batch]));
      checksum_ms += value;
      CUDA_CHECK(cudaEventElapsedTime(&value, abft_start[batch], abft_done[batch]));
      abft_ms += value;
    }
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
  float average_checksum_ms = checksum_ms / options.repeat;
  float average_abft_ms = abft_ms / options.repeat;
  double tflops = (2.0 * static_cast<double>(options.m) * options.n * options.k) / (average_total_ms * 1.0e9);
  float error = verify_samples(h_a, h_b, h_c, options);
  bool verification = options.inject_fault ? bad_tiles == 1 && corrected_tiles == 1 : bad_tiles == 0 && error < 1.0e-3f;

  std::printf("version: S1 native tiled FP8 GEMM + batched multi-stream ABFT\n");
  std::printf("tile: %dx%d, batch_tiles=%d, batch_count=%d\n", kTile, kTile, options.batch_tiles, batch_count);
  std::printf("shape: M=%d N=%d K=%d\n", options.m, options.n, options.k);
  std::printf("avg_overlapped_time_ms: %.6f\n", average_total_ms);
  std::printf("sum_gemm_batch_time_ms: %.6f\n", average_gemm_ms);
  std::printf("sum_checksum_batch_time_ms: %.6f\n", average_checksum_ms);
  std::printf("sum_abft_batch_time_ms: %.6f\n", average_abft_ms);
  std::printf("overlapped_tflops: %.6f\n", tflops);
  std::printf("sampled_max_error: %.6e\n", error);
  std::printf("bad_tiles: %d\n", bad_tiles);
  std::printf("corrected_tiles: %d\n", corrected_tiles);
  std::printf("verification: %s\n", verification ? "PASS" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
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
  CUDA_CHECK(cudaFree(d_a_sums));
  CUDA_CHECK(cudaFree(d_b_sums));
  CUDA_CHECK(cudaFree(d_row_deltas));
  CUDA_CHECK(cudaFree(d_col_deltas));
  CUDA_CHECK(cudaFree(d_bad_tiles));
  CUDA_CHECK(cudaFree(d_corrected_tiles));
  return verification ? 0 : 1;
}
