#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(error));                              \
      std::exit(EXIT_FAILURE);                                              \
    }                                                                      \
  } while (0)

#define CUBLAS_CHECK(call)                                                   \
  do {                                                                       \
    cublasStatus_t status = (call);                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
      std::fprintf(stderr, "cuBLAS error %s:%d: %s\n", __FILE__, __LINE__,  \
                   cublasGetStatusString(status));                            \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                        \
  } while (0)

struct Options {
  int device = 1;
  int m = 4096;
  int n = 4096;
  int k = 4096;
  int warmup = 20;
  int repeat = 50;
  int autotune_repeat = 5;
  int verify_samples = 8;
  int workspace_mb = 256;
  double peak_tflops = 296.0;
  bool fast_accum = false;
};

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

static double get_double(
    int argc, char** argv, const char* name, double value) {
  size_t length = std::strlen(name);
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
      return std::strtod(argv[i + 1], nullptr);
    }
    if (std::strncmp(argv[i], name, length) == 0 &&
        argv[i][length] == '=') {
      return std::strtod(argv[i] + length + 1, nullptr);
    }
  }
  return value;
}

static Options parse_options(int argc, char** argv) {
  Options options;
  options.device = get_int(argc, argv, "--device", options.device);
  options.m = get_int(argc, argv, "--m", options.m);
  options.n = get_int(argc, argv, "--n", options.n);
  options.k = get_int(argc, argv, "--k", options.k);
  options.warmup = get_int(argc, argv, "--warmup", options.warmup);
  options.repeat = get_int(argc, argv, "--repeat", options.repeat);
  options.autotune_repeat =
      get_int(argc, argv, "--autotune-repeat", options.autotune_repeat);
  options.verify_samples =
      get_int(argc, argv, "--verify-samples", options.verify_samples);
  options.workspace_mb =
      get_int(argc, argv, "--workspace-mb", options.workspace_mb);
  options.peak_tflops =
      get_double(argc, argv, "--peak-tflops", options.peak_tflops);

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--fast-accum") == 0) {
      options.fast_accum = true;
    }
    if (std::strcmp(argv[i], "--help") == 0) {
      std::printf(
          "Usage: %s [--device N --m N --n N --k N]\n"
          "       [--warmup N --repeat N --autotune-repeat N]\n"
          "       [--workspace-mb N --verify-samples N]\n"
          "       [--peak-tflops X --fast-accum]\n",
          argv[0]);
      std::exit(EXIT_SUCCESS);
    }
  }

  if (options.device < 0 || options.m <= 0 || options.n <= 0 ||
      options.k <= 0 || options.warmup < 0 || options.repeat <= 0 ||
      options.autotune_repeat <= 0 || options.verify_samples < 0 ||
      options.workspace_mb < 0 || options.peak_tflops <= 0.0) {
    std::fprintf(stderr, "Invalid benchmark options.\n");
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

__global__ void generate_fp8_kernel(
    __nv_fp8_e4m3* output, size_t count, uint32_t seed) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  uint32_t bits = mix_u32(seed + static_cast<uint32_t>(index));
  float unit = static_cast<float>(bits & 0x00ffffffU) / 16777216.0f;
  output[index] = __nv_fp8_e4m3(2.0f * unit - 1.0f);
}

struct VerificationResult {
  float max_absolute_error = 0.0f;
  float max_tolerance_ratio = 0.0f;
  int checked = 0;
};

static VerificationResult verify_samples(
    const std::vector<__nv_fp8_e4m3>& a,
    const std::vector<__nv_fp8_e4m3>& b,
    const std::vector<float>& c,
    const Options& options) {
  VerificationResult result;
  if (options.verify_samples == 0) {
    return result;
  }
  int row_step = std::max(1, options.m / options.verify_samples);
  int col_step = std::max(1, options.n / options.verify_samples);
  int checked_rows = 0;
  for (int row = 0;
       row < options.m && checked_rows < options.verify_samples;
       row += row_step, ++checked_rows) {
    int checked_cols = 0;
    for (int col = 0;
         col < options.n && checked_cols < options.verify_samples;
         col += col_step, ++checked_cols) {
      double reference = 0.0;
      for (int kk = 0; kk < options.k; ++kk) {
        reference +=
            static_cast<double>(static_cast<float>(
                a[static_cast<size_t>(row) * options.k + kk])) *
            static_cast<double>(static_cast<float>(
                b[kk + static_cast<size_t>(col) * options.k]));
      }
      float actual = c[row + static_cast<size_t>(col) * options.m];
      float absolute_error =
          static_cast<float>(std::fabs(actual - reference));
      float tolerance =
          0.5f +
          2.0e-2f *
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
  CUDA_CHECK(cudaSetDevice(options.device));

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, options.device));

  size_t a_count = static_cast<size_t>(options.m) * options.k;
  size_t b_count = static_cast<size_t>(options.k) * options.n;
  size_t c_count = static_cast<size_t>(options.m) * options.n;
  __nv_fp8_e4m3* d_a = nullptr;
  __nv_fp8_e4m3* d_b = nullptr;
  float* d_c = nullptr;
  float* d_scales = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes =
      static_cast<size_t>(options.workspace_mb) * 1024 * 1024;

  CUDA_CHECK(cudaMalloc(&d_a, a_count * sizeof(*d_a)));
  CUDA_CHECK(cudaMalloc(&d_b, b_count * sizeof(*d_b)));
  CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(*d_c)));
  CUDA_CHECK(cudaMalloc(&d_scales, 2 * sizeof(float)));
  if (workspace_bytes > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
  }

  constexpr int threads = 256;
  generate_fp8_kernel<<<
      static_cast<int>((a_count + threads - 1) / threads), threads>>>(
      d_a, a_count, 0x12345678U);
  generate_fp8_kernel<<<
      static_cast<int>((b_count + threads - 1) / threads), threads>>>(
      d_b, b_count, 0x9abcdef0U);
  const float host_scales[2] = {1.0f, 1.0f};
  CUDA_CHECK(cudaMemcpy(
      d_scales, host_scales, sizeof(host_scales), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(cublasLtCreate(&handle));
  CUBLAS_CHECK(cublasLtMatmulDescCreate(
      &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  const void* a_scale = d_scales;
  const void* b_scale = d_scales + 1;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &a_scale, sizeof(a_scale)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &b_scale, sizeof(b_scale)));
  if (options.fast_accum) {
    int8_t fast_accum = 1;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
        &fast_accum, sizeof(fast_accum)));
  }

  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &a_layout, CUDA_R_8F_E4M3, options.k, options.m, options.k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &b_layout, CUDA_R_8F_E4M3, options.k, options.n, options.k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &c_layout, CUDA_R_32F, options.m, options.n, options.m));
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes, sizeof(workspace_bytes)));

  constexpr int max_algorithms = 32;
  std::vector<cublasLtMatmulHeuristicResult_t> heuristics(max_algorithms);
  int returned_results = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      handle, operation, a_layout, b_layout, c_layout, c_layout,
      preference, max_algorithms, heuristics.data(), &returned_results));
  if (returned_results == 0) {
    std::fprintf(stderr, "cuBLASLt returned no FP8 algorithm.\n");
    return EXIT_FAILURE;
  }

  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  const float alpha = 1.0f;
  const float beta = 0.0f;

  auto run_algorithm = [&](int algorithm_index) {
    return cublasLtMatmul(
        handle, operation, &alpha,
        d_a, a_layout, d_b, b_layout, &beta,
        d_c, c_layout, d_c, c_layout,
        &heuristics[algorithm_index].algo,
        workspace, workspace_bytes, stream);
  };

  int best_algorithm = -1;
  float best_ms = std::numeric_limits<float>::max();
  for (int algorithm = 0; algorithm < returned_results; ++algorithm) {
    if (heuristics[algorithm].state != CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    cublasStatus_t status = run_algorithm(algorithm);
    if (status != CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iteration = 0;
         iteration < options.autotune_repeat; ++iteration) {
      status = run_algorithm(algorithm);
      if (status != CUBLAS_STATUS_SUCCESS) {
        break;
      }
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float average_ms = elapsed_ms / options.autotune_repeat;
    if (average_ms < best_ms) {
      best_ms = average_ms;
      best_algorithm = algorithm;
    }
  }
  if (best_algorithm < 0) {
    std::fprintf(stderr, "All cuBLASLt FP8 algorithms failed.\n");
    return EXIT_FAILURE;
  }

  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    CUBLAS_CHECK(run_algorithm(best_algorithm));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int iteration = 0; iteration < options.repeat; ++iteration) {
    CUBLAS_CHECK(run_algorithm(best_algorithm));
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  float average_ms = total_ms / options.repeat;
  double operations =
      2.0 * static_cast<double>(options.m) * options.n * options.k;
  double tflops = operations / (average_ms * 1.0e9);

  std::vector<__nv_fp8_e4m3> h_a;
  std::vector<__nv_fp8_e4m3> h_b;
  std::vector<float> h_c;
  VerificationResult verification;
  if (options.verify_samples > 0) {
    h_a.resize(a_count);
    h_b.resize(b_count);
    h_c.resize(c_count);
    CUDA_CHECK(cudaMemcpy(
        h_a.data(), d_a, a_count * sizeof(*d_a), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        h_b.data(), d_b, b_count * sizeof(*d_b), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        h_c.data(), d_c, c_count * sizeof(*d_c), cudaMemcpyDeviceToHost));
    verification = verify_samples(h_a, h_b, h_c, options);
  }
  bool passed =
      options.verify_samples == 0 ||
      verification.max_tolerance_ratio <= 1.0f;

  int algorithm_id = -1;
  size_t algorithm_id_size = sizeof(algorithm_id);
  CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
      &heuristics[best_algorithm].algo, CUBLASLT_ALGO_CONFIG_ID,
      &algorithm_id, sizeof(algorithm_id), &algorithm_id_size));

  std::printf("benchmark: cuBLASLt FP8 E4M3 GEMM\n");
  std::printf("device: %d (%s)\n", options.device, properties.name);
  std::printf("shape: M=%d N=%d K=%d\n",
              options.m, options.n, options.k);
  std::printf("layout: column-major C=A*B\n");
  std::printf("types: A=FP8 E4M3 B=FP8 E4M3 "
              "accumulator=FP32 output=FP32\n");
  std::printf("fast_accum: %s\n", options.fast_accum ? "true" : "false");
  std::printf("heuristic_algorithms: %d\n", returned_results);
  std::printf("selected_algorithm_index: %d\n", best_algorithm);
  std::printf("selected_algorithm_id: %d\n", algorithm_id);
  std::printf("workspace_bytes: %zu\n", workspace_bytes);
  std::printf("average_time_ms: %.6f\n", average_ms);
  std::printf("throughput_tflops: %.6f\n", tflops);
  std::printf("configured_peak_tflops: %.3f\n", options.peak_tflops);
  std::printf("peak_utilization_percent: %.3f\n",
              100.0 * tflops / options.peak_tflops);
  std::printf("verification_samples: %d\n", verification.checked);
  std::printf("sampled_max_absolute_error: %.6e\n",
              verification.max_absolute_error);
  std::printf("sampled_max_tolerance_ratio: %.6e\n",
              verification.max_tolerance_ratio);
  std::printf("verification: %s\n", passed ? "PASS" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_layout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_layout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_layout));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation));
  CUBLAS_CHECK(cublasLtDestroy(handle));
  CUDA_CHECK(cudaFree(workspace));
  CUDA_CHECK(cudaFree(d_scales));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_a));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
