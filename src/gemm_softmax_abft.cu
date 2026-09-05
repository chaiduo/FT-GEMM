#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
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
constexpr int kReductionThreads = 256;

struct Options {
  int device = 0;
  int m = 64;
  int n = 64;
  int k = 64;
  int fault_row = -1;
  int fault_col = -1;
  float fault_value = 1.0f;
  double abs_tolerance = 2.0e-3;
  double rel_tolerance = 2.0e-5;
  double row_sum_tolerance = 2.0e-6;
  double location_tolerance = 2.5e-1;
  bool inject_fault = true;
};

struct HostChecks {
  std::vector<double> expected_sum;
  std::vector<double> expected_weighted;
  std::vector<double> actual_sum;
  std::vector<double> actual_weighted;
  std::vector<double> probability_sum;
  std::vector<double> log_z;
  std::vector<double> correction_delta;
  std::vector<int> underflow_count;
  std::vector<int> detected;
  std::vector<int> estimated_col;

  explicit HostChecks(int rows)
      : expected_sum(rows),
        expected_weighted(rows),
        actual_sum(rows),
        actual_weighted(rows),
        probability_sum(rows),
        log_z(rows),
        correction_delta(rows),
        underflow_count(rows),
        detected(rows),
        estimated_col(rows) {}
};

static int div_up(int value, int divisor) { return (value + divisor - 1) / divisor; }

static int get_int(int argc, char** argv, const char* name, int value) {
  size_t length = std::strlen(name);
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], name) == 0 && index + 1 < argc) {
      return std::atoi(argv[index + 1]);
    }
    if (std::strncmp(argv[index], name, length) == 0 && argv[index][length] == '=') {
      return std::atoi(argv[index] + length + 1);
    }
  }
  return value;
}

static double get_double(int argc, char** argv, const char* name, double value) {
  size_t length = std::strlen(name);
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], name) == 0 && index + 1 < argc) {
      return std::strtod(argv[index + 1], nullptr);
    }
    if (std::strncmp(argv[index], name, length) == 0 && argv[index][length] == '=') {
      return std::strtod(argv[index] + length + 1, nullptr);
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
  options.fault_row = get_int(argc, argv, "--fault-row", options.fault_row);
  options.fault_col = get_int(argc, argv, "--fault-col", options.fault_col);
  options.fault_value = static_cast<float>(get_double(argc, argv, "--fault-value", options.fault_value));
  options.abs_tolerance = get_double(argc, argv, "--abs-tol", options.abs_tolerance);
  options.rel_tolerance = get_double(argc, argv, "--rel-tol", options.rel_tolerance);
  options.row_sum_tolerance = get_double(argc, argv, "--row-sum-tol", options.row_sum_tolerance);
  options.location_tolerance = get_double(argc, argv, "--location-tol", options.location_tolerance);

  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--no-fault") == 0) {
      options.inject_fault = false;
    }
    if (std::strcmp(argv[index], "--help") == 0) {
      std::printf(
          "Usage: %s [--device N --m N --n N --k N]\n"
          "       [--fault-row N --fault-col N --fault-value X]\n"
          "       [--abs-tol X --rel-tol X --row-sum-tol X]\n"
          "       [--location-tol X --no-fault]\n",
          argv[0]);
      std::exit(EXIT_SUCCESS);
    }
  }

  if (options.m <= 0 || options.n <= 1 || options.k <= 0 || options.fault_value == 0.0f || options.abs_tolerance < 0.0 ||
      options.rel_tolerance < 0.0 || options.row_sum_tolerance < 0.0 || options.location_tolerance < 0.0) {
    std::fprintf(stderr, "Invalid shape, fault, or tolerance option.\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.fault_row < 0) {
    options.fault_row = options.m / 2;
  }
  if (options.fault_col < 0) {
    options.fault_col = options.n / 2;
  }
  if (options.fault_row >= options.m || options.fault_col >= options.n) {
    std::fprintf(stderr, "Fault position must satisfy row < M and col < N.\n");
    std::exit(EXIT_FAILURE);
  }
  return options;
}

__host__ __device__ static double position_weight(int column, int columns) {
  // Unique, zero-sum integer weights: sum_j (2*j - (N-1)) == 0.
  return static_cast<double>(2 * column - (columns - 1));
}

__device__ __forceinline__ uint32_t mix_u32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

__global__ void generate_exact_fp8_kernel(Element* output, size_t count, uint32_t seed) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  int quantized = static_cast<int>(mix_u32(seed + static_cast<uint32_t>(index)) % 7U) - 3;
  output[index] = Element(static_cast<float>(quantized) * 0.0625f);
}

__global__ void fp8_qk_gemm_kernel(const Element* a, const Element* b, float* logits, int m, int n, int k, float scale) {
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || column >= n) {
    return;
  }

  float accumulator = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    accumulator += static_cast<float>(a[row * k + kk]) * static_cast<float>(b[column * k + kk]);
  }
  logits[row * n + column] = accumulator * scale;
}

template <int Threads>
__device__ double block_sum(double value, double* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = Threads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

template <int Threads>
__device__ double block_max(double value, double* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = Threads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] = fmax(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  return scratch[0];
}

__global__ void project_b_kernel(const Element* b, double* b_sum, double* b_weighted, int n, int k) {
  __shared__ double scratch[kReductionThreads];
  int kk = blockIdx.x;
  double local_sum = 0.0;
  double local_weighted = 0.0;

  for (int column = threadIdx.x; column < n; column += blockDim.x) {
    double value = static_cast<double>(static_cast<float>(b[column * k + kk]));
    local_sum += value;
    local_weighted += position_weight(column, n) * value;
  }

  double sum = block_sum<kReductionThreads>(local_sum, scratch);
  double weighted = block_sum<kReductionThreads>(local_weighted, scratch);
  if (threadIdx.x == 0) {
    b_sum[kk] = sum;
    b_weighted[kk] = weighted;
  }
}

__global__ void expected_logit_projection_kernel(const Element* a, const double* b_sum, const double* b_weighted,
                                                 double* expected_sum, double* expected_weighted, int m, int k,
                                                 double scale) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m) {
    return;
  }
  double sum = 0.0;
  double weighted = 0.0;
  for (int kk = 0; kk < k; ++kk) {
    double value = static_cast<double>(static_cast<float>(a[row * k + kk]));
    sum += value * b_sum[kk];
    weighted += value * b_weighted[kk];
  }
  expected_sum[row] = sum * scale;
  expected_weighted[row] = weighted * scale;
}

__global__ void softmax_with_log_checks_kernel(const float* logits, float* probabilities, double* actual_sum,
                                               double* actual_weighted, double* probability_sum, double* log_z,
                                               int* underflow_count, int m, int n) {
  __shared__ double scratch[kReductionThreads];
  int row = blockIdx.x;
  if (row >= m) {
    return;
  }

  double local_max = -DBL_MAX;
  for (int column = threadIdx.x; column < n; column += blockDim.x) {
    local_max = fmax(local_max, static_cast<double>(logits[row * n + column]));
  }
  double row_max = block_max<kReductionThreads>(local_max, scratch);

  double local_exp_sum = 0.0;
  for (int column = threadIdx.x; column < n; column += blockDim.x) {
    local_exp_sum += exp(static_cast<double>(logits[row * n + column]) - row_max);
  }
  double denominator = block_sum<kReductionThreads>(local_exp_sum, scratch);
  double row_log_z = row_max + log(denominator);

  double local_log_sum = 0.0;
  double local_weighted = 0.0;
  double local_probability_sum = 0.0;
  int local_underflow = 0;
  for (int column = threadIdx.x; column < n; column += blockDim.x) {
    double log_probability = static_cast<double>(logits[row * n + column]) - row_log_z;
    float probability = static_cast<float>(exp(log_probability));
    probabilities[row * n + column] = probability;
    local_probability_sum += static_cast<double>(probability);
    local_log_sum += log_probability;
    local_weighted += position_weight(column, n) * log_probability;
    if (probability == 0.0f) {
      ++local_underflow;
    }
  }

  double row_log_sum = block_sum<kReductionThreads>(local_log_sum, scratch);
  double row_weighted = block_sum<kReductionThreads>(local_weighted, scratch);
  double row_probability_sum = block_sum<kReductionThreads>(local_probability_sum, scratch);
  double row_underflow = block_sum<kReductionThreads>(static_cast<double>(local_underflow), scratch);

  if (threadIdx.x == 0) {
    // sum(log(P)) + N*logZ reconstructs the unweighted logit projection.
    actual_sum[row] = row_log_sum + static_cast<double>(n) * row_log_z;
    // The position weights sum to zero, so the shared -logZ term cancels.
    actual_weighted[row] = row_weighted;
    probability_sum[row] = row_probability_sum;
    log_z[row] = row_log_z;
    underflow_count[row] = static_cast<int>(row_underflow);
  }
}

__device__ bool exceeds_tolerance(double actual, double expected, double abs_tolerance, double rel_tolerance) {
  double threshold = abs_tolerance + rel_tolerance * fmax(1.0, fabs(expected));
  return fabs(actual - expected) > threshold;
}

__global__ void analyze_rows_kernel(const double* expected_sum, const double* expected_weighted, const double* actual_sum,
                                    const double* actual_weighted, const double* probability_sum,
                                    const int* underflow_count, int* detected, int* estimated_col,
                                    double* correction_delta, int* detected_rows, int* located_rows, int m, int n,
                                    double abs_tolerance, double rel_tolerance, double row_sum_tolerance,
                                    double location_tolerance) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m) {
    return;
  }

  double delta_sum = actual_sum[row] - expected_sum[row];
  double delta_weighted = actual_weighted[row] - expected_weighted[row];
  bool bad = fabs(probability_sum[row] - 1.0) > row_sum_tolerance ||
             exceeds_tolerance(actual_sum[row], expected_sum[row], abs_tolerance, rel_tolerance) ||
             exceeds_tolerance(actual_weighted[row], expected_weighted[row], abs_tolerance, rel_tolerance);

  detected[row] = bad ? 1 : 0;
  estimated_col[row] = -1;
  correction_delta[row] = delta_sum;
  if (!bad) {
    return;
  }
  atomicAdd(detected_rows, 1);

  double minimum_delta = abs_tolerance + rel_tolerance * fmax(1.0, fabs(expected_sum[row]));
  if (fabs(delta_sum) <= minimum_delta) {
    return;
  }

  double observed_weight = delta_weighted / delta_sum;
  int column = __double2int_rn(0.5 * (observed_weight + static_cast<double>(n - 1)));
  if (column >= 0 && column < n &&
      fabs(observed_weight - position_weight(column, n)) <= location_tolerance) {
    estimated_col[row] = column;
    atomicAdd(located_rows, 1);
  }
}

__global__ void inject_logit_fault_kernel(float* logits, int n, int row, int column, float delta) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    logits[row * n + column] += delta;
  }
}

__global__ void correct_detected_logits_kernel(float* logits, const int* detected, const int* estimated_col,
                                               const double* correction_delta, int m, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && detected[row] != 0 && estimated_col[row] >= 0) {
    logits[row * n + estimated_col[row]] -= static_cast<float>(correction_delta[row]);
  }
}

static void copy_checks_to_host(HostChecks& host, const double* d_expected_sum, const double* d_expected_weighted,
                                const double* d_actual_sum, const double* d_actual_weighted,
                                const double* d_probability_sum, const double* d_log_z, const int* d_underflow_count,
                                const int* d_detected, const int* d_estimated_col, const double* d_correction_delta,
                                int rows) {
  CUDA_CHECK(cudaMemcpy(host.expected_sum.data(), d_expected_sum, rows * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host.expected_weighted.data(), d_expected_weighted, rows * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host.actual_sum.data(), d_actual_sum, rows * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host.actual_weighted.data(), d_actual_weighted, rows * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host.probability_sum.data(), d_probability_sum, rows * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host.log_z.data(), d_log_z, rows * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host.underflow_count.data(), d_underflow_count, rows * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host.detected.data(), d_detected, rows * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host.estimated_col.data(), d_estimated_col, rows * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host.correction_delta.data(), d_correction_delta, rows * sizeof(double), cudaMemcpyDeviceToHost));
}

static double max_projection_residual(const HostChecks& checks) {
  double result = 0.0;
  for (size_t row = 0; row < checks.expected_sum.size(); ++row) {
    result = std::max(result, std::fabs(checks.actual_sum[row] - checks.expected_sum[row]));
    result = std::max(result, std::fabs(checks.actual_weighted[row] - checks.expected_weighted[row]));
  }
  return result;
}

int main(int argc, char** argv) {
  Options options = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(options.device));

  size_t a_count = static_cast<size_t>(options.m) * options.k;
  size_t b_count = static_cast<size_t>(options.n) * options.k;
  size_t output_count = static_cast<size_t>(options.m) * options.n;
  float scale = 1.0f / std::sqrt(static_cast<float>(options.k));

  Element* d_a = nullptr;
  Element* d_b = nullptr;
  float* d_logits = nullptr;
  float* d_probabilities = nullptr;
  float* d_reference = nullptr;
  double* d_b_sum = nullptr;
  double* d_b_weighted = nullptr;
  double* d_expected_sum = nullptr;
  double* d_expected_weighted = nullptr;
  double* d_actual_sum = nullptr;
  double* d_actual_weighted = nullptr;
  double* d_probability_sum = nullptr;
  double* d_log_z = nullptr;
  double* d_correction_delta = nullptr;
  int* d_underflow_count = nullptr;
  int* d_detected = nullptr;
  int* d_estimated_col = nullptr;
  int* d_detected_rows = nullptr;
  int* d_located_rows = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, a_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_b, b_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_logits, output_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_probabilities, output_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_reference, output_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b_sum, options.k * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_b_weighted, options.k * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_expected_sum, options.m * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_expected_weighted, options.m * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_actual_sum, options.m * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_actual_weighted, options.m * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_probability_sum, options.m * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_log_z, options.m * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_correction_delta, options.m * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_underflow_count, options.m * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_detected, options.m * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_estimated_col, options.m * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_detected_rows, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_located_rows, sizeof(int)));

  int initialization_blocks = div_up(static_cast<int>(std::max(a_count, b_count)), 256);
  generate_exact_fp8_kernel<<<initialization_blocks, 256>>>(d_a, a_count, 17U);
  generate_exact_fp8_kernel<<<initialization_blocks, 256>>>(d_b, b_count, 29U);

  dim3 gemm_block(kTile, kTile);
  dim3 gemm_grid(div_up(options.n, kTile), div_up(options.m, kTile));
  fp8_qk_gemm_kernel<<<gemm_grid, gemm_block>>>(d_a, d_b, d_logits, options.m, options.n, options.k, scale);
  project_b_kernel<<<options.k, kReductionThreads>>>(d_b, d_b_sum, d_b_weighted, options.n, options.k);
  expected_logit_projection_kernel<<<div_up(options.m, 128), 128>>>(
      d_a, d_b_sum, d_b_weighted, d_expected_sum, d_expected_weighted, options.m, options.k, scale);

  auto run_softmax_and_analysis = [&]() {
    softmax_with_log_checks_kernel<<<options.m, kReductionThreads>>>(
        d_logits, d_probabilities, d_actual_sum, d_actual_weighted, d_probability_sum, d_log_z, d_underflow_count,
        options.m, options.n);
    CUDA_CHECK(cudaMemset(d_detected_rows, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_located_rows, 0, sizeof(int)));
    analyze_rows_kernel<<<div_up(options.m, 256), 256>>>(
        d_expected_sum, d_expected_weighted, d_actual_sum, d_actual_weighted, d_probability_sum, d_underflow_count,
        d_detected, d_estimated_col, d_correction_delta, d_detected_rows, d_located_rows, options.m, options.n,
        options.abs_tolerance, options.rel_tolerance, options.row_sum_tolerance, options.location_tolerance);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  };

  run_softmax_and_analysis();
  CUDA_CHECK(cudaMemcpy(d_reference, d_probabilities, output_count * sizeof(float), cudaMemcpyDeviceToDevice));

  int clean_detected = 0;
  int clean_located = 0;
  CUDA_CHECK(cudaMemcpy(&clean_detected, d_detected_rows, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&clean_located, d_located_rows, sizeof(int), cudaMemcpyDeviceToHost));
  HostChecks clean_checks(options.m);
  copy_checks_to_host(clean_checks, d_expected_sum, d_expected_weighted, d_actual_sum, d_actual_weighted,
                      d_probability_sum, d_log_z, d_underflow_count, d_detected, d_estimated_col,
                      d_correction_delta, options.m);

  bool pass = clean_detected == 0;
  std::printf("GEMM + Softmax cross-nonlinearity ABFT\n");
  std::printf("shape: M=%d N=%d K=%d, scale=1/sqrt(K), input=FP8 E4M3, output=FP32\n", options.m, options.n,
              options.k);
  std::printf("identity 0: sum(log(P)) + N*logZ == sum(logits)\n");
  std::printf("identity 1: sum(w*log(P)) == sum(w*logits), w_j=2*j-(N-1)\n");
  std::printf("clean run: detected_rows=%d located_rows=%d max_projection_residual=%.6e\n", clean_detected,
              clean_located, max_projection_residual(clean_checks));

  if (options.inject_fault) {
    inject_logit_fault_kernel<<<1, 1>>>(d_logits, options.n, options.fault_row, options.fault_col,
                                       options.fault_value);
    run_softmax_and_analysis();

    int detected_rows = 0;
    int located_rows = 0;
    CUDA_CHECK(cudaMemcpy(&detected_rows, d_detected_rows, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&located_rows, d_located_rows, sizeof(int), cudaMemcpyDeviceToHost));
    HostChecks fault_checks(options.m);
    copy_checks_to_host(fault_checks, d_expected_sum, d_expected_weighted, d_actual_sum, d_actual_weighted,
                        d_probability_sum, d_log_z, d_underflow_count, d_detected, d_estimated_col,
                        d_correction_delta, options.m);

    int estimated_col = fault_checks.estimated_col[options.fault_row];
    double estimated_delta = fault_checks.correction_delta[options.fault_row];
    std::printf("fault: row=%d col=%d injected_delta=%+.6f\n", options.fault_row, options.fault_col,
                options.fault_value);
    std::printf("detection: detected_rows=%d located_rows=%d estimated_col=%d estimated_delta=%+.6f\n",
                detected_rows, located_rows, estimated_col, estimated_delta);

    bool detected_target = fault_checks.detected[options.fault_row] != 0;
    bool located_target = estimated_col == options.fault_col;
    pass = pass && detected_rows == 1 && located_rows == 1 && detected_target && located_target;

    correct_detected_logits_kernel<<<div_up(options.m, 256), 256>>>(
        d_logits, d_detected, d_estimated_col, d_correction_delta, options.m, options.n);
    run_softmax_and_analysis();

    int post_detected = 0;
    CUDA_CHECK(cudaMemcpy(&post_detected, d_detected_rows, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<float> reference(output_count);
    std::vector<float> recovered(output_count);
    CUDA_CHECK(cudaMemcpy(reference.data(), d_reference, output_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(recovered.data(), d_probabilities, output_count * sizeof(float), cudaMemcpyDeviceToHost));
    double max_probability_error = 0.0;
    for (size_t index = 0; index < output_count; ++index) {
      max_probability_error =
          std::max(max_probability_error, std::fabs(static_cast<double>(reference[index] - recovered[index])));
    }
    std::printf("recovery: post_detected_rows=%d max_probability_error=%.6e\n", post_detected,
                max_probability_error);
    pass = pass && post_detected == 0 && max_probability_error <= 1.0e-6;
  }

  std::printf("verification: %s\n", pass ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(d_located_rows));
  CUDA_CHECK(cudaFree(d_detected_rows));
  CUDA_CHECK(cudaFree(d_estimated_col));
  CUDA_CHECK(cudaFree(d_detected));
  CUDA_CHECK(cudaFree(d_underflow_count));
  CUDA_CHECK(cudaFree(d_correction_delta));
  CUDA_CHECK(cudaFree(d_log_z));
  CUDA_CHECK(cudaFree(d_probability_sum));
  CUDA_CHECK(cudaFree(d_actual_weighted));
  CUDA_CHECK(cudaFree(d_actual_sum));
  CUDA_CHECK(cudaFree(d_expected_weighted));
  CUDA_CHECK(cudaFree(d_expected_sum));
  CUDA_CHECK(cudaFree(d_b_weighted));
  CUDA_CHECK(cudaFree(d_b_sum));
  CUDA_CHECK(cudaFree(d_reference));
  CUDA_CHECK(cudaFree(d_probabilities));
  CUDA_CHECK(cudaFree(d_logits));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_a));
  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
