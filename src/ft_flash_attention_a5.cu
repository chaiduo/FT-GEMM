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

constexpr int kThreads = 128;
constexpr int kKeyTile = 64;

enum FaultStage : int {
  kFaultNone = 0,
  kFaultScore = 1,
  kFaultSoftmax = 2,
  kFaultSoftmaxState = 3,
  kFaultOutput = 4,
};

struct DeviceStats {
  int score_detected;
  int score_corrected;
  int score_replayed;
  int softmax_detected;
  int softmax_corrected;
  int softmax_replayed;
  int pv_detected;
  int pv_corrected;
  int failed_rows;
};

struct Options {
  int batch = 1;
  int heads = 2;
  int sequence = 128;
  int head_dim = 64;
  int warmup = 2;
  int repeat = 10;
  int fault_batch = 0;
  int fault_head = 0;
  int fault_query = -1;
  int fault_index = -1;
  float fault_value = 1.0f;
  float score_abs_tolerance = 2.0e-3f;
  float score_rel_tolerance = 5.0e-4f;
  float softmax_abs_tolerance = 2.0e-5f;
  float softmax_rel_tolerance = 2.0e-5f;
  float pv_abs_tolerance = 2.0e-3f;
  float pv_rel_tolerance = 2.0e-3f;
  float verify_abs_tolerance = 3.0e-3f;
  float verify_rel_tolerance = 2.0e-2f;
  bool causal = false;
  FaultStage fault_stage = kFaultNone;
};

struct VerificationResult {
  float max_absolute_error = 0.0f;
  float max_tolerance_ratio = 0.0f;
  int mismatches = 0;
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

static const char* get_string(int argc, char** argv, const char* name, const char* value) {
  size_t length = std::strlen(name);
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
      return argv[i + 1];
    }
    if (std::strncmp(argv[i], name, length) == 0 && argv[i][length] == '=') {
      return argv[i] + length + 1;
    }
  }
  return value;
}

static FaultStage parse_fault_stage(const char* name) {
  if (std::strcmp(name, "none") == 0) {
    return kFaultNone;
  }
  if (std::strcmp(name, "score") == 0) {
    return kFaultScore;
  }
  if (std::strcmp(name, "softmax") == 0) {
    return kFaultSoftmax;
  }
  if (std::strcmp(name, "softmax-state") == 0) {
    return kFaultSoftmaxState;
  }
  if (std::strcmp(name, "output") == 0) {
    return kFaultOutput;
  }
  std::fprintf(stderr, "Unknown --fault-stage '%s'.\n", name);
  std::exit(EXIT_FAILURE);
}

static const char* fault_stage_name(FaultStage stage) {
  switch (stage) {
    case kFaultScore:
      return "score";
    case kFaultSoftmax:
      return "softmax";
    case kFaultSoftmaxState:
      return "softmax-state";
    case kFaultOutput:
      return "output";
    default:
      return "none";
  }
}

static Options parse_options(int argc, char** argv) {
  Options options;
  options.batch = get_int(argc, argv, "--batch", options.batch);
  options.heads = get_int(argc, argv, "--heads", options.heads);
  options.sequence = get_int(argc, argv, "--sequence", options.sequence);
  options.head_dim = get_int(argc, argv, "--head-dim", options.head_dim);
  options.warmup = get_int(argc, argv, "--warmup", options.warmup);
  options.repeat = get_int(argc, argv, "--repeat", options.repeat);
  options.fault_batch = get_int(argc, argv, "--fault-batch", options.fault_batch);
  options.fault_head = get_int(argc, argv, "--fault-head", options.fault_head);
  options.fault_query = get_int(argc, argv, "--fault-query", options.fault_query);
  options.fault_index = get_int(argc, argv, "--fault-index", options.fault_index);
  options.fault_value = get_float(argc, argv, "--fault-value", options.fault_value);
  options.score_abs_tolerance = get_float(argc, argv, "--score-abs-tol", options.score_abs_tolerance);
  options.score_rel_tolerance = get_float(argc, argv, "--score-rel-tol", options.score_rel_tolerance);
  options.softmax_abs_tolerance = get_float(argc, argv, "--softmax-abs-tol", options.softmax_abs_tolerance);
  options.softmax_rel_tolerance = get_float(argc, argv, "--softmax-rel-tol", options.softmax_rel_tolerance);
  options.pv_abs_tolerance = get_float(argc, argv, "--pv-abs-tol", options.pv_abs_tolerance);
  options.pv_rel_tolerance = get_float(argc, argv, "--pv-rel-tol", options.pv_rel_tolerance);
  options.verify_abs_tolerance = get_float(argc, argv, "--verify-abs-tol", options.verify_abs_tolerance);
  options.verify_rel_tolerance = get_float(argc, argv, "--verify-rel-tol", options.verify_rel_tolerance);
  options.fault_stage = parse_fault_stage(get_string(argc, argv, "--fault-stage", "none"));

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--causal") == 0) {
      options.causal = true;
    }
    if (std::strcmp(argv[i], "--help") == 0) {
      std::printf(
          "Usage: %s [--batch N --heads N --sequence N --head-dim 64|128]\n"
          "       [--warmup N --repeat N --causal]\n"
          "       [--fault-stage none|score|softmax|softmax-state|output]\n"
          "       [--fault-batch N --fault-head N --fault-query N]\n"
          "       [--fault-index N --fault-value X]\n"
          "       [--score-abs-tol X --score-rel-tol X]\n"
          "       [--softmax-abs-tol X --softmax-rel-tol X]\n"
          "       [--pv-abs-tol X --pv-rel-tol X]\n",
          argv[0]);
      std::exit(EXIT_SUCCESS);
    }
  }

  if (options.fault_query < 0) {
    options.fault_query = options.sequence / 2;
  }
  if (options.fault_index < 0) {
    options.fault_index =
        options.fault_stage == kFaultOutput ? options.head_dim / 2 : (options.causal ? options.fault_query / 2 : options.sequence / 2);
  }
  bool valid_fault_index = options.fault_stage == kFaultOutput
                               ? options.fault_index >= 0 && options.fault_index < options.head_dim
                               : options.fault_index >= 0 && options.fault_index < options.sequence;
  if (options.batch <= 0 || options.heads <= 0 || options.sequence <= 0 ||
      (options.head_dim != 64 && options.head_dim != 128) || options.warmup < 0 || options.repeat <= 0 ||
      options.fault_batch < 0 || options.fault_batch >= options.batch || options.fault_head < 0 ||
      options.fault_head >= options.heads || options.fault_query < 0 || options.fault_query >= options.sequence ||
      !valid_fault_index || options.fault_value == 0.0f || options.score_abs_tolerance < 0.0f ||
      options.score_rel_tolerance < 0.0f || options.softmax_abs_tolerance < 0.0f ||
      options.softmax_rel_tolerance < 0.0f || options.pv_abs_tolerance < 0.0f ||
      options.pv_rel_tolerance < 0.0f || options.verify_abs_tolerance < 0.0f ||
      options.verify_rel_tolerance < 0.0f) {
    std::fprintf(stderr, "Invalid FT-FlashAttention options.\n");
    std::exit(EXIT_FAILURE);
  }
  if (options.causal && options.fault_stage != kFaultOutput && options.fault_index > options.fault_query) {
    std::fprintf(stderr, "For causal attention, the fault key must not exceed the fault query.\n");
    std::exit(EXIT_FAILURE);
  }
  return options;
}

__device__ __forceinline__ uint32_t mix_u32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

__global__ void generate_fp8_kernel(Element* output, size_t count, uint32_t seed) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  int quantized = static_cast<int>(mix_u32(seed + static_cast<uint32_t>(index)) % 17U) - 8;
  output[index] = Element(static_cast<float>(quantized) * 0.0625f);
}

__host__ __device__ __forceinline__ float position_weight(int index, int extent) {
  return static_cast<float>(2 * index - (extent - 1));
}

__device__ __forceinline__ bool exceeds_tolerance(float actual, float expected, float abs_tolerance,
                                                   float rel_tolerance) {
  float threshold = abs_tolerance + rel_tolerance * fmaxf(1.0f, fabsf(expected));
  return fabsf(actual - expected) > threshold;
}

template <int Threads>
__device__ float block_sum(float value, float* scratch) {
  constexpr int kWarps = Threads / 32;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();
  float result = lane < kWarps ? scratch[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      result += __shfl_down_sync(0xffffffffU, result, offset);
    }
  }
  if (threadIdx.x == 0) {
    scratch[0] = result;
  }
  __syncthreads();
  float block_result = scratch[0];
  __syncthreads();
  return block_result;
}

template <int Threads, int Components>
__device__ void block_sum_vector(float (&values)[Components], float* scratch) {
  constexpr int kWarps = Threads / 32;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int component = 0; component < Components; ++component) {
      values[component] += __shfl_down_sync(0xffffffffU, values[component], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int component = 0; component < Components; ++component) {
      scratch[component * kWarps + warp] = values[component];
    }
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int component = 0; component < Components; ++component) {
      float value = lane < kWarps ? scratch[component * kWarps + lane] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
      }
      if (lane == 0) {
        scratch[component] = value;
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int component = 0; component < Components; ++component) {
    values[component] = scratch[component];
  }
  __syncthreads();
}

template <int Threads>
__device__ float block_max(float value, float* scratch) {
  constexpr int kWarps = Threads / 32;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();
  float result = lane < kWarps ? scratch[lane] : -FLT_MAX;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      result = fmaxf(result, __shfl_down_sync(0xffffffffU, result, offset));
    }
  }
  if (threadIdx.x == 0) {
    scratch[0] = result;
  }
  __syncthreads();
  float block_result = scratch[0];
  __syncthreads();
  return block_result;
}

__device__ __forceinline__ size_t qkv_offset(int batch, int head, int row, int dim, int heads,
                                             int sequence, int head_dim) {
  return (((static_cast<size_t>(batch) * heads + head) * sequence + row) * head_dim + dim);
}

template <int HeadDim>
__device__ void compute_score_tile(const float* q_shared, const Element* k, float* scores,
                                   int batch_id, int head, int heads, int sequence, int key_begin,
                                   int valid_keys, float score_scale, bool inject_fault,
                                   int fault_index, float fault_value) {
  constexpr int kLanesPerScore = 8;
  constexpr int kScoresPerWarp = 32 / kLanesPerScore;
  constexpr int kScoresPerRound = (kThreads / 32) * kScoresPerWarp;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int score_group = warp * kScoresPerWarp + lane / kLanesPerScore;
  int score_lane = lane % kLanesPerScore;

#pragma unroll
  for (int round = 0; round < kKeyTile / kScoresPerRound; ++round) {
    int local_key = round * kScoresPerRound + score_group;
    float partial = 0.0f;
    if (local_key < valid_keys) {
      int key_index = key_begin + local_key;
#pragma unroll
      for (int dim = score_lane; dim < HeadDim; dim += kLanesPerScore) {
        partial += q_shared[dim] *
                   static_cast<float>(k[qkv_offset(batch_id, head, key_index, dim, heads,
                                                   sequence, HeadDim)]);
      }
    }
#pragma unroll
    for (int offset = kLanesPerScore / 2; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(0xffffffffU, partial, offset, kLanesPerScore);
    }
    if (score_lane == 0 && local_key < kKeyTile) {
      if (local_key < valid_keys) {
        int key_index = key_begin + local_key;
        float score = partial * score_scale;
        if (inject_fault && key_index == fault_index) {
          score += fault_value;
        }
        scores[local_key] = score;
      } else {
        scores[local_key] = -FLT_MAX;
      }
    }
  }
}

template <int HeadDim>
__global__ void prepare_attention_metadata_kernel(
    const Element* k, const Element* v, float* k_sum, float* k_weighted,
    float* k_abs_sum, float* k_weighted_abs_sum, float* v_sum, float* v_weighted,
    int heads, int sequence, int key_tiles) {
  int key_tile = blockIdx.x;
  int batch_head = blockIdx.y;
  int batch_id = batch_head / heads;
  int head = batch_head % heads;
  int key_begin = key_tile * kKeyTile;
  int valid_keys = min(kKeyTile, sequence - key_begin);
  int tid = threadIdx.x;

  if (tid < HeadDim) {
    float sum = 0.0f;
    float weighted = 0.0f;
    float abs_sum = 0.0f;
    float weighted_abs_sum = 0.0f;
    for (int local_key = 0; local_key < valid_keys; ++local_key) {
      int key_index = key_begin + local_key;
      float value =
          static_cast<float>(k[qkv_offset(batch_id, head, key_index, tid, heads, sequence, HeadDim)]);
      float weight = position_weight(key_index, sequence);
      sum += value;
      weighted += weight * value;
      abs_sum += fabsf(value);
      weighted_abs_sum += fabsf(weight) * fabsf(value);
    }
    size_t index = (static_cast<size_t>(batch_head) * key_tiles + key_tile) * HeadDim + tid;
    k_sum[index] = sum;
    k_weighted[index] = weighted;
    k_abs_sum[index] = abs_sum;
    k_weighted_abs_sum[index] = weighted_abs_sum;
  }

  if (tid < valid_keys) {
    int key_index = key_begin + tid;
    float sum = 0.0f;
    float weighted = 0.0f;
#pragma unroll
    for (int dim = 0; dim < HeadDim; ++dim) {
      float value =
          static_cast<float>(v[qkv_offset(batch_id, head, key_index, dim, heads, sequence, HeadDim)]);
      sum += value;
      weighted += position_weight(dim, HeadDim) * value;
    }
    size_t index = static_cast<size_t>(batch_head) * sequence + key_index;
    v_sum[index] = sum;
    v_weighted[index] = weighted;
  }
}

template <int HeadDim>
__global__ __launch_bounds__(kThreads) void ft_flash_attention_a5_kernel(
    const Element* q, const Element* k, const Element* v, float* output, int batch, int heads,
    const float* k_sum, const float* k_weighted, const float* k_abs_sum,
    const float* k_weighted_abs_sum, const float* v_sum, const float* v_weighted,
    int sequence, int key_tiles, bool causal, float score_scale, int fault_stage, int fault_batch, int fault_head,
    int fault_query, int fault_index, float fault_value, float score_abs_tolerance,
    float score_rel_tolerance, float softmax_abs_tolerance, float softmax_rel_tolerance,
    float pv_abs_tolerance, float pv_rel_tolerance, DeviceStats* stats) {
  __shared__ float q_shared[HeadDim];
  __shared__ float scores[kKeyTile];
  __shared__ float exp_scores[kKeyTile];
  __shared__ float output_shared[HeadDim];
  __shared__ float scratch[kThreads];
  __shared__ float online_m;
  __shared__ float online_l;
  __shared__ float verifier_m;
  __shared__ float verifier_l;
  __shared__ float online_u1;
  __shared__ float online_u2;
  __shared__ float verifier_u1;
  __shared__ float verifier_u2;
  __shared__ float expected_output_sum;
  __shared__ float expected_output_weighted;
  __shared__ float stage_alpha;
  __shared__ float stage_verifier_alpha;
  __shared__ float stage_new_m;
  __shared__ float stage_verifier_new_m;
  __shared__ int recovery_action;
  __shared__ int recovery_index;
  __shared__ float recovery_delta;

  int query = blockIdx.x;
  int batch_head = blockIdx.y;
  int batch_id = batch_head / heads;
  int head = batch_head % heads;
  int tid = threadIdx.x;
  if (batch_id >= batch || query >= sequence) {
    return;
  }

  if (tid < HeadDim) {
    q_shared[tid] = static_cast<float>(q[qkv_offset(batch_id, head, query, tid, heads, sequence, HeadDim)]);
  }
  if (tid == 0) {
    online_m = -FLT_MAX;
    online_l = 0.0f;
    verifier_m = -FLT_MAX;
    verifier_l = 0.0f;
    online_u1 = 0.0f;
    online_u2 = 0.0f;
    verifier_u1 = 0.0f;
    verifier_u2 = 0.0f;
    expected_output_sum = 0.0f;
    expected_output_weighted = 0.0f;
  }
  __syncthreads();

  float output_accumulator = 0.0f;
  bool target_row = batch_id == fault_batch && head == fault_head && query == fault_query;

  for (int key_begin = 0; key_begin < sequence; key_begin += kKeyTile) {
    int valid_keys = min(kKeyTile, sequence - key_begin);
    if (causal) {
      valid_keys = min(valid_keys, query + 1 - key_begin);
      if (valid_keys <= 0) {
        break;
      }
    }

    // Stage 1: compute one transient QK score tile and verify it before exp().
    compute_score_tile<HeadDim>(q_shared, k, scores, batch_id, head, heads, sequence,
                                key_begin, valid_keys, score_scale,
                                fault_stage == kFaultScore && target_row, fault_index, fault_value);
    __syncthreads();

    float actual_score_projection[2] = {
        tid < valid_keys ? scores[tid] : 0.0f,
        tid < valid_keys ? position_weight(key_begin + tid, sequence) * scores[tid] : 0.0f};
    block_sum_vector<kThreads>(actual_score_projection, scratch);
    float actual_score_sum = actual_score_projection[0];
    float actual_score_weighted = actual_score_projection[1];

    float expected_score_sum_local = 0.0f;
    float expected_score_weighted_local = 0.0f;
    float score_scale_sum_local = 0.0f;
    float score_scale_weighted_local = 0.0f;
    if (tid < HeadDim) {
      float qval = q_shared[tid];
      int full_tile_keys = min(kKeyTile, sequence - key_begin);
      if (valid_keys == full_tile_keys) {
        int key_tile = key_begin / kKeyTile;
        size_t metadata_index =
            (static_cast<size_t>(batch_head) * key_tiles + key_tile) * HeadDim + tid;
        expected_score_sum_local = qval * k_sum[metadata_index] * score_scale;
        expected_score_weighted_local = qval * k_weighted[metadata_index] * score_scale;
        score_scale_sum_local = fabsf(qval) * k_abs_sum[metadata_index] * fabsf(score_scale);
        score_scale_weighted_local =
            fabsf(qval) * k_weighted_abs_sum[metadata_index] * fabsf(score_scale);
      } else {
        float local_sum = 0.0f;
        float local_weighted = 0.0f;
        float local_abs_sum = 0.0f;
        float local_weighted_abs_sum = 0.0f;
        for (int local_key = 0; local_key < valid_keys; ++local_key) {
          int key_index = key_begin + local_key;
          float kval = static_cast<float>(
              k[qkv_offset(batch_id, head, key_index, tid, heads, sequence, HeadDim)]);
          float key_weight = position_weight(key_index, sequence);
          local_sum += kval;
          local_weighted += key_weight * kval;
          local_abs_sum += fabsf(kval);
          local_weighted_abs_sum += fabsf(key_weight) * fabsf(kval);
        }
        expected_score_sum_local = qval * local_sum * score_scale;
        expected_score_weighted_local = qval * local_weighted * score_scale;
        score_scale_sum_local = fabsf(qval) * local_abs_sum * fabsf(score_scale);
        score_scale_weighted_local =
            fabsf(qval) * local_weighted_abs_sum * fabsf(score_scale);
      }
    }
    float expected_score_projection[4] = {expected_score_sum_local, expected_score_weighted_local,
                                          score_scale_sum_local, score_scale_weighted_local};
    block_sum_vector<kThreads>(expected_score_projection, scratch);
    float expected_score_sum = expected_score_projection[0];
    float expected_score_weighted = expected_score_projection[1];
    float score_scale_sum = expected_score_projection[2];
    float score_scale_weighted = expected_score_projection[3];

    if (tid == 0) {
      recovery_action = 0;
      recovery_index = -1;
      recovery_delta = 0.0f;
      float delta_sum = actual_score_sum - expected_score_sum;
      float delta_weighted = actual_score_weighted - expected_score_weighted;
      float sum_threshold = score_abs_tolerance + score_rel_tolerance * fmaxf(1.0f, score_scale_sum);
      float weighted_threshold =
          score_abs_tolerance + score_rel_tolerance * fmaxf(1.0f, score_scale_weighted);
      bool bad = fabsf(delta_sum) > sum_threshold || fabsf(delta_weighted) > weighted_threshold;
      if (bad) {
        atomicAdd(&stats->score_detected, 1);
        if (fabsf(delta_sum) > sum_threshold) {
          float observed_weight = delta_weighted / delta_sum;
          int key_index = __float2int_rn(0.5f * (observed_weight + static_cast<float>(sequence - 1)));
          int local_key = key_index - key_begin;
          if (local_key >= 0 && local_key < valid_keys &&
              fabsf(observed_weight - position_weight(key_index, sequence)) <= 0.25f) {
            recovery_action = 1;
            recovery_index = local_key;
            recovery_delta = delta_sum;
            atomicAdd(&stats->score_corrected, 1);
          }
        }
        if (recovery_action == 0) {
          recovery_action = 2;
          atomicAdd(&stats->score_replayed, 1);
        }
      }
    }
    __syncthreads();

    if (recovery_action == 1) {
      if (tid == recovery_index) {
        scores[tid] -= recovery_delta;
      }
    } else if (recovery_action == 2) {
      compute_score_tile<HeadDim>(q_shared, k, scores, batch_id, head, heads, sequence,
                                  key_begin, valid_keys, score_scale, false, fault_index,
                                  fault_value);
    }
    __syncthreads();

    float primary_tile_max = block_max<kThreads>(tid < valid_keys ? scores[tid] : -FLT_MAX, scratch);
    float verifier_tile_max = block_max<kThreads>(tid < valid_keys ? scores[tid] : -FLT_MAX, scratch);
    if (tid == 0) {
      if (fabsf(primary_tile_max - verifier_tile_max) > softmax_abs_tolerance) {
        atomicAdd(&stats->softmax_detected, 1);
        atomicAdd(&stats->softmax_replayed, 1);
      }
      stage_new_m = fmaxf(online_m, verifier_tile_max);
      stage_verifier_new_m = fmaxf(verifier_m, verifier_tile_max);
      stage_alpha = online_m == -FLT_MAX ? 0.0f : __expf(online_m - stage_new_m);
      stage_verifier_alpha =
          verifier_m == -FLT_MAX ? 0.0f : __expf(verifier_m - stage_verifier_new_m);
    }
    __syncthreads();

    // Stage 2: independently recompute exponential moments before committing online state.
    float primary_exp = 0.0f;
    if (tid < valid_keys) {
      primary_exp = __expf(scores[tid] - stage_new_m);
      int key_index = key_begin + tid;
      if (fault_stage == kFaultSoftmax && target_row && key_index == fault_index) {
        primary_exp += fault_value;
      }
      exp_scores[tid] = primary_exp;
    } else if (tid < kKeyTile) {
      exp_scores[tid] = 0.0f;
    }
    __syncthreads();

    float weight = tid < valid_keys ? position_weight(key_begin + tid, sequence) : 0.0f;
    float actual_exp_projection[4] = {
        tid < valid_keys ? exp_scores[tid] : 0.0f,
        tid < valid_keys ? weight * exp_scores[tid] : 0.0f,
        tid < valid_keys ? weight * weight * exp_scores[tid] : 0.0f,
        tid < valid_keys && exp_scores[tid] < 0.0f ? 1.0f : 0.0f};
    block_sum_vector<kThreads>(actual_exp_projection, scratch);
    float actual_exp_sum = actual_exp_projection[0];
    float actual_exp_weighted = actual_exp_projection[1];
    float actual_exp_second = actual_exp_projection[2];
    float negative_exp_count = actual_exp_projection[3];

    float verifier_exp = tid < valid_keys ? __expf(scores[tid] - stage_verifier_new_m) : 0.0f;
    float verifier_exp_projection[3] = {verifier_exp, weight * verifier_exp,
                                        weight * weight * verifier_exp};
    block_sum_vector<kThreads>(verifier_exp_projection, scratch);
    float verifier_exp_sum = verifier_exp_projection[0];
    float verifier_exp_weighted = verifier_exp_projection[1];
    float verifier_exp_second = verifier_exp_projection[2];

    if (tid == 0) {
      recovery_action = 0;
      recovery_index = -1;
      recovery_delta = 0.0f;
      float delta0 = actual_exp_sum - verifier_exp_sum;
      float delta1 = actual_exp_weighted - verifier_exp_weighted;
      float delta2 = actual_exp_second - verifier_exp_second;
      bool bad = negative_exp_count > 0.0f ||
                 exceeds_tolerance(actual_exp_sum, verifier_exp_sum, softmax_abs_tolerance,
                                   softmax_rel_tolerance) ||
                 exceeds_tolerance(actual_exp_weighted, verifier_exp_weighted, softmax_abs_tolerance,
                                   softmax_rel_tolerance) ||
                 exceeds_tolerance(actual_exp_second, verifier_exp_second, softmax_abs_tolerance,
                                   softmax_rel_tolerance);
      if (bad) {
        atomicAdd(&stats->softmax_detected, 1);
        float delta_threshold =
            softmax_abs_tolerance + softmax_rel_tolerance * fmaxf(1.0f, fabsf(verifier_exp_sum));
        if (fabsf(delta0) > delta_threshold) {
          float observed_weight = delta1 / delta0;
          int key_index = __float2int_rn(0.5f * (observed_weight + static_cast<float>(sequence - 1)));
          int local_key = key_index - key_begin;
          float second_residual = fabsf(delta2 / delta0 - observed_weight * observed_weight);
          if (local_key >= 0 && local_key < valid_keys &&
              fabsf(observed_weight - position_weight(key_index, sequence)) <= 0.25f &&
              second_residual <= fmaxf(1.0f, fabsf(observed_weight)) * 0.5f &&
              exp_scores[local_key] - delta0 >= -softmax_abs_tolerance) {
            recovery_action = 1;
            recovery_index = local_key;
            recovery_delta = delta0;
            atomicAdd(&stats->softmax_corrected, 1);
          }
        }
        if (recovery_action == 0) {
          recovery_action = 2;
          atomicAdd(&stats->softmax_replayed, 1);
        }
      }
    }
    __syncthreads();

    if (recovery_action == 1 && tid == recovery_index) {
      exp_scores[tid] = fmaxf(0.0f, exp_scores[tid] - recovery_delta);
    } else if (recovery_action == 2 && tid < valid_keys) {
      exp_scores[tid] = __expf(scores[tid] - stage_verifier_new_m);
    }
    __syncthreads();

    if (tid == 0) {
      online_m = stage_verifier_new_m;
      online_l = stage_alpha * online_l + verifier_exp_sum;
      online_u1 = stage_alpha * online_u1 + verifier_exp_weighted;
      online_u2 = stage_alpha * online_u2 + verifier_exp_second;
      verifier_m = stage_verifier_new_m;
      verifier_l = stage_verifier_alpha * verifier_l + verifier_exp_sum;
      verifier_u1 = stage_verifier_alpha * verifier_u1 + verifier_exp_weighted;
      verifier_u2 = stage_verifier_alpha * verifier_u2 + verifier_exp_second;

      if (fault_stage == kFaultSoftmaxState && target_row && fault_index >= key_begin &&
          fault_index < key_begin + valid_keys) {
        online_l += fault_value;
      }
      bool state_bad =
          exceeds_tolerance(online_m, verifier_m, softmax_abs_tolerance, softmax_rel_tolerance) ||
          exceeds_tolerance(online_l, verifier_l, softmax_abs_tolerance, softmax_rel_tolerance) ||
          exceeds_tolerance(online_u1, verifier_u1, softmax_abs_tolerance, softmax_rel_tolerance) ||
          exceeds_tolerance(online_u2, verifier_u2, softmax_abs_tolerance, softmax_rel_tolerance);
      if (state_bad) {
        atomicAdd(&stats->softmax_detected, 1);
        atomicAdd(&stats->softmax_replayed, 1);
        online_m = verifier_m;
        online_l = verifier_l;
        online_u1 = verifier_u1;
        online_u2 = verifier_u2;
      }
    }
    __syncthreads();

    // Stage 3: update PV and an independent pair of output projections.
    if (tid < HeadDim) {
      output_accumulator *= stage_alpha;
      for (int local_key = 0; local_key < valid_keys; ++local_key) {
        int key_index = key_begin + local_key;
        float vvalue =
            static_cast<float>(v[qkv_offset(batch_id, head, key_index, tid, heads, sequence, HeadDim)]);
        output_accumulator += exp_scores[local_key] * vvalue;
      }
    }

    float expected_output_sum_local = 0.0f;
    float expected_output_weighted_local = 0.0f;
    if (tid < valid_keys) {
      int key_index = key_begin + tid;
      size_t metadata_index = static_cast<size_t>(batch_head) * sequence + key_index;
      expected_output_sum_local = exp_scores[tid] * v_sum[metadata_index];
      expected_output_weighted_local = exp_scores[tid] * v_weighted[metadata_index];
    }
    float expected_output_projection[2] = {expected_output_sum_local,
                                           expected_output_weighted_local};
    block_sum_vector<kThreads>(expected_output_projection, scratch);
    float tile_expected_output_sum = expected_output_projection[0];
    float tile_expected_output_weighted = expected_output_projection[1];
    if (tid == 0) {
      expected_output_sum = stage_alpha * expected_output_sum + tile_expected_output_sum;
      expected_output_weighted =
          stage_alpha * expected_output_weighted + tile_expected_output_weighted;
    }
    __syncthreads();
  }

  if (tid < HeadDim) {
    output_shared[tid] = output_accumulator / online_l;
    if (fault_stage == kFaultOutput && target_row && tid == fault_index) {
      output_shared[tid] += fault_value;
    }
  }
  __syncthreads();

  float actual_output_projection[2] = {
      tid < HeadDim ? output_shared[tid] : 0.0f,
      tid < HeadDim ? position_weight(tid, HeadDim) * output_shared[tid] : 0.0f};
  block_sum_vector<kThreads>(actual_output_projection, scratch);
  float actual_output_sum = actual_output_projection[0];
  float actual_output_weighted = actual_output_projection[1];
  if (tid == 0) {
    recovery_action = 0;
    recovery_index = -1;
    recovery_delta = 0.0f;
    float expected_sum = expected_output_sum / online_l;
    float expected_weighted = expected_output_weighted / online_l;
    float delta_sum = actual_output_sum - expected_sum;
    float delta_weighted = actual_output_weighted - expected_weighted;
    bool bad = exceeds_tolerance(actual_output_sum, expected_sum, pv_abs_tolerance, pv_rel_tolerance) ||
               exceeds_tolerance(actual_output_weighted, expected_weighted, pv_abs_tolerance,
                                 pv_rel_tolerance);
    if (bad) {
      atomicAdd(&stats->pv_detected, 1);
      float threshold = pv_abs_tolerance + pv_rel_tolerance * fmaxf(1.0f, fabsf(expected_sum));
      if (fabsf(delta_sum) > threshold) {
        float observed_weight = delta_weighted / delta_sum;
        int dim = __float2int_rn(0.5f * (observed_weight + static_cast<float>(HeadDim - 1)));
        if (dim >= 0 && dim < HeadDim &&
            fabsf(observed_weight - position_weight(dim, HeadDim)) <= 0.25f) {
          recovery_action = 1;
          recovery_index = dim;
          recovery_delta = delta_sum;
          atomicAdd(&stats->pv_corrected, 1);
        }
      }
      if (recovery_action == 0) {
        atomicAdd(&stats->failed_rows, 1);
      }
    }
  }
  __syncthreads();

  if (recovery_action == 1 && tid == recovery_index) {
    output_shared[tid] -= recovery_delta;
  }
  __syncthreads();

  if (tid < HeadDim) {
    output[qkv_offset(batch_id, head, query, tid, heads, sequence, HeadDim)] = output_shared[tid];
  }
}

static VerificationResult verify_attention(const std::vector<Element>& q, const std::vector<Element>& k,
                                           const std::vector<Element>& v, const std::vector<float>& output,
                                           const Options& options) {
  VerificationResult result;
  std::vector<double> logits(options.sequence);
  std::vector<double> probabilities(options.sequence);
  double scale = 1.0 / std::sqrt(static_cast<double>(options.head_dim));

  for (int batch = 0; batch < options.batch; ++batch) {
    for (int head = 0; head < options.heads; ++head) {
      for (int query = 0; query < options.sequence; ++query) {
        int key_count = options.causal ? query + 1 : options.sequence;
        double row_max = -DBL_MAX;
        for (int key = 0; key < key_count; ++key) {
          double score = 0.0;
          for (int dim = 0; dim < options.head_dim; ++dim) {
            size_t q_index =
                (((static_cast<size_t>(batch) * options.heads + head) * options.sequence + query) *
                     options.head_dim +
                 dim);
            size_t k_index =
                (((static_cast<size_t>(batch) * options.heads + head) * options.sequence + key) *
                     options.head_dim +
                 dim);
            score += static_cast<double>(static_cast<float>(q[q_index])) *
                     static_cast<double>(static_cast<float>(k[k_index]));
          }
          logits[key] = score * scale;
          row_max = std::max(row_max, logits[key]);
        }
        double denominator = 0.0;
        for (int key = 0; key < key_count; ++key) {
          probabilities[key] = std::exp(logits[key] - row_max);
          denominator += probabilities[key];
        }
        for (int key = 0; key < key_count; ++key) {
          probabilities[key] /= denominator;
        }
        for (int dim = 0; dim < options.head_dim; ++dim) {
          double reference = 0.0;
          for (int key = 0; key < key_count; ++key) {
            size_t v_index =
                (((static_cast<size_t>(batch) * options.heads + head) * options.sequence + key) *
                     options.head_dim +
                 dim);
            reference += probabilities[key] * static_cast<double>(static_cast<float>(v[v_index]));
          }
          size_t output_index =
              (((static_cast<size_t>(batch) * options.heads + head) * options.sequence + query) *
                   options.head_dim +
               dim);
          float absolute_error = static_cast<float>(std::fabs(static_cast<double>(output[output_index]) - reference));
          float tolerance = options.verify_abs_tolerance +
                            options.verify_rel_tolerance * std::max(1.0f, static_cast<float>(std::fabs(reference)));
          result.max_absolute_error = std::max(result.max_absolute_error, absolute_error);
          result.max_tolerance_ratio = std::max(result.max_tolerance_ratio, absolute_error / tolerance);
          if (absolute_error > tolerance) {
            ++result.mismatches;
          }
        }
      }
    }
  }
  return result;
}

static bool stats_match_fault(const DeviceStats& stats, FaultStage stage) {
  switch (stage) {
    case kFaultScore:
      return stats.score_detected > 0 && stats.score_corrected > 0 && stats.failed_rows == 0;
    case kFaultSoftmax:
      return stats.softmax_detected > 0 &&
             (stats.softmax_corrected > 0 || stats.softmax_replayed > 0) && stats.failed_rows == 0;
    case kFaultSoftmaxState:
      return stats.softmax_detected > 0 && stats.softmax_replayed > 0 && stats.failed_rows == 0;
    case kFaultOutput:
      return stats.pv_detected > 0 && stats.pv_corrected > 0 && stats.failed_rows == 0;
    default:
      return stats.score_detected == 0 && stats.softmax_detected == 0 && stats.pv_detected == 0 &&
             stats.failed_rows == 0;
  }
}

int main(int argc, char** argv) {
  Options options = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(0));

  size_t element_count = static_cast<size_t>(options.batch) * options.heads * options.sequence *
                         options.head_dim;
  Element* d_q = nullptr;
  Element* d_k = nullptr;
  Element* d_v = nullptr;
  float* d_output = nullptr;
  DeviceStats* d_stats = nullptr;
  float* d_k_sum = nullptr;
  float* d_k_weighted = nullptr;
  float* d_k_abs_sum = nullptr;
  float* d_k_weighted_abs_sum = nullptr;
  float* d_v_sum = nullptr;
  float* d_v_weighted = nullptr;
  int key_tiles = (options.sequence + kKeyTile - 1) / kKeyTile;
  size_t k_metadata_count = static_cast<size_t>(options.batch) * options.heads * key_tiles * options.head_dim;
  size_t v_metadata_count = static_cast<size_t>(options.batch) * options.heads * options.sequence;
  CUDA_CHECK(cudaMalloc(&d_q, element_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_k, element_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_v, element_count * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&d_output, element_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_stats, sizeof(DeviceStats)));
  CUDA_CHECK(cudaMalloc(&d_k_sum, k_metadata_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k_weighted, k_metadata_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k_abs_sum, k_metadata_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k_weighted_abs_sum, k_metadata_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v_sum, v_metadata_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v_weighted, v_metadata_count * sizeof(float)));

  int generation_threads = 256;
  int generation_blocks = static_cast<int>((element_count + generation_threads - 1) / generation_threads);
  generate_fp8_kernel<<<generation_blocks, generation_threads>>>(d_q, element_count, 0x12345678U);
  generate_fp8_kernel<<<generation_blocks, generation_threads>>>(d_k, element_count, 0x9abcdef0U);
  generate_fp8_kernel<<<generation_blocks, generation_threads>>>(d_v, element_count, 0x31415926U);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t metadata_start;
  cudaEvent_t metadata_stop;
  CUDA_CHECK(cudaEventCreate(&metadata_start));
  CUDA_CHECK(cudaEventCreate(&metadata_stop));
  CUDA_CHECK(cudaEventRecord(metadata_start));
  dim3 metadata_grid(key_tiles, options.batch * options.heads);
  if (options.head_dim == 64) {
    prepare_attention_metadata_kernel<64><<<metadata_grid, kThreads>>>(
        d_k, d_v, d_k_sum, d_k_weighted, d_k_abs_sum, d_k_weighted_abs_sum, d_v_sum,
        d_v_weighted, options.heads, options.sequence, key_tiles);
  } else {
    prepare_attention_metadata_kernel<128><<<metadata_grid, kThreads>>>(
        d_k, d_v, d_k_sum, d_k_weighted, d_k_abs_sum, d_k_weighted_abs_sum, d_v_sum,
        d_v_weighted, options.heads, options.sequence, key_tiles);
  }
  CUDA_CHECK(cudaEventRecord(metadata_stop));
  CUDA_CHECK(cudaEventSynchronize(metadata_stop));
  CUDA_CHECK(cudaGetLastError());
  float metadata_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&metadata_ms, metadata_start, metadata_stop));

  dim3 grid(options.sequence, options.batch * options.heads);
  float score_scale = 1.0f / std::sqrt(static_cast<float>(options.head_dim));
  auto launch = [&](FaultStage fault_stage, cudaStream_t stream) {
    if (options.head_dim == 64) {
      ft_flash_attention_a5_kernel<64><<<grid, kThreads, 0, stream>>>(
          d_q, d_k, d_v, d_output, options.batch, options.heads,
          d_k_sum, d_k_weighted, d_k_abs_sum, d_k_weighted_abs_sum, d_v_sum, d_v_weighted,
          options.sequence, key_tiles, options.causal, score_scale, static_cast<int>(fault_stage), options.fault_batch, options.fault_head,
          options.fault_query, options.fault_index, options.fault_value, options.score_abs_tolerance,
          options.score_rel_tolerance, options.softmax_abs_tolerance, options.softmax_rel_tolerance,
          options.pv_abs_tolerance, options.pv_rel_tolerance, d_stats);
    } else {
      ft_flash_attention_a5_kernel<128><<<grid, kThreads, 0, stream>>>(
          d_q, d_k, d_v, d_output, options.batch, options.heads,
          d_k_sum, d_k_weighted, d_k_abs_sum, d_k_weighted_abs_sum, d_v_sum, d_v_weighted,
          options.sequence, key_tiles, options.causal, score_scale, static_cast<int>(fault_stage), options.fault_batch, options.fault_head,
          options.fault_query, options.fault_index, options.fault_value, options.score_abs_tolerance,
          options.score_rel_tolerance, options.softmax_abs_tolerance, options.softmax_rel_tolerance,
          options.pv_abs_tolerance, options.pv_rel_tolerance, d_stats);
    }
  };

  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    CUDA_CHECK(cudaMemsetAsync(d_stats, 0, sizeof(DeviceStats), stream));
    launch(kFaultNone, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemsetAsync(d_stats, 0, sizeof(DeviceStats), stream));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int iteration = 0; iteration < options.repeat; ++iteration) {
    launch(kFaultNone, stream);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  float average_ms = total_ms / options.repeat;

  CUDA_CHECK(cudaMemsetAsync(d_stats, 0, sizeof(DeviceStats), stream));
  launch(options.fault_stage, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());

  std::vector<Element> h_q(element_count);
  std::vector<Element> h_k(element_count);
  std::vector<Element> h_v(element_count);
  std::vector<float> h_output(element_count);
  DeviceStats stats{};
  CUDA_CHECK(cudaMemcpy(h_q.data(), d_q, element_count * sizeof(Element), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_k.data(), d_k, element_count * sizeof(Element), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_v.data(), d_v, element_count * sizeof(Element), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, element_count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&stats, d_stats, sizeof(DeviceStats), cudaMemcpyDeviceToHost));

  VerificationResult verification = verify_attention(h_q, h_k, h_v, h_output, options);
  bool numerical_pass = verification.mismatches == 0;
  bool fault_pass = stats_match_fault(stats, options.fault_stage);
  bool passed = numerical_pass && fault_pass;
  double attended_pairs = options.causal
                              ? 0.5 * static_cast<double>(options.sequence) * (options.sequence + 1)
                              : static_cast<double>(options.sequence) * options.sequence;
  double attention_flops =
      4.0 * options.batch * options.heads * attended_pairs * options.head_dim;
  double approximate_tflops = attention_flops / (average_ms * 1.0e9);

  std::printf("version: FT-FlashAttention A5 cooperative QK dot products\n");
  std::printf("layout: Q/K/V=[B,H,N,D], FP8 E4M3; O=FP32\n");
  std::printf("shape: B=%d H=%d N=%d D=%d causal=%d\n", options.batch, options.heads,
              options.sequence, options.head_dim, options.causal ? 1 : 0);
  std::printf("key_tile: %d\n", kKeyTile);
  std::printf("avg_no_fault_time_ms: %.6f\n", average_ms);
  std::printf("approx_attention_tflops: %.6f\n", approximate_tflops);
  std::printf("metadata_prepare_time_ms: %.6f\n", metadata_ms);
  std::printf("first_call_approx_time_ms: %.6f\n", metadata_ms + average_ms);
  std::printf("fault_stage: %s\n", fault_stage_name(options.fault_stage));
  std::printf("fault_target: batch=%d head=%d query=%d index=%d delta=%+.6e\n",
              options.fault_batch, options.fault_head, options.fault_query, options.fault_index,
              options.fault_value);
  std::printf("score_detected: %d\n", stats.score_detected);
  std::printf("score_corrected: %d\n", stats.score_corrected);
  std::printf("score_replayed: %d\n", stats.score_replayed);
  std::printf("softmax_detected: %d\n", stats.softmax_detected);
  std::printf("softmax_corrected: %d\n", stats.softmax_corrected);
  std::printf("softmax_replayed: %d\n", stats.softmax_replayed);
  std::printf("pv_detected: %d\n", stats.pv_detected);
  std::printf("pv_corrected: %d\n", stats.pv_corrected);
  std::printf("failed_rows: %d\n", stats.failed_rows);
  std::printf("max_absolute_error: %.6e\n", verification.max_absolute_error);
  std::printf("max_tolerance_ratio: %.6e\n", verification.max_tolerance_ratio);
  std::printf("mismatches: %d\n", verification.mismatches);
  std::printf("verification: %s\n", passed ? "PASS" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(metadata_start));
  CUDA_CHECK(cudaEventDestroy(metadata_stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_stats));
  CUDA_CHECK(cudaFree(d_k_sum));
  CUDA_CHECK(cudaFree(d_k_weighted));
  CUDA_CHECK(cudaFree(d_k_abs_sum));
  CUDA_CHECK(cudaFree(d_k_weighted_abs_sum));
  CUDA_CHECK(cudaFree(d_v_sum));
  CUDA_CHECK(cudaFree(d_v_weighted));
  return passed ? 0 : 1;
}
