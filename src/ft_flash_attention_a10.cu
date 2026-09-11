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

#include "cute/algorithm/gemm.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/arch/mma_sm90_gmma_ext.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/tensor.hpp"

#define CUDA_CHECK(call)                                                                             \
  do {                                                                                               \
    cudaError_t error = (call);                                                                      \
    if (error != cudaSuccess) {                                                                      \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      std::exit(EXIT_FAILURE);                                                                       \
    }                                                                                                \
  } while (0)

using Element = cutlass::float_e4m3_t;

constexpr int kConsumerWarpgroups = 2;
constexpr int kThreads = kConsumerWarpgroups * 128;
constexpr int kQueryTile = kConsumerWarpgroups * 64;
constexpr int kKeyTile = 128;
constexpr int kScoreStride = kQueryTile + 1;
constexpr float kProbabilityScale = 256.0f;

using QkMmaShape = cute::Shape<cute::_64, cute::_128, cute::_32>;
using QkGmmaOp = decltype(cute::GMMA::ss_op_selector<Element, Element, float, QkMmaShape,
                                                      cute::GMMA::Major::K, cute::GMMA::Major::K>());
using QkTiledMma = decltype(cute::make_tiled_mma(QkGmmaOp{}));

template <int HeadDim>
struct AttentionTraits {
  using PvMmaShape = cute::Shape<cute::_64, cute::Int<HeadDim>, cute::_32>;
  using PvGmmaOp = decltype(cute::GMMA::ss_op_selector<Element, Element, float, PvMmaShape,
                                                         cute::GMMA::Major::K, cute::GMMA::Major::K>());
  using PvTiledMma = decltype(cute::make_tiled_mma(PvGmmaOp{}));
  using SmemLayoutQ = decltype(cute::tile_to_shape(
      cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
      cute::Shape<cute::_64, cute::Int<HeadDim>>{}));
  using SmemLayoutK = decltype(cute::tile_to_shape(
      cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
      cute::Shape<cute::_128, cute::Int<HeadDim>>{}));
  using SmemLayoutP = decltype(cute::tile_to_shape(
      cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
      cute::Shape<cute::_64, cute::_128>{}));
  using SmemLayoutV = decltype(cute::tile_to_shape(
      cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
      cute::Shape<cute::Int<HeadDim>, cute::_128>{}));
};

template <int HeadDim>
struct alignas(128) AttentionSharedStorage {
  using Traits = AttentionTraits<HeadDim>;
  alignas(128) Element q[kConsumerWarpgroups][cute::cosize_v<typename Traits::SmemLayoutQ>];
  union alignas(128) {
    Element k[cute::cosize_v<typename Traits::SmemLayoutK>];
    Element p[kConsumerWarpgroups][cute::cosize_v<typename Traits::SmemLayoutP>];
  } key_probability;
  alignas(128) Element v[cute::cosize_v<typename Traits::SmemLayoutV>];
  union alignas(128) {
    float score[kScoreStride * kKeyTile];
    float pv[kScoreStride * HeadDim];
  } score_pv;
  alignas(128) float output[kScoreStride * HeadDim];
  float expected_score_sum[kQueryTile];
  float expected_score_weighted[kQueryTile];
  float score_scale_sum[kQueryTile];
  float score_scale_weighted[kQueryTile];
  float online_m[kQueryTile];
  float online_l[kQueryTile];
  float verifier_m[kQueryTile];
  float verifier_l[kQueryTile];
  float online_u1[kQueryTile];
  float online_u2[kQueryTile];
  float verifier_u1[kQueryTile];
  float verifier_u2[kQueryTile];
  float stage_alpha[kQueryTile];
  float expected_output_sum[kQueryTile];
  float expected_output_weighted[kQueryTile];
  float recovery_delta[kQueryTile];
  int recovery_action[kQueryTile];
  int recovery_index[kQueryTile];
};

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
__global__ __launch_bounds__(kThreads) void ft_flash_attention_a10_kernel(
    const Element* q, const Element* k, const Element* v, float* output, int batch, int heads,
    const float* k_sum, const float* k_weighted, const float* k_abs_sum,
    const float* k_weighted_abs_sum, const float* v_sum, const float* v_weighted,
    int sequence, int key_tiles, bool causal, float score_scale, int fault_stage, int fault_batch,
    int fault_head, int fault_query, int fault_index, float fault_value,
    float score_abs_tolerance, float score_rel_tolerance, float softmax_abs_tolerance,
    float softmax_rel_tolerance, float pv_abs_tolerance, float pv_rel_tolerance,
    DeviceStats* stats) {
  using Traits = AttentionTraits<HeadDim>;
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  auto& storage = *reinterpret_cast<AttentionSharedStorage<HeadDim>*>(shared_bytes);
  int tid = threadIdx.x;
  int consumer_group = tid / 128;
  int consumer_thread = tid % 128;
  int mma_row = consumer_thread >> 1;
  auto sQ = cute::make_tensor(cute::make_smem_ptr(storage.q[consumer_group]),
                              typename Traits::SmemLayoutQ{});
  auto sK = cute::make_tensor(cute::make_smem_ptr(storage.key_probability.k), typename Traits::SmemLayoutK{});
  auto sP = cute::make_tensor(
      cute::make_smem_ptr(storage.key_probability.p[consumer_group]),
      typename Traits::SmemLayoutP{});
  auto sV = cute::make_tensor(cute::make_smem_ptr(storage.v), typename Traits::SmemLayoutV{});
  using ScoreLayout = decltype(cute::make_layout(
      cute::make_shape(cute::_64{}, cute::_128{}),
      cute::make_stride(cute::_1{}, cute::Int<kScoreStride>{})));
  using PvLayout = decltype(cute::make_layout(
      cute::make_shape(cute::_64{}, cute::Int<HeadDim>{}),
      cute::make_stride(cute::_1{}, cute::Int<kScoreStride>{})));
  auto sScore = cute::make_tensor(
      cute::make_smem_ptr(storage.score_pv.score + consumer_group * 64), ScoreLayout{});
  auto sPv = cute::make_tensor(
      cute::make_smem_ptr(storage.score_pv.pv + consumer_group * 64), PvLayout{});

  int row = tid >> 1;
  int pair_lane = tid & 1;
  int query_begin = blockIdx.x * kQueryTile;
  int query = query_begin + row;
  int batch_head = blockIdx.y;
  int batch_id = batch_head / heads;
  int head = batch_head % heads;
  bool valid_query = batch_id < batch && query < sequence;
  bool target_row =
      valid_query && batch_id == fault_batch && head == fault_head && query == fault_query;

  for (int index = consumer_thread; index < 64 * HeadDim; index += 128) {
    int local_row = index / HeadDim;
    int dim = index % HeadDim;
    int global_query = query_begin + consumer_group * 64 + local_row;
    sQ(local_row, dim) =
        global_query < sequence
            ? q[qkv_offset(batch_id, head, global_query, dim, heads, sequence, HeadDim)]
            : Element(0.0f);
  }
  for (int index = tid; index < kScoreStride * HeadDim; index += kThreads) {
    storage.output[index] = 0.0f;
  }
  if (tid < kQueryTile) {
    storage.online_m[tid] = -FLT_MAX;
    storage.online_l[tid] = 0.0f;
    storage.verifier_m[tid] = -FLT_MAX;
    storage.verifier_l[tid] = 0.0f;
    storage.online_u1[tid] = 0.0f;
    storage.online_u2[tid] = 0.0f;
    storage.verifier_u1[tid] = 0.0f;
    storage.verifier_u2[tid] = 0.0f;
    storage.expected_output_sum[tid] = 0.0f;
    storage.expected_output_weighted[tid] = 0.0f;
  }
  __syncthreads();

  int key_limit = causal ? min(sequence, query_begin + kQueryTile) : sequence;
  for (int key_begin = 0; key_begin < key_limit; key_begin += kKeyTile) {
    int tile_keys = min(kKeyTile, sequence - key_begin);
    int row_valid_keys = valid_query ? tile_keys : 0;
    if (causal) {
      row_valid_keys = min(row_valid_keys, max(0, query + 1 - key_begin));
    }

    for (int index = tid; index < kKeyTile * HeadDim; index += kThreads) {
      int local_key = index / HeadDim;
      int dim = index % HeadDim;
      int key_index = key_begin + local_key;
      Element kval = key_index < sequence
                         ? k[qkv_offset(batch_id, head, key_index, dim, heads, sequence, HeadDim)]
                         : Element(0.0f);
      Element vval = key_index < sequence
                         ? v[qkv_offset(batch_id, head, key_index, dim, heads, sequence, HeadDim)]
                         : Element(0.0f);
      sK(local_key, dim) = kval;
      sV(dim, local_key) = vval;
    }
    __syncthreads();

    float expected_sum = 0.0f;
    float expected_weighted = 0.0f;
    float expected_scale = 0.0f;
    float expected_weighted_scale = 0.0f;
    if (row_valid_keys > 0) {
      int full_tile_keys = min(kKeyTile, sequence - key_begin);
      for (int dim = pair_lane; dim < HeadDim; dim += 2) {
        float qval = static_cast<float>(sQ(mma_row, dim));
        float local_sum = 0.0f;
        float local_weighted = 0.0f;
        float local_abs_sum = 0.0f;
        float local_weighted_abs_sum = 0.0f;
        if (row_valid_keys == full_tile_keys) {
          int key_tile = key_begin / kKeyTile;
          size_t metadata_index =
              (static_cast<size_t>(batch_head) * key_tiles + key_tile) * HeadDim + dim;
          local_sum = k_sum[metadata_index];
          local_weighted = k_weighted[metadata_index];
          local_abs_sum = k_abs_sum[metadata_index];
          local_weighted_abs_sum = k_weighted_abs_sum[metadata_index];
        } else {
          for (int local_key = 0; local_key < row_valid_keys; ++local_key) {
            float kval = static_cast<float>(sK(local_key, dim));
            float weight = position_weight(key_begin + local_key, sequence);
            local_sum += kval;
            local_weighted += weight * kval;
            local_abs_sum += fabsf(kval);
            local_weighted_abs_sum += fabsf(weight) * fabsf(kval);
          }
        }
        expected_sum += qval * local_sum * score_scale;
        expected_weighted += qval * local_weighted * score_scale;
        expected_scale += fabsf(qval) * local_abs_sum * fabsf(score_scale);
        expected_weighted_scale +=
            fabsf(qval) * local_weighted_abs_sum * fabsf(score_scale);
      }
    }
    expected_sum += __shfl_xor_sync(0xffffffffU, expected_sum, 1, 2);
    expected_weighted += __shfl_xor_sync(0xffffffffU, expected_weighted, 1, 2);
    expected_scale += __shfl_xor_sync(0xffffffffU, expected_scale, 1, 2);
    expected_weighted_scale +=
        __shfl_xor_sync(0xffffffffU, expected_weighted_scale, 1, 2);
    if (pair_lane == 0) {
      storage.expected_score_sum[row] = expected_sum;
      storage.expected_score_weighted[row] = expected_weighted;
      storage.score_scale_sum[row] = expected_scale;
      storage.score_scale_weighted[row] = expected_weighted_scale;
    }
    __syncthreads();

    QkTiledMma qk_mma;
    auto qk_thread = qk_mma.get_slice(consumer_thread);
    auto score_accumulator =
        cute::partition_fragment_C(qk_mma, cute::make_shape(cute::_64{}, cute::_128{}));
    cute::clear(score_accumulator);
    cute::warpgroup_fence_operand(score_accumulator);
    cute::warpgroup_arrive();
#pragma unroll
    for (int dim_tile = 0; dim_tile < HeadDim / 32; ++dim_tile) {
      auto q_tile = cute::local_tile(
          sQ, cute::Shape<cute::_64, cute::_32>{}, cute::make_coord(0, dim_tile));
      auto k_tile = cute::local_tile(
          sK, cute::Shape<cute::_128, cute::_32>{}, cute::make_coord(0, dim_tile));
      auto q_descriptor = qk_thread.partition_A(q_tile);
      auto k_descriptor = qk_thread.partition_B(k_tile);
      auto q_fragment = qk_thread.make_fragment_A(q_descriptor);
      auto k_fragment = qk_thread.make_fragment_B(k_descriptor);
      cute::copy(q_descriptor, q_fragment);
      cute::copy(k_descriptor, k_fragment);
      qk_mma.accumulate_ =
          dim_tile == 0 ? cute::GMMA::ScaleOut::Zero : cute::GMMA::ScaleOut::One;
      cute::gemm(qk_mma, q_fragment, k_fragment, score_accumulator);
    }
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(score_accumulator);
    auto score_destination = qk_thread.partition_C(sScore);
    cute::copy(score_accumulator, score_destination);
    __syncthreads();

    for (int index = tid; index < kQueryTile * kKeyTile; index += kThreads) {
      int local_row = index % kQueryTile;
      int local_key = index / kQueryTile;
      int global_query = query_begin + local_row;
      int global_key = key_begin + local_key;
      bool active = global_query < sequence && global_key < sequence &&
                    (!causal || global_key <= global_query);
      float score = active ? storage.score_pv.score[local_key * kScoreStride + local_row] * score_scale
                           : -FLT_MAX;
      if (active && fault_stage == kFaultScore && batch_id == fault_batch &&
          head == fault_head && global_query == fault_query && global_key == fault_index) {
        score += fault_value;
      }
      storage.score_pv.score[local_key * kScoreStride + local_row] = score;
    }
    __syncthreads();

    float actual_score_sum = 0.0f;
    float actual_score_weighted = 0.0f;
    for (int local_key = pair_lane; local_key < row_valid_keys; local_key += 2) {
      float score = storage.score_pv.score[local_key * kScoreStride + row];
      actual_score_sum += score;
      actual_score_weighted +=
          position_weight(key_begin + local_key, sequence) * score;
    }
    actual_score_sum += __shfl_xor_sync(0xffffffffU, actual_score_sum, 1, 2);
    actual_score_weighted +=
        __shfl_xor_sync(0xffffffffU, actual_score_weighted, 1, 2);
    if (pair_lane == 0) {
      storage.recovery_action[row] = 0;
      storage.recovery_index[row] = -1;
      storage.recovery_delta[row] = 0.0f;
      if (row_valid_keys > 0) {
        float delta_sum = actual_score_sum - storage.expected_score_sum[row];
        float delta_weighted =
            actual_score_weighted - storage.expected_score_weighted[row];
        float sum_threshold =
            score_abs_tolerance +
            score_rel_tolerance * fmaxf(1.0f, storage.score_scale_sum[row]);
        float weighted_threshold =
            score_abs_tolerance +
            score_rel_tolerance * fmaxf(1.0f, storage.score_scale_weighted[row]);
        bool bad =
            fabsf(delta_sum) > sum_threshold || fabsf(delta_weighted) > weighted_threshold;
        if (bad) {
          atomicAdd(&stats->score_detected, 1);
          if (fabsf(delta_sum) > sum_threshold) {
            float observed_weight = delta_weighted / delta_sum;
            int key_index =
                __float2int_rn(0.5f * (observed_weight + static_cast<float>(sequence - 1)));
            int local_key = key_index - key_begin;
            if (local_key >= 0 && local_key < row_valid_keys &&
                fabsf(observed_weight - position_weight(key_index, sequence)) <= 0.25f) {
              storage.recovery_action[row] = 1;
              storage.recovery_index[row] = local_key;
              storage.recovery_delta[row] = delta_sum;
              atomicAdd(&stats->score_corrected, 1);
            }
          }
          if (storage.recovery_action[row] == 0) {
            storage.recovery_action[row] = 2;
            atomicAdd(&stats->score_replayed, 1);
          }
        }
      }
    }
    __syncthreads();

    if (storage.recovery_action[row] == 1 && pair_lane == 0) {
      int local_key = storage.recovery_index[row];
      storage.score_pv.score[local_key * kScoreStride + row] -= storage.recovery_delta[row];
    } else if (storage.recovery_action[row] == 2) {
      for (int local_key = pair_lane; local_key < row_valid_keys; local_key += 2) {
        float replay_score = 0.0f;
        for (int dim = 0; dim < HeadDim; ++dim) {
          replay_score +=
              static_cast<float>(sQ(mma_row, dim)) * static_cast<float>(sK(local_key, dim));
        }
        storage.score_pv.score[local_key * kScoreStride + row] = replay_score * score_scale;
      }
    }
    __syncthreads();

    float primary_max = -FLT_MAX;
    float verifier_max = -FLT_MAX;
    for (int local_key = pair_lane; local_key < row_valid_keys; local_key += 2) {
      float score = storage.score_pv.score[local_key * kScoreStride + row];
      primary_max = fmaxf(primary_max, score);
      verifier_max = fmaxf(verifier_max, score);
    }
    primary_max =
        fmaxf(primary_max, __shfl_xor_sync(0xffffffffU, primary_max, 1, 2));
    verifier_max =
        fmaxf(verifier_max, __shfl_xor_sync(0xffffffffU, verifier_max, 1, 2));
    float old_m = storage.online_m[row];
    float old_verifier_m = storage.verifier_m[row];
    float new_m = row_valid_keys > 0 ? fmaxf(old_m, primary_max) : old_m;
    float new_verifier_m =
        row_valid_keys > 0 ? fmaxf(old_verifier_m, verifier_max) : old_verifier_m;
    float alpha = old_m == -FLT_MAX ? 0.0f : __expf(old_m - new_m);
    float verifier_alpha =
        old_verifier_m == -FLT_MAX ? 0.0f : __expf(old_verifier_m - new_verifier_m);

    float actual_u0 = 0.0f;
    float actual_u1 = 0.0f;
    float actual_u2 = 0.0f;
    float verifier_u0 = 0.0f;
    float verifier_tile_u1 = 0.0f;
    float verifier_tile_u2 = 0.0f;
    float negative_count = 0.0f;
    for (int local_key = pair_lane; local_key < kKeyTile; local_key += 2) {
      float primary_probability = 0.0f;
      float verifier_probability = 0.0f;
      if (local_key < row_valid_keys) {
        int key_index = key_begin + local_key;
        primary_probability =
            __expf(storage.score_pv.score[local_key * kScoreStride + row] - new_m) *
            kProbabilityScale;
        verifier_probability =
            __expf(storage.score_pv.score[local_key * kScoreStride + row] - new_verifier_m) *
            kProbabilityScale;
        if (fault_stage == kFaultSoftmax && target_row && key_index == fault_index) {
          primary_probability += fault_value * kProbabilityScale;
        }
      }
      Element primary_quantized(primary_probability);
      Element verifier_quantized(verifier_probability);
      sP(mma_row, local_key) = primary_quantized;
      float actual = static_cast<float>(primary_quantized);
      float expected = static_cast<float>(verifier_quantized);
      float weight = position_weight(key_begin + local_key, sequence);
      if (local_key < row_valid_keys) {
        actual_u0 += actual;
        actual_u1 += weight * actual;
        actual_u2 += weight * weight * actual;
        verifier_u0 += expected;
        verifier_tile_u1 += weight * expected;
        verifier_tile_u2 += weight * weight * expected;
        negative_count += actual < 0.0f ? 1.0f : 0.0f;
      }
    }
    actual_u0 += __shfl_xor_sync(0xffffffffU, actual_u0, 1, 2);
    actual_u1 += __shfl_xor_sync(0xffffffffU, actual_u1, 1, 2);
    actual_u2 += __shfl_xor_sync(0xffffffffU, actual_u2, 1, 2);
    verifier_u0 += __shfl_xor_sync(0xffffffffU, verifier_u0, 1, 2);
    verifier_tile_u1 +=
        __shfl_xor_sync(0xffffffffU, verifier_tile_u1, 1, 2);
    verifier_tile_u2 +=
        __shfl_xor_sync(0xffffffffU, verifier_tile_u2, 1, 2);
    negative_count += __shfl_xor_sync(0xffffffffU, negative_count, 1, 2);

    if (pair_lane == 0) {
      storage.recovery_action[row] = 0;
      storage.recovery_index[row] = -1;
      storage.recovery_delta[row] = 0.0f;
      float delta0 = actual_u0 - verifier_u0;
      float delta1 = actual_u1 - verifier_tile_u1;
      float delta2 = actual_u2 - verifier_tile_u2;
      bool bad =
          negative_count > 0.0f ||
          exceeds_tolerance(actual_u0, verifier_u0, softmax_abs_tolerance,
                            softmax_rel_tolerance) ||
          exceeds_tolerance(actual_u1, verifier_tile_u1, softmax_abs_tolerance,
                            softmax_rel_tolerance) ||
          exceeds_tolerance(actual_u2, verifier_tile_u2, softmax_abs_tolerance,
                            softmax_rel_tolerance);
      if (bad) {
        atomicAdd(&stats->softmax_detected, 1);
        float threshold =
            softmax_abs_tolerance +
            softmax_rel_tolerance * fmaxf(1.0f, fabsf(verifier_u0));
        if (fabsf(delta0) > threshold) {
          float observed_weight = delta1 / delta0;
          int key_index =
              __float2int_rn(0.5f * (observed_weight + static_cast<float>(sequence - 1)));
          int local_key = key_index - key_begin;
          float second_residual =
              fabsf(delta2 / delta0 - observed_weight * observed_weight);
          if (local_key >= 0 && local_key < row_valid_keys &&
              fabsf(observed_weight - position_weight(key_index, sequence)) <= 0.25f &&
              second_residual <= fmaxf(1.0f, fabsf(observed_weight)) * 0.5f) {
            storage.recovery_action[row] = 1;
            storage.recovery_index[row] = local_key;
            storage.recovery_delta[row] = delta0;
            atomicAdd(&stats->softmax_corrected, 1);
          }
        }
        if (storage.recovery_action[row] == 0) {
          storage.recovery_action[row] = 2;
          atomicAdd(&stats->softmax_replayed, 1);
        }
      }
    }
    __syncthreads();

    if (storage.recovery_action[row] != 0) {
      int begin =
          storage.recovery_action[row] == 1 ? storage.recovery_index[row] : pair_lane;
      int end =
          storage.recovery_action[row] == 1 ? begin + 1 : row_valid_keys;
      int step = storage.recovery_action[row] == 1 ? 1 : 2;
      for (int local_key = begin; local_key < end; local_key += step) {
        float verifier_probability =
            __expf(storage.score_pv.score[local_key * kScoreStride + row] - new_verifier_m) *
            kProbabilityScale;
        sP(mma_row, local_key) = Element(verifier_probability);
      }
    }
    __syncthreads();

    actual_u0 = 0.0f;
    actual_u1 = 0.0f;
    actual_u2 = 0.0f;
    verifier_u0 = 0.0f;
    verifier_tile_u1 = 0.0f;
    verifier_tile_u2 = 0.0f;
    float tile_expected_output_sum = 0.0f;
    float tile_expected_output_weighted = 0.0f;
    for (int local_key = pair_lane; local_key < row_valid_keys; local_key += 2) {
      int key_index = key_begin + local_key;
      float probability = static_cast<float>(sP(mma_row, local_key));
      float weight = position_weight(key_index, sequence);
      float verifier_probability =
          static_cast<float>(Element(__expf(
              storage.score_pv.score[local_key * kScoreStride + row] - new_verifier_m) *
                                     kProbabilityScale));
      actual_u0 += probability;
      actual_u1 += weight * probability;
      actual_u2 += weight * weight * probability;
      verifier_u0 += verifier_probability;
      verifier_tile_u1 += weight * verifier_probability;
      verifier_tile_u2 += weight * weight * verifier_probability;
      size_t metadata_index = static_cast<size_t>(batch_head) * sequence + key_index;
      tile_expected_output_sum += probability * v_sum[metadata_index];
      tile_expected_output_weighted += probability * v_weighted[metadata_index];
    }
    actual_u0 += __shfl_xor_sync(0xffffffffU, actual_u0, 1, 2);
    actual_u1 += __shfl_xor_sync(0xffffffffU, actual_u1, 1, 2);
    actual_u2 += __shfl_xor_sync(0xffffffffU, actual_u2, 1, 2);
    verifier_u0 += __shfl_xor_sync(0xffffffffU, verifier_u0, 1, 2);
    verifier_tile_u1 +=
        __shfl_xor_sync(0xffffffffU, verifier_tile_u1, 1, 2);
    verifier_tile_u2 +=
        __shfl_xor_sync(0xffffffffU, verifier_tile_u2, 1, 2);
    tile_expected_output_sum +=
        __shfl_xor_sync(0xffffffffU, tile_expected_output_sum, 1, 2);
    tile_expected_output_weighted +=
        __shfl_xor_sync(0xffffffffU, tile_expected_output_weighted, 1, 2);

    if (pair_lane == 0) {
      storage.online_m[row] = new_m;
      storage.online_l[row] = alpha * storage.online_l[row] + actual_u0;
      storage.online_u1[row] = alpha * storage.online_u1[row] + actual_u1;
      storage.online_u2[row] = alpha * storage.online_u2[row] + actual_u2;
      storage.verifier_m[row] = new_verifier_m;
      storage.verifier_l[row] =
          verifier_alpha * storage.verifier_l[row] + verifier_u0;
      storage.verifier_u1[row] =
          verifier_alpha * storage.verifier_u1[row] + verifier_tile_u1;
      storage.verifier_u2[row] =
          verifier_alpha * storage.verifier_u2[row] + verifier_tile_u2;
      if (fault_stage == kFaultSoftmaxState && target_row &&
          fault_index >= key_begin && fault_index < key_begin + row_valid_keys) {
        storage.online_l[row] += fault_value * kProbabilityScale;
      }
      bool state_bad =
          exceeds_tolerance(storage.online_m[row], storage.verifier_m[row],
                            softmax_abs_tolerance, softmax_rel_tolerance) ||
          exceeds_tolerance(storage.online_l[row], storage.verifier_l[row],
                            softmax_abs_tolerance, softmax_rel_tolerance) ||
          exceeds_tolerance(storage.online_u1[row], storage.verifier_u1[row],
                            softmax_abs_tolerance, softmax_rel_tolerance) ||
          exceeds_tolerance(storage.online_u2[row], storage.verifier_u2[row],
                            softmax_abs_tolerance, softmax_rel_tolerance);
      if (state_bad) {
        atomicAdd(&stats->softmax_detected, 1);
        atomicAdd(&stats->softmax_replayed, 1);
        storage.online_m[row] = storage.verifier_m[row];
        storage.online_l[row] = storage.verifier_l[row];
        storage.online_u1[row] = storage.verifier_u1[row];
        storage.online_u2[row] = storage.verifier_u2[row];
      }
      storage.stage_alpha[row] = alpha;
      storage.expected_output_sum[row] =
          alpha * storage.expected_output_sum[row] + tile_expected_output_sum;
      storage.expected_output_weighted[row] =
          alpha * storage.expected_output_weighted[row] +
          tile_expected_output_weighted;
    }
    __syncthreads();

    typename Traits::PvTiledMma pv_mma;
    auto pv_thread = pv_mma.get_slice(consumer_thread);
    auto pv_accumulator = cute::partition_fragment_C(
        pv_mma, cute::make_shape(cute::_64{}, cute::Int<HeadDim>{}));
    cute::clear(pv_accumulator);
    cute::warpgroup_fence_operand(pv_accumulator);
    cute::warpgroup_arrive();
#pragma unroll
    for (int key_tile = 0; key_tile < kKeyTile / 32; ++key_tile) {
      auto p_tile = cute::local_tile(
          sP, cute::Shape<cute::_64, cute::_32>{}, cute::make_coord(0, key_tile));
      auto v_tile = cute::local_tile(
          sV, cute::Shape<cute::Int<HeadDim>, cute::_32>{},
          cute::make_coord(0, key_tile));
      auto p_descriptor = pv_thread.partition_A(p_tile);
      auto v_descriptor = pv_thread.partition_B(v_tile);
      auto p_fragment = pv_thread.make_fragment_A(p_descriptor);
      auto v_fragment = pv_thread.make_fragment_B(v_descriptor);
      cute::copy(p_descriptor, p_fragment);
      cute::copy(v_descriptor, v_fragment);
      pv_mma.accumulate_ =
          key_tile == 0 ? cute::GMMA::ScaleOut::Zero : cute::GMMA::ScaleOut::One;
      cute::gemm(pv_mma, p_fragment, v_fragment, pv_accumulator);
    }
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
    cute::warpgroup_fence_operand(pv_accumulator);
    auto pv_destination = pv_thread.partition_C(sPv);
    cute::copy(pv_accumulator, pv_destination);
    __syncthreads();

    for (int index = tid; index < kQueryTile * HeadDim; index += kThreads) {
      int local_row = index % kQueryTile;
      int dim = index / kQueryTile;
      storage.output[dim * kScoreStride + local_row] =
          storage.stage_alpha[local_row] *
              storage.output[dim * kScoreStride + local_row] +
          storage.score_pv.pv[dim * kScoreStride + local_row];
    }
    __syncthreads();
  }

  for (int dim = pair_lane; dim < HeadDim; dim += 2) {
    float value =
        valid_query ? storage.output[dim * kScoreStride + row] / storage.online_l[row]
                    : 0.0f;
    if (fault_stage == kFaultOutput && target_row && dim == fault_index) {
      value += fault_value;
    }
    storage.output[dim * kScoreStride + row] = value;
  }
  __syncthreads();

  float actual_output_sum = 0.0f;
  float actual_output_weighted = 0.0f;
  for (int dim = pair_lane; dim < HeadDim; dim += 2) {
    float value = storage.output[dim * kScoreStride + row];
    actual_output_sum += value;
    actual_output_weighted += position_weight(dim, HeadDim) * value;
  }
  actual_output_sum += __shfl_xor_sync(0xffffffffU, actual_output_sum, 1, 2);
  actual_output_weighted +=
      __shfl_xor_sync(0xffffffffU, actual_output_weighted, 1, 2);
  if (pair_lane == 0) {
    storage.recovery_action[row] = 0;
    storage.recovery_index[row] = -1;
    storage.recovery_delta[row] = 0.0f;
    if (valid_query) {
      float expected_sum =
          storage.expected_output_sum[row] / storage.online_l[row];
      float expected_weighted =
          storage.expected_output_weighted[row] / storage.online_l[row];
      float delta_sum = actual_output_sum - expected_sum;
      float delta_weighted = actual_output_weighted - expected_weighted;
      bool bad =
          exceeds_tolerance(actual_output_sum, expected_sum, pv_abs_tolerance,
                            pv_rel_tolerance) ||
          exceeds_tolerance(actual_output_weighted, expected_weighted, pv_abs_tolerance,
                            pv_rel_tolerance);
      if (bad) {
        atomicAdd(&stats->pv_detected, 1);
        float threshold =
            pv_abs_tolerance + pv_rel_tolerance * fmaxf(1.0f, fabsf(expected_sum));
        if (fabsf(delta_sum) > threshold) {
          float observed_weight = delta_weighted / delta_sum;
          int dim = __float2int_rn(
              0.5f * (observed_weight + static_cast<float>(HeadDim - 1)));
          if (dim >= 0 && dim < HeadDim &&
              fabsf(observed_weight - position_weight(dim, HeadDim)) <= 0.25f) {
            storage.recovery_action[row] = 1;
            storage.recovery_index[row] = dim;
            storage.recovery_delta[row] = delta_sum;
            atomicAdd(&stats->pv_corrected, 1);
          }
        }
        if (storage.recovery_action[row] == 0) {
          atomicAdd(&stats->failed_rows, 1);
        }
      }
    }
  }
  __syncthreads();

  if (storage.recovery_action[row] == 1 && pair_lane == 0) {
    int dim = storage.recovery_index[row];
    storage.output[dim * kScoreStride + row] -= storage.recovery_delta[row];
  }
  __syncthreads();

  for (int dim = pair_lane; dim < HeadDim; dim += 2) {
    if (valid_query) {
      output[qkv_offset(batch_id, head, query, dim, heads, sequence, HeadDim)] =
          storage.output[dim * kScoreStride + row];
    }
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

  dim3 grid((options.sequence + kQueryTile - 1) / kQueryTile, options.batch * options.heads);
  float score_scale = 1.0f / std::sqrt(static_cast<float>(options.head_dim));
  if (options.head_dim == 64) {
    CUDA_CHECK(cudaFuncSetAttribute(ft_flash_attention_a10_kernel<64>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    sizeof(AttentionSharedStorage<64>)));
  } else {
    CUDA_CHECK(cudaFuncSetAttribute(ft_flash_attention_a10_kernel<128>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    sizeof(AttentionSharedStorage<128>)));
  }
  auto launch = [&](FaultStage fault_stage, cudaStream_t stream) {
    if (options.head_dim == 64) {
      ft_flash_attention_a10_kernel<64>
          <<<grid, kThreads, sizeof(AttentionSharedStorage<64>), stream>>>(
          d_q, d_k, d_v, d_output, options.batch, options.heads,
          d_k_sum, d_k_weighted, d_k_abs_sum, d_k_weighted_abs_sum, d_v_sum, d_v_weighted,
          options.sequence, key_tiles, options.causal, score_scale, static_cast<int>(fault_stage), options.fault_batch, options.fault_head,
          options.fault_query, options.fault_index, options.fault_value, options.score_abs_tolerance,
          options.score_rel_tolerance, options.softmax_abs_tolerance, options.softmax_rel_tolerance,
          options.pv_abs_tolerance, options.pv_rel_tolerance, d_stats);
    } else {
      ft_flash_attention_a10_kernel<128>
          <<<grid, kThreads, sizeof(AttentionSharedStorage<128>), stream>>>(
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

  std::printf("version: FT-FlashAttention A10 two-consumer-warpgroup WGMMA\n");
  std::printf("layout: Q/K/V=[B,H,N,D], FP8 E4M3; O=FP32\n");
  std::printf("shape: B=%d H=%d N=%d D=%d causal=%d\n", options.batch, options.heads,
              options.sequence, options.head_dim, options.causal ? 1 : 0);
  std::printf("query_tile: %d, key_tile: %d\n", kQueryTile, kKeyTile);
  std::printf("shared_storage_bytes: %zu\n", options.head_dim == 64
                                                   ? sizeof(AttentionSharedStorage<64>)
                                                   : sizeof(AttentionSharedStorage<128>));
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
