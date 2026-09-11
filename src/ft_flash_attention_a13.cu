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
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#define CUDA_CHECK(call)                                                                             \
  do {                                                                                               \
    cudaError_t error = (call);                                                                      \
    if (error != cudaSuccess) {                                                                      \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      std::exit(EXIT_FAILURE);                                                                       \
    }                                                                                                \
  } while (0)

using Element = cutlass::float_e4m3_t;

constexpr int kThreads = 128;
constexpr int kQueryTile = 64;
constexpr int kKeyTile = 128;
constexpr float kProbabilityScale = 256.0f;

using QkMmaShape = cute::Shape<cute::_64, cute::_128, cute::_32>;
using QkGmmaOp = decltype(cute::GMMA::ss_op_selector<Element, Element, float, QkMmaShape,
                                                      cute::GMMA::Major::K, cute::GMMA::Major::K>());
using QkTiledMma = decltype(cute::make_tiled_mma(QkGmmaOp{}));

template <int HeadDim>
struct AttentionTraits {
  using PvMmaShape = cute::Shape<cute::_64, cute::Int<HeadDim>, cute::_32>;
  using PvGmmaOp = decltype(cute::GMMA::rs_op_selector<Element, Element, float, PvMmaShape,
                                                         cute::GMMA::Major::K, cute::GMMA::Major::K>());
  using PvTiledMma = decltype(cute::make_tiled_mma(PvGmmaOp{}));
  using SmemLayoutQ = decltype(cute::tile_to_shape(
      cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
      cute::Shape<cute::_64, cute::Int<HeadDim>>{}));
  using SmemLayoutK = decltype(cute::tile_to_shape(
      cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
      cute::Shape<cute::_128, cute::Int<HeadDim>>{}));
  using SmemLayoutV = decltype(cute::tile_to_shape(
      cute::SM90::GMMA::Layout_K_INTER_Atom<Element>{},
      cute::Shape<cute::Int<HeadDim>, cute::_128>{}));
};

template <int HeadDim>
struct alignas(128) AttentionSharedStorage {
  using Traits = AttentionTraits<HeadDim>;
  alignas(128) Element q[cute::cosize_v<typename Traits::SmemLayoutQ>];
  alignas(128) Element k[cute::cosize_v<typename Traits::SmemLayoutK>];
  alignas(128) Element v[cute::cosize_v<typename Traits::SmemLayoutV>];
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

__device__ __forceinline__ float row_group_sum(float value) {
  value += __shfl_xor_sync(0xffffffffU, value, 2, 4);
  value += __shfl_xor_sync(0xffffffffU, value, 1, 4);
  return value;
}

__device__ __forceinline__ float row_group_sum(float value, unsigned mask) {
  value += __shfl_xor_sync(mask, value, 2, 4);
  value += __shfl_xor_sync(mask, value, 1, 4);
  return value;
}

__device__ __forceinline__ float row_group_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, 2, 4));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, 1, 4));
  return value;
}

template <typename AccLayout>
CUTLASS_DEVICE auto accumulator_rowcol_layout(AccLayout layout) {
  using namespace cute;
  static_assert(decltype(rank<0>(layout))::value == 3);
  static_assert(decltype(size<0, 0>(layout))::value == 2);
  static_assert(decltype(size<0, 1>(layout))::value == 2);
  return make_layout(make_layout(get<0, 1>(layout), get<1>(layout)),
                     make_layout(get<0, 0>(layout), get<0, 2>(layout), get<2>(layout)));
}

template <typename MmaTraits, typename AccLayout>
CUTLASS_DEVICE auto accumulator_a_operand_layout(AccLayout layout) {
  using namespace cute;
  static_assert(decltype(rank<0>(layout))::value == 3);
  static_assert(decltype(size<0, 0>(layout))::value == 2);
  static_assert(decltype(size<0, 1>(layout))::value == 2);
  static_assert(sizeof(typename MmaTraits::ValTypeA) == 1);
  auto divided = logical_divide(get<0, 2>(layout), Tile<cute::Layout<Shape<_2, _2>>>{});
  return make_layout(make_layout(cute::Layout<_4>{}, get<0, 0, 0>(divided),
                                 get<0, 0, 1>(divided)),
                     get<1>(layout),
                     coalesce(make_layout(get<0, 1>(divided), get<2>(layout))));
}

template <typename InputEngine, typename Layout, typename OutputEngine>
CUTLASS_DEVICE void convert_fragment(
    cute::Tensor<InputEngine, Layout> const& input,
    cute::Tensor<OutputEngine, Layout>& output) {
  using Input = typename InputEngine::value_type;
  using Output = typename OutputEngine::value_type;
  constexpr int kFragmentSize =
      std::max(sizeof(Input) / sizeof(Output), sizeof(Output) / sizeof(Input));
  auto input_arrays =
      cute::recast<cutlass::Array<Input, kFragmentSize> const>(input);
  auto output_arrays = cute::recast<cutlass::Array<Output, kFragmentSize>>(output);
  cutlass::NumericArrayConverter<Output, Input, kFragmentSize> convert;
#pragma unroll
  for (int index = 0; index < cute::size(input_arrays); ++index) {
    output_arrays[index] = convert(input_arrays[index]);
  }
}

template <typename Fragment>
CUTLASS_DEVICE void permute_probability_fragment(Fragment& fragment) {
  using namespace cute;
  auto fragment_64 = group_modes<1, 3>(recast<uint2>(fragment));
#pragma unroll
  for (int row = 0; row < size<1>(fragment_64); ++row) {
#pragma unroll
    for (int index = 0; index < size<0, 2>(fragment_64) / 2; ++index) {
      cutlass::swap(fragment_64(make_coord(_0{}, _1{}, 2 * index), row),
                    fragment_64(make_coord(_0{}, _0{}, 2 * index + 1), row));
    }
  }
}

template <typename Fragment>
CUTLASS_DEVICE void permute_probability_operand(Fragment& fragment) {
  using namespace cute;
  int quad_lane = threadIdx.x % 4;
  bool lane_03 = quad_lane == 0 || quad_lane == 3;
  int selector_upper = lane_03 ? 0x5410 : 0x1054;
  int selector_lower = lane_03 ? 0x7632 : 0x3276;
  constexpr int kUpperMap[4] = {0, 3, 1, 2};
  auto fragment_64 = recast<uint2>(fragment);
#pragma unroll
  for (int index = 0; index < size(fragment_64); ++index) {
    uint32_t upper = fragment_64[index].x;
    uint32_t lower = fragment_64[index].y;
    uint32_t upper0 = lane_03 ? upper : lower;
    uint32_t lower0 = lane_03 ? lower : upper;
    upper0 = __shfl_sync(0xffffffffU, upper0, kUpperMap[quad_lane], 4);
    lower0 = __shfl_sync(0xffffffffU, lower0, kUpperMap[quad_lane] ^ 1, 4);
    fragment_64[index].x = __byte_perm(upper0, lower0, selector_upper);
    fragment_64[index].y = __byte_perm(upper0, lower0, selector_lower);
  }
}

template <typename Fragment>
CUTLASS_DEVICE void permute_output_fragment(Fragment& fragment) {
  using namespace cute;
  auto grouped = group_modes<1, 3>(fragment);
#pragma unroll
  for (int row = 0; row < size<1>(grouped); ++row) {
#pragma unroll
    for (int middle = 0; middle < size<0, 1>(grouped); ++middle) {
#pragma unroll
      for (int index = 0; index < size<0, 2>(grouped) / 2; ++index) {
        cutlass::swap(grouped(make_coord(_1{}, middle, 2 * index), row),
                      grouped(make_coord(_0{}, middle, 2 * index + 1), row));
      }
    }
  }
}

template <bool ZeroInit, typename Mma, typename TensorA, typename TensorB, typename TensorC>
CUTLASS_DEVICE void warpgroup_gemm(
    Mma& mma, TensorA const& a, TensorB const& b, TensorC& c) {
  constexpr bool kRegisterA =
      !cute::is_base_of<cute::GMMA::DescriptorIterator,
                        typename Mma::FrgTypeA>::value;
  cute::warpgroup_fence_operand(c);
  if constexpr (kRegisterA) {
    cute::warpgroup_fence_operand(const_cast<TensorA&>(a));
  }
  cute::warpgroup_arrive();
  if constexpr (ZeroInit) {
    mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
  }
#pragma unroll
  for (int k_block = 0; k_block < cute::size<2>(a); ++k_block) {
    cute::gemm(mma, a(cute::_, cute::_, k_block), b(cute::_, cute::_, k_block), c);
    mma.accumulate_ = cute::GMMA::ScaleOut::One;
  }
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
  cute::warpgroup_fence_operand(c);
  if constexpr (kRegisterA) {
    cute::warpgroup_fence_operand(const_cast<TensorA&>(a));
  }
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
__global__ __launch_bounds__(kThreads) void ft_flash_attention_a13_kernel(
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
  auto sQ = cute::make_tensor(cute::make_smem_ptr(storage.q), typename Traits::SmemLayoutQ{});
  auto sK = cute::make_tensor(cute::make_smem_ptr(storage.k), typename Traits::SmemLayoutK{});
  auto sV = cute::make_tensor(cute::make_smem_ptr(storage.v), typename Traits::SmemLayoutV{});

  int tid = threadIdx.x;
  int query_begin = blockIdx.x * kQueryTile;
  int batch_head = blockIdx.y;
  int batch_id = batch_head / heads;
  int head = batch_head % heads;
  int quad_lane = tid & 3;

  for (int index = tid; index < kQueryTile * HeadDim; index += kThreads) {
    int local_row = index / HeadDim;
    int dim = index % HeadDim;
    int query = query_begin + local_row;
    sQ(local_row, dim) = query < sequence
        ? q[qkv_offset(batch_id, head, query, dim, heads, sequence, HeadDim)]
        : Element(0.0f);
  }
  __syncthreads();

  QkTiledMma qk_mma;
  typename Traits::PvTiledMma pv_mma;
  auto qk_thread = qk_mma.get_slice(tid);
  auto pv_thread = pv_mma.get_slice(tid);
  auto q_descriptor = qk_thread.partition_fragment_A(sQ);
  auto k_descriptor = qk_thread.partition_fragment_B(sK);
  auto v_descriptor = pv_thread.partition_fragment_B(sV);
  auto output_accumulator = cute::partition_fragment_C(
      pv_mma, cute::make_shape(cute::_64{}, cute::Int<HeadDim>{}));
  cute::clear(output_accumulator);
  auto output_view = cute::make_tensor(
      output_accumulator.data(), accumulator_rowcol_layout(output_accumulator.layout()));
  auto score_coordinates_raw = qk_thread.partition_C(
      cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::_128{})));
  auto score_coordinates = cute::make_tensor(
      score_coordinates_raw.data(), accumulator_rowcol_layout(score_coordinates_raw.layout()));
  auto output_coordinates_raw = pv_thread.partition_C(
      cute::make_identity_tensor(cute::make_shape(cute::_64{}, cute::Int<HeadDim>{})));
  auto output_coordinates = cute::make_tensor(
      output_coordinates_raw.data(), accumulator_rowcol_layout(output_coordinates_raw.layout()));
  constexpr int kRowsPerThread = decltype(cute::size<0>(score_coordinates))::value;

  float online_m[kRowsPerThread], online_l[kRowsPerThread];
  float verifier_m[kRowsPerThread], verifier_l[kRowsPerThread];
  float online_u1[kRowsPerThread], online_u2[kRowsPerThread];
  float verifier_u1[kRowsPerThread], verifier_u2[kRowsPerThread];
  float expected_output_sum[kRowsPerThread], expected_output_weighted[kRowsPerThread];
#pragma unroll
  for (int local = 0; local < kRowsPerThread; ++local) {
    online_m[local] = verifier_m[local] = -FLT_MAX;
    online_l[local] = verifier_l[local] = 0.0f;
    online_u1[local] = online_u2[local] = 0.0f;
    verifier_u1[local] = verifier_u2[local] = 0.0f;
    expected_output_sum[local] = expected_output_weighted[local] = 0.0f;
  }

  int key_limit = causal ? min(sequence, query_begin + kQueryTile) : sequence;
  for (int key_begin = 0; key_begin < key_limit; key_begin += kKeyTile) {
    int tile_keys = min(kKeyTile, sequence - key_begin);
    for (int index = tid; index < kKeyTile * HeadDim; index += kThreads) {
      int local_key = index / HeadDim;
      int dim = index % HeadDim;
      int key_index = key_begin + local_key;
      sK(local_key, dim) = key_index < sequence
          ? k[qkv_offset(batch_id, head, key_index, dim, heads, sequence, HeadDim)]
          : Element(0.0f);
      sV(dim, local_key) = key_index < sequence
          ? v[qkv_offset(batch_id, head, key_index, dim, heads, sequence, HeadDim)]
          : Element(0.0f);
    }
    __syncthreads();

    float expected_score_sum[kRowsPerThread];
    float expected_score_weighted[kRowsPerThread];
    float expected_score_scale[kRowsPerThread];
    float expected_score_weighted_scale[kRowsPerThread];
#pragma unroll
    for (int local = 0; local < kRowsPerThread; ++local) {
      int row = cute::get<0>(score_coordinates(local, 0));
      int query = query_begin + row;
      int valid_keys = query < sequence ? tile_keys : 0;
      if (causal) { valid_keys = min(valid_keys, max(0, query + 1 - key_begin)); }
      float sum = 0.0f, weighted = 0.0f, magnitude = 0.0f, weighted_magnitude = 0.0f;
      if (valid_keys > 0) {
        int full_tile_keys = min(kKeyTile, sequence - key_begin);
        for (int dim = quad_lane; dim < HeadDim; dim += 4) {
          float qvalue = static_cast<float>(sQ(row, dim));
          float key_sum = 0.0f, key_weighted = 0.0f;
          float key_magnitude = 0.0f, key_weighted_magnitude = 0.0f;
          if (valid_keys == full_tile_keys) {
            int key_tile = key_begin / kKeyTile;
            size_t metadata_index =
                (static_cast<size_t>(batch_head) * key_tiles + key_tile) * HeadDim + dim;
            key_sum = k_sum[metadata_index];
            key_weighted = k_weighted[metadata_index];
            key_magnitude = k_abs_sum[metadata_index];
            key_weighted_magnitude = k_weighted_abs_sum[metadata_index];
          } else {
            for (int local_key = 0; local_key < valid_keys; ++local_key) {
              float key_value = static_cast<float>(sK(local_key, dim));
              float weight = position_weight(key_begin + local_key, sequence);
              key_sum += key_value;
              key_weighted += weight * key_value;
              key_magnitude += fabsf(key_value);
              key_weighted_magnitude += fabsf(weight) * fabsf(key_value);
            }
          }
          sum += qvalue * key_sum * score_scale;
          weighted += qvalue * key_weighted * score_scale;
          magnitude += fabsf(qvalue) * key_magnitude * fabsf(score_scale);
          weighted_magnitude += fabsf(qvalue) * key_weighted_magnitude * fabsf(score_scale);
        }
      }
      expected_score_sum[local] = row_group_sum(sum);
      expected_score_weighted[local] = row_group_sum(weighted);
      expected_score_scale[local] = row_group_sum(magnitude);
      expected_score_weighted_scale[local] = row_group_sum(weighted_magnitude);
    }

    auto score_accumulator =
        cute::partition_fragment_C(qk_mma, cute::make_shape(cute::_64{}, cute::_128{}));
    warpgroup_gemm<true>(qk_mma, q_descriptor, k_descriptor, score_accumulator);
    auto score_view = cute::make_tensor(
        score_accumulator.data(), accumulator_rowcol_layout(score_accumulator.layout()));
    auto verifier_score = cute::make_fragment_like(score_accumulator);
    cute::copy(score_accumulator, verifier_score);
    auto verifier_score_view = cute::make_tensor(
        verifier_score.data(), accumulator_rowcol_layout(verifier_score.layout()));

#pragma unroll
    for (int local = 0; local < kRowsPerThread; ++local) {
      int row = cute::get<0>(score_coordinates(local, 0));
      int query = query_begin + row;
      float actual_sum = 0.0f, actual_weighted = 0.0f;
#pragma unroll
      for (int column = 0; column < cute::size<1>(score_view); ++column) {
        int local_key = cute::get<1>(score_coordinates(local, column));
        int key_index = key_begin + local_key;
        bool active = query < sequence && key_index < sequence && (!causal || key_index <= query);
        float score = active ? score_view(local, column) * score_scale : -FLT_MAX;
        if (active && fault_stage == kFaultScore && batch_id == fault_batch &&
            head == fault_head && query == fault_query && key_index == fault_index) {
          score += fault_value;
        }
        score_view(local, column) = score;
        verifier_score_view(local, column) = active
            ? verifier_score_view(local, column) * score_scale : -FLT_MAX;
        if (active) {
          actual_sum += score;
          actual_weighted += position_weight(key_index, sequence) * score;
        }
      }
      actual_sum = row_group_sum(actual_sum);
      actual_weighted = row_group_sum(actual_weighted);
      float delta_sum = actual_sum - expected_score_sum[local];
      float delta_weighted = actual_weighted - expected_score_weighted[local];
      float sum_threshold = score_abs_tolerance +
          score_rel_tolerance * fmaxf(1.0f, expected_score_scale[local]);
      float weighted_threshold = score_abs_tolerance +
          score_rel_tolerance * fmaxf(1.0f, expected_score_weighted_scale[local]);
      int recovery_action = 0;
      int recovery_key = -1;
      float recovery_delta = 0.0f;
      bool bad = fabsf(delta_sum) > sum_threshold ||
                 fabsf(delta_weighted) > weighted_threshold;
      if (quad_lane == 0 && bad) {
        atomicAdd(&stats->score_detected, 1);
        if (fabsf(delta_sum) > sum_threshold) {
          float observed_weight = delta_weighted / delta_sum;
          int key_index = __float2int_rn(
              0.5f * (observed_weight + static_cast<float>(sequence - 1)));
          if (key_index >= key_begin && key_index < key_begin + tile_keys &&
              (!causal || key_index <= query) &&
              fabsf(observed_weight - position_weight(key_index, sequence)) <= 0.25f) {
            recovery_action = 1;
            recovery_key = key_index - key_begin;
            recovery_delta = delta_sum;
            atomicAdd(&stats->score_corrected, 1);
          }
        }
        if (recovery_action == 0) {
          recovery_action = 2;
          atomicAdd(&stats->score_replayed, 1);
        }
      }
      recovery_action = __shfl_sync(0xffffffffU, recovery_action, 0, 4);
      recovery_key = __shfl_sync(0xffffffffU, recovery_key, 0, 4);
      recovery_delta = __shfl_sync(0xffffffffU, recovery_delta, 0, 4);
      if (recovery_action != 0) {
#pragma unroll
        for (int column = 0; column < cute::size<1>(score_view); ++column) {
          int local_key = cute::get<1>(score_coordinates(local, column));
          if (recovery_action == 1 && local_key == recovery_key) {
            score_view(local, column) -= recovery_delta;
          } else if (recovery_action == 2) {
            int key_index = key_begin + local_key;
            if (query < sequence && key_index < sequence && (!causal || key_index <= query)) {
              float replay = 0.0f;
              for (int dim = 0; dim < HeadDim; ++dim) {
                replay += static_cast<float>(sQ(row, dim)) *
                          static_cast<float>(sK(local_key, dim));
              }
              score_view(local, column) = replay * score_scale;
            }
          }
        }
      }
    }

    float stage_alpha[kRowsPerThread];
#pragma unroll
    for (int local = 0; local < kRowsPerThread; ++local) {
      int row = cute::get<0>(score_coordinates(local, 0));
      int query = query_begin + row;
      float primary_max = -FLT_MAX, verifier_max_value = -FLT_MAX;
#pragma unroll
      for (int column = 0; column < cute::size<1>(score_view); ++column) {
        primary_max = fmaxf(primary_max, score_view(local, column));
        verifier_max_value = fmaxf(verifier_max_value, verifier_score_view(local, column));
      }
      primary_max = row_group_max(primary_max);
      verifier_max_value = row_group_max(verifier_max_value);
      float new_m = fmaxf(online_m[local], primary_max);
      float new_verifier_m = fmaxf(verifier_m[local], verifier_max_value);
      float alpha = online_m[local] == -FLT_MAX ? 0.0f : __expf(online_m[local] - new_m);
      float verifier_alpha = verifier_m[local] == -FLT_MAX
          ? 0.0f : __expf(verifier_m[local] - new_verifier_m);
      stage_alpha[local] = alpha;
      float actual_u0 = 0.0f, actual_u1 = 0.0f, actual_u2 = 0.0f;
      float check_u0 = 0.0f, check_u1 = 0.0f, check_u2 = 0.0f;
#pragma unroll
      for (int column = 0; column < cute::size<1>(score_view); ++column) {
        int local_key = cute::get<1>(score_coordinates(local, column));
        int key_index = key_begin + local_key;
        bool active = query < sequence && key_index < sequence && (!causal || key_index <= query);
        float primary_probability = active
            ? __expf(score_view(local, column) - new_m) * kProbabilityScale : 0.0f;
        float verifier_probability = active
            ? __expf(verifier_score_view(local, column) - new_verifier_m) * kProbabilityScale : 0.0f;
        if (active && fault_stage == kFaultSoftmax && batch_id == fault_batch &&
            head == fault_head && query == fault_query && key_index == fault_index) {
          primary_probability += fault_value * kProbabilityScale;
        }
        float actual = static_cast<float>(Element(primary_probability));
        float expected = static_cast<float>(Element(verifier_probability));
        score_view(local, column) = actual;
        verifier_score_view(local, column) = expected;
        if (active) {
          float weight = position_weight(key_index, sequence);
          actual_u0 += actual;
          actual_u1 += weight * actual;
          actual_u2 += weight * weight * actual;
          check_u0 += expected;
          check_u1 += weight * expected;
          check_u2 += weight * weight * expected;
        }
      }
      actual_u0 = row_group_sum(actual_u0);
      actual_u1 = row_group_sum(actual_u1);
      actual_u2 = row_group_sum(actual_u2);
      check_u0 = row_group_sum(check_u0);
      check_u1 = row_group_sum(check_u1);
      check_u2 = row_group_sum(check_u2);
      float delta0 = actual_u0 - check_u0;
      float delta1 = actual_u1 - check_u1;
      float delta2 = actual_u2 - check_u2;
      int recovery_action = 0, recovery_key = -1;
      bool bad = exceeds_tolerance(actual_u0, check_u0, softmax_abs_tolerance, softmax_rel_tolerance) ||
                 exceeds_tolerance(actual_u1, check_u1, softmax_abs_tolerance, softmax_rel_tolerance) ||
                 exceeds_tolerance(actual_u2, check_u2, softmax_abs_tolerance, softmax_rel_tolerance);
      if (quad_lane == 0 && bad) {
        atomicAdd(&stats->softmax_detected, 1);
        float threshold = softmax_abs_tolerance +
            softmax_rel_tolerance * fmaxf(1.0f, fabsf(check_u0));
        if (fabsf(delta0) > threshold) {
          float observed_weight = delta1 / delta0;
          int key_index = __float2int_rn(
              0.5f * (observed_weight + static_cast<float>(sequence - 1)));
          float second_residual = fabsf(delta2 / delta0 - observed_weight * observed_weight);
          if (key_index >= key_begin && key_index < key_begin + tile_keys &&
              (!causal || key_index <= query) &&
              fabsf(observed_weight - position_weight(key_index, sequence)) <= 0.25f &&
              second_residual <= fmaxf(1.0f, fabsf(observed_weight)) * 0.5f) {
            recovery_action = 1;
            recovery_key = key_index - key_begin;
            atomicAdd(&stats->softmax_corrected, 1);
          }
        }
        if (recovery_action == 0) {
          recovery_action = 2;
          atomicAdd(&stats->softmax_replayed, 1);
        }
      }
      recovery_action = __shfl_sync(0xffffffffU, recovery_action, 0, 4);
      recovery_key = __shfl_sync(0xffffffffU, recovery_key, 0, 4);
      if (recovery_action != 0) {
        unsigned recovery_mask = __activemask();
        actual_u0 = actual_u1 = actual_u2 = 0.0f;
#pragma unroll
        for (int column = 0; column < cute::size<1>(score_view); ++column) {
          int local_key = cute::get<1>(score_coordinates(local, column));
          if (recovery_action == 2 || local_key == recovery_key) {
            score_view(local, column) = verifier_score_view(local, column);
          }
          int key_index = key_begin + local_key;
          bool active = query < sequence && key_index < sequence && (!causal || key_index <= query);
          if (active) {
            float probability = score_view(local, column);
            float weight = position_weight(key_index, sequence);
            actual_u0 += probability;
            actual_u1 += weight * probability;
            actual_u2 += weight * weight * probability;
          }
        }
        actual_u0 = row_group_sum(actual_u0, recovery_mask);
        actual_u1 = row_group_sum(actual_u1, recovery_mask);
        actual_u2 = row_group_sum(actual_u2, recovery_mask);
      }
      online_m[local] = new_m;
      online_l[local] = alpha * online_l[local] + actual_u0;
      online_u1[local] = alpha * online_u1[local] + actual_u1;
      online_u2[local] = alpha * online_u2[local] + actual_u2;
      verifier_m[local] = new_verifier_m;
      verifier_l[local] = verifier_alpha * verifier_l[local] + check_u0;
      verifier_u1[local] = verifier_alpha * verifier_u1[local] + check_u1;
      verifier_u2[local] = verifier_alpha * verifier_u2[local] + check_u2;
      if (fault_stage == kFaultSoftmaxState && batch_id == fault_batch &&
          head == fault_head && query == fault_query && fault_index >= key_begin &&
          fault_index < key_begin + tile_keys) {
        online_l[local] += fault_value * kProbabilityScale;
      }
      bool state_bad = exceeds_tolerance(online_m[local], verifier_m[local], softmax_abs_tolerance, softmax_rel_tolerance) ||
                       exceeds_tolerance(online_l[local], verifier_l[local], softmax_abs_tolerance, softmax_rel_tolerance) ||
                       exceeds_tolerance(online_u1[local], verifier_u1[local], softmax_abs_tolerance, softmax_rel_tolerance) ||
                       exceeds_tolerance(online_u2[local], verifier_u2[local], softmax_abs_tolerance, softmax_rel_tolerance);
      if (state_bad) {
        if (quad_lane == 0) {
          atomicAdd(&stats->softmax_detected, 1);
          atomicAdd(&stats->softmax_replayed, 1);
        }
        online_m[local] = verifier_m[local];
        online_l[local] = verifier_l[local];
        online_u1[local] = verifier_u1[local];
        online_u2[local] = verifier_u2[local];
      }
      float tile_expected_sum = 0.0f, tile_expected_weighted = 0.0f;
#pragma unroll
      for (int column = 0; column < cute::size<1>(score_view); ++column) {
        int local_key = cute::get<1>(score_coordinates(local, column));
        int key_index = key_begin + local_key;
        if (query < sequence && key_index < sequence && (!causal || key_index <= query)) {
          float probability = score_view(local, column);
          size_t metadata_index = static_cast<size_t>(batch_head) * sequence + key_index;
          tile_expected_sum += probability * v_sum[metadata_index];
          tile_expected_weighted += probability * v_weighted[metadata_index];
        }
      }
      tile_expected_sum = row_group_sum(tile_expected_sum);
      tile_expected_weighted = row_group_sum(tile_expected_weighted);
      expected_output_sum[local] = alpha * expected_output_sum[local] + tile_expected_sum;
      expected_output_weighted[local] =
          alpha * expected_output_weighted[local] + tile_expected_weighted;
    }

#pragma unroll
    for (int local = 0; local < kRowsPerThread; ++local) {
#pragma unroll
      for (int column = 0; column < cute::size<1>(output_view); ++column) {
        output_view(local, column) *= stage_alpha[local];
      }
    }
    auto probability_accumulator = cute::make_tensor(
        score_accumulator.data(),
        accumulator_a_operand_layout<typename Traits::PvTiledMma>(score_accumulator.layout()));
    auto probability_fragment = cute::make_tensor_like<Element>(probability_accumulator);
    convert_fragment(probability_accumulator, probability_fragment);
    permute_probability_operand(probability_fragment);
    pv_mma.accumulate_ = cute::GMMA::ScaleOut::One;
    warpgroup_gemm<false>(pv_mma, probability_fragment, v_descriptor, output_accumulator);
    __syncthreads();
  }

  static_assert(decltype(cute::size<0>(output_view))::value == kRowsPerThread);
#pragma unroll
  for (int local = 0; local < kRowsPerThread; ++local) {
    int row = cute::get<0>(output_coordinates(local, 0));
    int query = query_begin + row;
    bool valid_query = query < sequence;
    float inverse_sum = valid_query && online_l[local] > 0.0f ? 1.0f / online_l[local] : 0.0f;
    float actual_sum = 0.0f, actual_weighted = 0.0f;
#pragma unroll
    for (int column = 0; column < cute::size<1>(output_view); ++column) {
      int dim = cute::get<1>(output_coordinates(local, column));
      float value = output_view(local, column) * inverse_sum;
      if (valid_query && fault_stage == kFaultOutput && batch_id == fault_batch &&
          head == fault_head && query == fault_query && dim == fault_index) {
        value += fault_value;
      }
      output_view(local, column) = value;
      if (valid_query) {
        actual_sum += value;
        actual_weighted += position_weight(dim, HeadDim) * value;
      }
    }
    actual_sum = row_group_sum(actual_sum);
    actual_weighted = row_group_sum(actual_weighted);
    float expected_sum = expected_output_sum[local] * inverse_sum;
    float expected_weighted = expected_output_weighted[local] * inverse_sum;
    float delta_sum = actual_sum - expected_sum;
    float delta_weighted = actual_weighted - expected_weighted;
    int recovery_action = 0, recovery_dim = -1;
    bool bad = valid_query &&
        (exceeds_tolerance(actual_sum, expected_sum, pv_abs_tolerance, pv_rel_tolerance) ||
         exceeds_tolerance(actual_weighted, expected_weighted, pv_abs_tolerance, pv_rel_tolerance));
    if (quad_lane == 0 && bad) {
      atomicAdd(&stats->pv_detected, 1);
      float threshold = pv_abs_tolerance + pv_rel_tolerance * fmaxf(1.0f, fabsf(expected_sum));
      if (fabsf(delta_sum) > threshold) {
        float observed_weight = delta_weighted / delta_sum;
        int dim = __float2int_rn(0.5f * (observed_weight + static_cast<float>(HeadDim - 1)));
        if (dim >= 0 && dim < HeadDim &&
            fabsf(observed_weight - position_weight(dim, HeadDim)) <= 0.25f) {
          recovery_action = 1;
          recovery_dim = dim;
          atomicAdd(&stats->pv_corrected, 1);
        }
      }
      if (recovery_action == 0) { atomicAdd(&stats->failed_rows, 1); }
    }
    recovery_action = __shfl_sync(0xffffffffU, recovery_action, 0, 4);
    recovery_dim = __shfl_sync(0xffffffffU, recovery_dim, 0, 4);
#pragma unroll
    for (int column = 0; column < cute::size<1>(output_view); ++column) {
      int dim = cute::get<1>(output_coordinates(local, column));
      if (recovery_action == 1 && dim == recovery_dim) {
        output_view(local, column) -= delta_sum;
      }
      if (valid_query) {
        output[qkv_offset(batch_id, head, query, dim, heads, sequence, HeadDim)] =
            output_view(local, column);
      }
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
    CUDA_CHECK(cudaFuncSetAttribute(ft_flash_attention_a13_kernel<64>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    sizeof(AttentionSharedStorage<64>)));
  } else {
    CUDA_CHECK(cudaFuncSetAttribute(ft_flash_attention_a13_kernel<128>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    sizeof(AttentionSharedStorage<128>)));
  }
  auto launch = [&](FaultStage fault_stage, cudaStream_t stream) {
    if (options.head_dim == 64) {
      ft_flash_attention_a13_kernel<64>
          <<<grid, kThreads, sizeof(AttentionSharedStorage<64>), stream>>>(
          d_q, d_k, d_v, d_output, options.batch, options.heads,
          d_k_sum, d_k_weighted, d_k_abs_sum, d_k_weighted_abs_sum, d_v_sum, d_v_weighted,
          options.sequence, key_tiles, options.causal, score_scale, static_cast<int>(fault_stage), options.fault_batch, options.fault_head,
          options.fault_query, options.fault_index, options.fault_value, options.score_abs_tolerance,
          options.score_rel_tolerance, options.softmax_abs_tolerance, options.softmax_rel_tolerance,
          options.pv_abs_tolerance, options.pv_rel_tolerance, d_stats);
    } else {
      ft_flash_attention_a13_kernel<128>
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

  std::printf("version: FT-FlashAttention A13 register-resident Softmax/PV\n");
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
