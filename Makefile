NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Wno-deprecated-gpu-targets
ARCH_FLAGS ?= -gencode arch=compute_90,code=sm_90 \
	-gencode arch=compute_90,code=compute_90
CUTLASS_ARCH_FLAGS ?= -gencode arch=compute_90a,code=sm_90a \
	-gencode arch=compute_90a,code=compute_90a
BUILD_DIR ?= build
STAGE_DIR := src
FAULT_TRIALS ?= 100
FAULT_MIN_VALUE ?= 1
FAULT_MAX_VALUE ?= 256
FAULT_SEED ?= 12345
FAULT_REL_TOL ?= 0.00002
CUTLASS_ROOT ?= /data01/docker/overlay2/350eb5e5f9f6bcd790bb1206eaee8cf86c207c2589b3dee54c78dacab78d8cfe/diff/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass

CUBLASLT_FP8_TARGET := $(BUILD_DIR)/cublaslt_fp8_gemm_benchmark
GEMM_SOFTMAX_ABFT_TARGET := $(BUILD_DIR)/gemm_softmax_abft
STAGE0_TARGET := $(BUILD_DIR)/stage0_tiled_abft_fp8_gemm
STAGE1_TARGET := $(BUILD_DIR)/stage1_tma_wgmma_abft_fp8_gemm
STAGE2_TARGET := $(BUILD_DIR)/stage2_expected_cache_abft_fp8_gemm
STAGE3_TARGET := $(BUILD_DIR)/stage3_warpspecialized_abft_fp8_gemm
STAGE4_TARGET := $(BUILD_DIR)/stage4_cluster_multicast_abft_fp8_gemm
S4_FAULT_CAMPAIGN_TARGET := $(BUILD_DIR)/s4_random_fault_campaign

.PHONY: all key_stages stage0 stage1 stage2 stage3 stage4 s4_fault_campaign gemm_softmax_abft cublaslt_fp8 clean

all: key_stages

key_stages: stage0 stage1 stage2 stage3 stage4

stage0: $(STAGE0_TARGET)

stage1: $(STAGE1_TARGET)

stage2: $(STAGE2_TARGET)

stage3: $(STAGE3_TARGET)

stage4: $(STAGE4_TARGET)

s4_fault_campaign: $(S4_FAULT_CAMPAIGN_TARGET)
	CUDA_VISIBLE_DEVICES=1 ./$(S4_FAULT_CAMPAIGN_TARGET) --m 4096 --n 4096 --k 4096 --warmup 1 --repeat 1 --rel-tol $(FAULT_REL_TOL) --fault-min-value $(FAULT_MIN_VALUE) --fault-max-value $(FAULT_MAX_VALUE) --fault-trials $(FAULT_TRIALS) --fault-seed $(FAULT_SEED)

gemm_softmax_abft: $(GEMM_SOFTMAX_ABFT_TARGET)

cublaslt_fp8: $(CUBLASLT_FP8_TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(STAGE0_TARGET): $(STAGE_DIR)/s0_tiled_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(STAGE1_TARGET): $(STAGE_DIR)/s1_tma_wgmma_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(STAGE2_TARGET): $(STAGE_DIR)/s2_expected_cache_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(STAGE3_TARGET): $(STAGE_DIR)/s3_warpspecialized_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(STAGE4_TARGET): $(STAGE_DIR)/s4_cluster_multicast_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(S4_FAULT_CAMPAIGN_TARGET): $(STAGE_DIR)/s4_random_fault_campaign.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(GEMM_SOFTMAX_ABFT_TARGET): $(STAGE_DIR)/gemm_softmax_abft.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(CUBLASLT_FP8_TARGET): $(STAGE_DIR)/cublaslt_fp8_gemm_benchmark.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUTLASS_ARCH_FLAGS) $< -o $@ -lcublasLt -lcublas

clean:
	rm -rf $(BUILD_DIR)
