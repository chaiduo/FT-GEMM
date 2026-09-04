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

TARGET := $(BUILD_DIR)/abft_fp8_gemm
CUTLASS_TARGET := $(BUILD_DIR)/cutlass_fp8_gemm
V0_NATIVE_TARGET := $(BUILD_DIR)/v0_native_fp8_gemm
V1_NATIVE_TARGET := $(BUILD_DIR)/v1_native_tiled_fp8_gemm
V2_NATIVE_TARGET := $(BUILD_DIR)/v2_native_tiled_abft_fp8_gemm
V3_NATIVE_TARGET := $(BUILD_DIR)/v3_native_pipeline_abft_fp8_gemm
V4_WGMMA_TARGET := $(BUILD_DIR)/v4_wgmma_fp8_gemm
V5_WGMMA_TARGET := $(BUILD_DIR)/v5_wgmma_tma_fp8_gemm
V6_WGMMA_TARGET := $(BUILD_DIR)/v6_wgmma_fused_abft_fp8_gemm
V7_WGMMA_TARGET := $(BUILD_DIR)/v7_wgmma_optimized_abft_fp8_gemm
V8_WGMMA_TARGET := $(BUILD_DIR)/v8_wgmma_cached_abft_fp8_gemm
V9_WGMMA_TARGET := $(BUILD_DIR)/v9_wgmma_three_stage_abft_fp8_gemm
V10_WGMMA_TARGET := $(BUILD_DIR)/v10_wgmma_padded_pipeline_abft_fp8_gemm
V11_WGMMA_TARGET := $(BUILD_DIR)/v11_wgmma_expected_cache_abft_fp8_gemm
V12_WGMMA_TARGET := $(BUILD_DIR)/v12_wgmma_wide_warpspecialized_abft_fp8_gemm
V13_WGMMA_TARGET := $(BUILD_DIR)/v13_wgmma_k128_cluster_abft_fp8_gemm
CUBLASLT_FP8_TARGET := $(BUILD_DIR)/cublaslt_fp8_gemm_benchmark
STAGE0_TARGET := $(BUILD_DIR)/stage0_tiled_abft_fp8_gemm
STAGE1_TARGET := $(BUILD_DIR)/stage1_pipeline_abft_fp8_gemm
STAGE2_TARGET := $(BUILD_DIR)/stage2_wgmma_pipeline_abft_fp8_gemm
STAGE3_TARGET := $(BUILD_DIR)/stage3_fused_online_abft_fp8_gemm
STAGE4_TARGET := $(BUILD_DIR)/stage4_cached_input_abft_fp8_gemm
STAGE5_TARGET := $(BUILD_DIR)/stage5_expected_cache_abft_fp8_gemm
STAGE6_TARGET := $(BUILD_DIR)/stage6_warpspecialized_abft_fp8_gemm
STAGE7_TARGET := $(BUILD_DIR)/stage7_cluster_multicast_abft_fp8_gemm
S7_FAULT_CAMPAIGN_TARGET := $(BUILD_DIR)/s7_random_fault_campaign
WGMMA_TARGET := $(BUILD_DIR)/wgmma_fp8_abft

.PHONY: all key_stages stage0 stage1 stage2 stage3 stage4 stage5 stage6 stage7 s7_fault_campaign native_v0 native_v1 native_v2 native_v3 wgmma_v4 wgmma_v5 wgmma_v6 wgmma_v7 wgmma_v8 wgmma_v9 wgmma_v10 wgmma_v11 wgmma_v12 wgmma_v13 cublaslt_fp8 cutlass raw_wgmma clean run fault

all: key_stages

key_stages: stage0 stage1 stage2 stage3 stage4 stage5 stage6 stage7

stage0: $(STAGE0_TARGET)

stage1: $(STAGE1_TARGET)

stage2: $(STAGE2_TARGET)

stage3: $(STAGE3_TARGET)

stage4: $(STAGE4_TARGET)

stage5: $(STAGE5_TARGET)

stage6: $(STAGE6_TARGET)

stage7: $(STAGE7_TARGET)

s7_fault_campaign: $(S7_FAULT_CAMPAIGN_TARGET)
	CUDA_VISIBLE_DEVICES=1 ./$(S7_FAULT_CAMPAIGN_TARGET) --m 4096 --n 4096 --k 4096 --warmup 1 --repeat 1 --rel-tol $(FAULT_REL_TOL) --fault-min-value $(FAULT_MIN_VALUE) --fault-max-value $(FAULT_MAX_VALUE) --fault-trials $(FAULT_TRIALS) --fault-seed $(FAULT_SEED)

native_v0: $(V0_NATIVE_TARGET)

native_v1: $(V1_NATIVE_TARGET)

native_v2: $(V2_NATIVE_TARGET)

native_v3: $(V3_NATIVE_TARGET)

wgmma_v4: $(V4_WGMMA_TARGET)

wgmma_v5: $(V5_WGMMA_TARGET)

wgmma_v6: $(V6_WGMMA_TARGET)

wgmma_v7: $(V7_WGMMA_TARGET)

wgmma_v8: $(V8_WGMMA_TARGET)

wgmma_v9: $(V9_WGMMA_TARGET)

wgmma_v10: $(V10_WGMMA_TARGET)

wgmma_v11: $(V11_WGMMA_TARGET)

wgmma_v12: $(V12_WGMMA_TARGET)

wgmma_v13: $(V13_WGMMA_TARGET)

cublaslt_fp8: $(CUBLASLT_FP8_TARGET)

cutlass: $(CUTLASS_TARGET)

raw_wgmma: $(WGMMA_TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(TARGET): src/abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(STAGE0_TARGET): $(STAGE_DIR)/s0_tiled_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(STAGE1_TARGET): $(STAGE_DIR)/s1_pipeline_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(STAGE2_TARGET): $(STAGE_DIR)/s2_wgmma_pipeline_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(STAGE3_TARGET): $(STAGE_DIR)/s3_fused_online_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(STAGE4_TARGET): $(STAGE_DIR)/s4_cached_input_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(STAGE5_TARGET): $(STAGE_DIR)/s5_expected_cache_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(STAGE6_TARGET): $(STAGE_DIR)/s6_warpspecialized_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(STAGE7_TARGET): $(STAGE_DIR)/s7_cluster_multicast_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(S7_FAULT_CAMPAIGN_TARGET): $(STAGE_DIR)/s7_random_fault_campaign.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(V0_NATIVE_TARGET): src/v0_native_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(V1_NATIVE_TARGET): src/v1_native_tiled_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(V2_NATIVE_TARGET): src/v2_native_tiled_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(V3_NATIVE_TARGET): src/v3_native_pipeline_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(V4_WGMMA_TARGET): src/v4_wgmma_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(V5_WGMMA_TARGET): src/v5_wgmma_tma_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(V6_WGMMA_TARGET): src/v6_wgmma_fused_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(V7_WGMMA_TARGET): src/v7_wgmma_optimized_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(V8_WGMMA_TARGET): src/v8_wgmma_cached_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(V9_WGMMA_TARGET): src/v9_wgmma_three_stage_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(V10_WGMMA_TARGET): src/v10_wgmma_padded_pipeline_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(V11_WGMMA_TARGET): src/v11_wgmma_expected_cache_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(V12_WGMMA_TARGET): src/v12_wgmma_wide_warpspecialized_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(V13_WGMMA_TARGET): src/v13_wgmma_k128_cluster_abft_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(CUBLASLT_FP8_TARGET): src/cublaslt_fp8_gemm_benchmark.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUTLASS_ARCH_FLAGS) $< -o $@ -lcublasLt -lcublas

$(CUTLASS_TARGET): src/cutlass_fp8_gemm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(WGMMA_TARGET): src/wgmma_fp8_abft.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

run: $(TARGET)
	./$(TARGET)

fault: $(TARGET)
	-./$(TARGET) --inject-fault

clean:
	rm -rf $(BUILD_DIR)
