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
FT_FLASH_ATTN_A0_TARGET := $(BUILD_DIR)/ft_flash_attention_a0
FT_FLASH_ATTN_A1_TARGET := $(BUILD_DIR)/ft_flash_attention_a1
FT_FLASH_ATTN_A2_TARGET := $(BUILD_DIR)/ft_flash_attention_a2
FT_FLASH_ATTN_A3_TARGET := $(BUILD_DIR)/ft_flash_attention_a3
FT_FLASH_ATTN_A4_TARGET := $(BUILD_DIR)/ft_flash_attention_a4
FT_FLASH_ATTN_A5_TARGET := $(BUILD_DIR)/ft_flash_attention_a5
FT_FLASH_ATTN_A6_TARGET := $(BUILD_DIR)/ft_flash_attention_a6
FT_FLASH_ATTN_A7_TARGET := $(BUILD_DIR)/ft_flash_attention_a7
FT_FLASH_ATTN_A8_TARGET := $(BUILD_DIR)/ft_flash_attention_a8
FT_FLASH_ATTN_A9_TARGET := $(BUILD_DIR)/ft_flash_attention_a9
FT_FLASH_ATTN_A10_TARGET := $(BUILD_DIR)/ft_flash_attention_a10
FT_FLASH_ATTN_A11_TARGET := $(BUILD_DIR)/ft_flash_attention_a11
FT_FLASH_ATTN_A12_TARGET := $(BUILD_DIR)/ft_flash_attention_a12
FT_FLASH_ATTN_A13_TARGET := $(BUILD_DIR)/ft_flash_attention_a13
FT_FLASH_ATTN_A14_TARGET := $(BUILD_DIR)/ft_flash_attention_a14

.PHONY: all key_stages stage0 stage1 stage2 stage3 stage4 s4_fault_campaign ft_flash_attention_a0 ft_flash_attention_a1 ft_flash_attention_a2 ft_flash_attention_a3 ft_flash_attention_a4 ft_flash_attention_a5 ft_flash_attention_a6 ft_flash_attention_a7 ft_flash_attention_a8 ft_flash_attention_a9 ft_flash_attention_a10 ft_flash_attention_a11 ft_flash_attention_a12 ft_flash_attention_a13 ft_flash_attention_a14 gemm_softmax_abft cublaslt_fp8 clean

all: key_stages

key_stages: stage0 stage1 stage2 stage3 stage4

stage0: $(STAGE0_TARGET)

stage1: $(STAGE1_TARGET)

stage2: $(STAGE2_TARGET)

stage3: $(STAGE3_TARGET)

stage4: $(STAGE4_TARGET)

s4_fault_campaign: $(S4_FAULT_CAMPAIGN_TARGET)
	CUDA_VISIBLE_DEVICES=7 ./$(S4_FAULT_CAMPAIGN_TARGET) --m 4096 --n 4096 --k 4096 --warmup 1 --repeat 1 --rel-tol $(FAULT_REL_TOL) --fault-min-value $(FAULT_MIN_VALUE) --fault-max-value $(FAULT_MAX_VALUE) --fault-trials $(FAULT_TRIALS) --fault-seed $(FAULT_SEED)

ft_flash_attention_a0: $(FT_FLASH_ATTN_A0_TARGET)

ft_flash_attention_a1: $(FT_FLASH_ATTN_A1_TARGET)

ft_flash_attention_a2: $(FT_FLASH_ATTN_A2_TARGET)

ft_flash_attention_a3: $(FT_FLASH_ATTN_A3_TARGET)

ft_flash_attention_a4: $(FT_FLASH_ATTN_A4_TARGET)

ft_flash_attention_a5: $(FT_FLASH_ATTN_A5_TARGET)

ft_flash_attention_a6: $(FT_FLASH_ATTN_A6_TARGET)

ft_flash_attention_a7: $(FT_FLASH_ATTN_A7_TARGET)

ft_flash_attention_a8: $(FT_FLASH_ATTN_A8_TARGET)

ft_flash_attention_a9: $(FT_FLASH_ATTN_A9_TARGET)

ft_flash_attention_a10: $(FT_FLASH_ATTN_A10_TARGET)

ft_flash_attention_a11: $(FT_FLASH_ATTN_A11_TARGET)

ft_flash_attention_a12: $(FT_FLASH_ATTN_A12_TARGET)

ft_flash_attention_a13: $(FT_FLASH_ATTN_A13_TARGET)

ft_flash_attention_a14: $(FT_FLASH_ATTN_A14_TARGET)

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

$(FT_FLASH_ATTN_A0_TARGET): $(STAGE_DIR)/ft_flash_attention_a0.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(FT_FLASH_ATTN_A1_TARGET): $(STAGE_DIR)/ft_flash_attention_a1.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(FT_FLASH_ATTN_A2_TARGET): $(STAGE_DIR)/ft_flash_attention_a2.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(FT_FLASH_ATTN_A3_TARGET): $(STAGE_DIR)/ft_flash_attention_a3.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(FT_FLASH_ATTN_A4_TARGET): $(STAGE_DIR)/ft_flash_attention_a4.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math $(ARCH_FLAGS) $< -o $@

$(FT_FLASH_ATTN_A5_TARGET): $(STAGE_DIR)/ft_flash_attention_a5.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math $(ARCH_FLAGS) $< -o $@

$(FT_FLASH_ATTN_A6_TARGET): $(STAGE_DIR)/ft_flash_attention_a6.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math $(ARCH_FLAGS) $< -o $@

$(FT_FLASH_ATTN_A7_TARGET): $(STAGE_DIR)/ft_flash_attention_a7.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math $(ARCH_FLAGS) $< -o $@

$(FT_FLASH_ATTN_A8_TARGET): $(STAGE_DIR)/ft_flash_attention_a8.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(FT_FLASH_ATTN_A9_TARGET): $(STAGE_DIR)/ft_flash_attention_a9.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(FT_FLASH_ATTN_A10_TARGET): $(STAGE_DIR)/ft_flash_attention_a10.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(FT_FLASH_ATTN_A11_TARGET): $(STAGE_DIR)/ft_flash_attention_a11.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(FT_FLASH_ATTN_A12_TARGET): $(STAGE_DIR)/ft_flash_attention_a12.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(FT_FLASH_ATTN_A13_TARGET): $(STAGE_DIR)/ft_flash_attention_a13.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(FT_FLASH_ATTN_A14_TARGET): $(STAGE_DIR)/ft_flash_attention_a14.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) --use_fast_math --expt-relaxed-constexpr $(CUTLASS_ARCH_FLAGS) -I$(CUTLASS_ROOT)/include $< -o $@

$(GEMM_SOFTMAX_ABFT_TARGET): $(STAGE_DIR)/gemm_softmax_abft.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(ARCH_FLAGS) $< -o $@

$(CUBLASLT_FP8_TARGET): $(STAGE_DIR)/cublaslt_fp8_gemm_benchmark.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUTLASS_ARCH_FLAGS) $< -o $@ -lcublasLt -lcublas

clean:
	rm -rf $(BUILD_DIR)
