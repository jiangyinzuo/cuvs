/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "device_common.cuh"
#include "kernel_debug.cuh"

namespace cuvs::neighbors::my_anns_v1::detail {

template <typename IndexT>
class ComputeRandomEntryPoints {
 public:
  ComputeRandomEntryPoints(const IndexT* dev_seed_ptr,
                           uint32_t num_seeds,
                           uint32_t num_random_samplings,
                           uint64_t rand_xor_mask)
    : dev_seed_ptr(dev_seed_ptr),
      num_seeds(num_seeds),
      num_random_samplings(num_random_samplings),
      rand_xor_mask(rand_xor_mask)
  {
  }

  template <typename DistanceT, class DATASET_DESCRIPTOR_T, class VisitedTable>
  __device__ __forceinline__ void operator()(const uint32_t query_id,
                                             IndexT* result_indices_buffer,
                                             DistanceT* result_distances_buffer,
                                             const DATASET_DESCRIPTOR_T* dataset_desc,
                                             const uint32_t result_buffer_size,
                                             VisitedTable visited_table
#ifdef _GRAPH_QUALITY_ANALYSIS
                                             ,
                                             MyAnnsV1Metrics* my_anns_v1_metrics,
                                             uint64_t* local_distance_calculation_counter1,
                                             uint64_t* local_distance_calculation_counter2
#endif
  ) const
  {
    // compute distance to randomly selecting nodes
    // _CLK_START();
    const IndexT* const local_seed_ptr =
      dev_seed_ptr ? dev_seed_ptr + (num_seeds * query_id) : nullptr;
    device::compute_distance_to_random_nodes(result_indices_buffer,
                                             result_distances_buffer,
                                             *dataset_desc,
                                             result_buffer_size,
                                             num_random_samplings,
                                             rand_xor_mask,
                                             local_seed_ptr,
                                             num_seeds,
                                             visited_table
#ifdef _GRAPH_QUALITY_ANALYSIS
                                             ,
                                             my_anns_v1_metrics,
                                             local_distance_calculation_counter1,
                                             local_distance_calculation_counter2
#endif
    );
  }

  __device__ __forceinline__ bool must_visited(IndexT vector_id) const { return false; }

 private:
  const IndexT* dev_seed_ptr;  // [num_queries, num_seeds]
  uint32_t num_seeds;
  uint32_t num_random_samplings;
  uint64_t rand_xor_mask;
};

class AlwaysUnvisited {
 public:
  __device__ __forceinline__ bool must_visited(int32_t vector_id) const { return false; }
};

// https://github.com/vllm-project/vllm/blob/f90d34b4985dd57262b93be2b75c9099babb2188/csrc/quantization/gptq_marlin/marlin.cuh
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true)
{
  const int BYTES = 16;
  uint32_t smem   = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .pred p;\n"
    "   setp.ne.b32 p, %0, 0;\n"
    "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}\n" ::"r"((int)pred),
    "r"(smem),
    "l"(glob_ptr),
    "n"(BYTES));
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr)
{
  const int BYTES = 16;
  uint32_t smem   = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   cp.async.cg.shared.global [%0], [%1], %2;\n"
    "}\n" ::"r"(smem),
    "l"(glob_ptr),
    "n"(BYTES));
}

__device__ inline void cp_async_fence() { asm volatile("cp.async.commit_group;\n" ::); }

template <int n>
__device__ inline void cp_async_wait()
{
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

/// Blocks until all previous cp.async.commit_group operations have committed.
// cp.async.wait_all is equivalent to :
// cp.async.commit_group;
// cp.async.wait_group 0;
__device__ inline void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }

template <typename IndexT, typename DistanceT>
struct MemcpyEntryPoints {
  MemcpyEntryPoints(const IndexT* result_indices_buffer,
                    const DistanceT* result_distances_buffer,
                    uint32_t num_entry_points)
    : global_mem_result_indices_buffer(result_indices_buffer),
      global_mem_result_distances_buffer(result_distances_buffer),
      num_entry_points(num_entry_points)
  {
  }

  __device__ __forceinline__ void operator()(const std::uint32_t query_id,
                                             IndexT* __restrict__ smem_result_indices_buffer,
                                             DistanceT* __restrict__ smem_result_distances_buffer,
                                             const std::uint32_t itopk_size) const
  {
    print_buffer(global_mem_result_distances_buffer, itopk_size);
    print_buffer(global_mem_result_indices_buffer, itopk_size);
    assert(itopk_size % 4 == 0);
    // async copy from global memory to shared memory
    // every thread copies 16 bytes
    constexpr int BYTES_PER_THREAD               = 16;
    constexpr int WARP_SIZE                      = 32;
    const int indices_buffer_elements_per_thread = BYTES_PER_THREAD / sizeof(IndexT);
    auto global_buffer_offset                    = query_id * itopk_size;
    for (uint32_t i = 0; i < itopk_size; i += WARP_SIZE * indices_buffer_elements_per_thread) {
      uint32_t idx = i + threadIdx.x * indices_buffer_elements_per_thread;
      bool pred    = idx < itopk_size;
      cp_async4_pred(smem_result_indices_buffer + idx,
                     global_mem_result_indices_buffer + global_buffer_offset + idx,
                     pred);
    }

    const int distances_buffer_elements_per_thread = BYTES_PER_THREAD / sizeof(DistanceT);
    for (uint32_t i = 0; i < itopk_size; i += WARP_SIZE * distances_buffer_elements_per_thread) {
      uint32_t idx = i + threadIdx.x * distances_buffer_elements_per_thread;
      bool pred    = idx < itopk_size;
      cp_async4_pred(smem_result_distances_buffer + idx,
                     global_mem_result_distances_buffer + global_buffer_offset + idx,
                     pred);
    }
    cp_async_wait_all();
    print_buffer(smem_result_distances_buffer, itopk_size);
    print_buffer(smem_result_indices_buffer, itopk_size);
  }

  __device__ __forceinline__ bool must_visited(IndexT vector_id) const
  {
    return vector_id < num_entry_points;
  }

  const IndexT* __restrict__ global_mem_result_indices_buffer;       // [num_queries, itopk_size]
  const DistanceT* __restrict__ global_mem_result_distances_buffer;  // [num_queries, itopk_size]
  uint32_t num_entry_points;
};

}  // namespace cuvs::neighbors::my_anns_v1::detail
