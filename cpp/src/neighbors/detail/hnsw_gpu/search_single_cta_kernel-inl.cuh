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

#include "device_common.hpp"

#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger-ext.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/neighbors/common.hpp>

// TODO: This shouldn't be invoking anything from spatial/knn
#include "../ann_utils.cuh"

#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda/atomic>
#include <cuda/std/atomic>

#include "hnsw_graph.cuh"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdint.h>
#include <thread>
#include <vector>

namespace cuvs::neighbors::experimental::hnsw_gpu::detail {

template <unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class VISITED_TABLE_T,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T>
RAFT_KERNEL __launch_bounds__(1024, 1) search_kernel(
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries, top_k]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, top_k]
  const std::uint32_t top_k,
  const DATASET_DESCRIPTOR_T* dataset_desc,
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const HNSWGraph<DATASET_DESCRIPTOR_T> hnsw_graph,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const
    visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,  // [num_queries]
  const std::uint32_t visited_table_size,
  SAMPLE_FILTER_T sample_filter)
{
  using LOAD_T = device::LOAD_128BIT_T;

  using DATA_T     = typename DATASET_DESCRIPTOR_T::DATA_T;
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;

  const auto query_id = blockIdx.y;
  extern __shared__ uint8_t smem[];

  // Layout of result_buffer
  // +----------------------+------------------------------+---------+
  // | internal_top_k       | neighbors of internal_top_k  | padding |
  // | <internal_topk_size> | <search_width * graph_degree>| upto 32 |
  // +----------------------+------------------------------+---------+
  // |<---             result_buffer_size              --->|
  const auto result_buffer_size    = internal_topk + (search_width * hnsw_graph.graph_degree_);
  const auto result_buffer_size_32 = raft::round_up_safe<uint32_t>(result_buffer_size, 32);

  // Set smem working buffer for the distance calculation
  dataset_desc = dataset_desc->setup_workspace(smem, queries_ptr, query_id);

  auto* __restrict__ result_indices_buffer =
    reinterpret_cast<INDEX_T*>(smem + dataset_desc->smem_ws_size_in_bytes());
  auto* __restrict__ result_distances_buffer =
    reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size_32);
  auto* __restrict__ visited_table_buffer =
    reinterpret_cast<INDEX_T*>(result_distances_buffer + result_buffer_size_32);
  auto* __restrict__ parent_list_buffer =
    reinterpret_cast<INDEX_T*>(visited_table_buffer + (1 << 8));
  auto* __restrict__ topk_ws = reinterpret_cast<std::uint32_t*>(parent_list_buffer + search_width);
  auto* terminate_flag       = reinterpret_cast<std::uint32_t*>(topk_ws + 3);
  auto* __restrict__ smem_work_ptr = reinterpret_cast<std::uint32_t*>(terminate_flag + 1);

  // A flag for filtering.
  auto filter_flag = terminate_flag;

  if (threadIdx.x == 0) {
    terminate_flag[0] = 0;
    topk_ws[0]        = ~0u;
  }

  // Init visited table
  VISITED_TABLE_T visited_table(visited_table_buffer, visited_table_size / sizeof(uint64_t));

  // compute distance to randomly selecting nodes at the highest level

  for (int level = hnsw_graph.max_level_ - 1; level > 0; level--) {
    bool changed = true;
    while (changed) {}
  }
}

template <bool Persistent,
          unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T>
auto dispatch_kernel = []() {
  if constexpr (Persistent) {
    static_assert(false, "Persistent search kernel is not implemented yet.");
  } else {
    return search_kernel<MAX_ITOPK,
                         MAX_CANDIDATES,
                         TOPK_BY_BITONIC_SORT,
                         DATASET_DESCRIPTOR_T,
                         SAMPLE_FILTER_T>;
  }
}();

template <bool Persistent, typename DATASET_DESCRIPTOR_T, typename SAMPLE_FILTER_T>
struct search_kernel_config {
  using kernel_t =
    decltype(dispatch_kernel<Persistent, 64, 64, 0, DATASET_DESCRIPTOR_T, SAMPLE_FILTER_T>);

  void select_and_run(uint32_t block_size, uint32_t num_queries)
  {
    dim3 block_dims(1, num_queries, 1);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
};

template <typename DataT, typename IndexT, typename DistanceT, typename SampleFilterT>
void select_and_run(raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
                    IndexT* topk_indices_ptr,       // [num_queries, topk]
                    DistanceT* topk_distances_ptr,  // [num_queries, topk]
                    const DataT* queries_ptr,       // [num_queries, dataset_dim]
                    uint32_t num_queries,
                    const IndexT* dev_seed_ptr,         // [num_queries, num_seeds]
                    uint32_t* num_executed_iterations,  // [num_queries,]
                    const search_params& ps,
                    uint32_t topk,
                    uint32_t num_itopk_candidates,
                    uint32_t block_size,  //
                    uint32_t smem_size,
                    int64_t hash_bitlen,
                    IndexT* hashmap_ptr,
                    size_t small_hash_bitlen,
                    size_t small_hash_reset_interval,
                    uint32_t num_seeds,
                    SampleFilterT sample_filter,
                    cudaStream_t stream)
{
  assert(dev_seed_ptr != nullptr);
}
}  // namespace cuvs::neighbors::experimental::hnsw_gpu::detail
