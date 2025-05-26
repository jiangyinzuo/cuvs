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

#include "search_single_cta_kernel.cuh"

#include "compute_distance-ext.cuh"
#include "device_common.cuh"
#include "entry_points_policy.cuh"
#include "graph_analysis_macros.h"
#include "hashmap.hpp"
#include "pickup_next_parents.cuh"
#include "search_plan.cuh"
#include "sort.cuh"
#include "topk_by_radix.cuh"
#include "topk_for_my_anns_v1/topk.h"  // TODO replace with raft topk
#include "utils.hpp"
#include "visited_table.cuh"

#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/my_anns_v1_metrics.cuh>

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

#include <cassert>
#include <cstdio>
#include <limits>
#include <stdint.h>

#include "kernel_debug.cuh"

namespace cuvs::neighbors::my_anns_v1::detail {
namespace single_cta_search {

// This function move the invalid index element to the end of the itopk list.
// Require : array_length % 32 == 0 && The invalid entry is only one.
template <class IdxT>
RAFT_DEVICE_INLINE_FUNCTION void move_invalid_to_end_of_list(IdxT* const index_array,
                                                             float* const distance_array,
                                                             const std::uint32_t array_length)
{
  constexpr std::uint32_t warp_size     = 32;
  constexpr std::uint32_t invalid_index = utils::get_max_value<IdxT>();
  const std::uint32_t lane_id           = threadIdx.x % warp_size;

  if (threadIdx.x >= warp_size) { return; }

  bool found_invalid = false;
  if (array_length % warp_size == 0) {
    for (std::uint32_t i = lane_id; i < array_length; i += warp_size) {
      const auto index    = index_array[i];
      const auto distance = distance_array[i];

      if (found_invalid) {
        index_array[i - 1]    = index;
        distance_array[i - 1] = distance;
      } else {
        // Check if the index is invalid
        const auto I_found_invalid = (index == invalid_index);
        const auto who_has_invalid = raft::ballot(I_found_invalid);
        // if a value that is loaded by a smaller lane id thread, shift the array
        if (who_has_invalid << (warp_size - lane_id)) {
          index_array[i - 1]    = index;
          distance_array[i - 1] = distance;
        }

        found_invalid = who_has_invalid;
      }
    }
  }
  if (lane_id == 0) {
    index_array[array_length - 1]    = invalid_index;
    distance_array[array_length - 1] = utils::get_max_value<float>();
  }
}

// One query one thread block
template <unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T,
          class EntryPointsPolicy,
          class VisitedTable>
__device__ void search_core(
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries, top_k]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, top_k]
  const std::uint32_t top_k,
  const DATASET_DESCRIPTOR_T* dataset_desc,
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const knn_graph,   // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,  // [num_queries]
  const std::uint32_t query_id,
  SAMPLE_FILTER_T sample_filter,
  const EntryPointsPolicy entry_points_policy,
  VisitedTable visited_table
#ifdef _GRAPH_QUALITY_ANALYSIS
  ,
  MyAnnsV1Metrics* my_anns_v1_metrics
#endif
)
{
  using LOAD_T = device::LOAD_128BIT_T;

  using DATA_T     = typename DATASET_DESCRIPTOR_T::DATA_T;
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;

  static_assert(std::is_same_v<EntryPointsPolicy, ComputeRandomEntryPoints<INDEX_T>> ||
                  std::is_same_v<EntryPointsPolicy, MemcpyEntryPoints<INDEX_T, DISTANCE_T>>,
                "Unknown EntryPointsPolicy");

#ifdef _GRAPH_QUALITY_ANALYSIS
  __shared__ uint64_t local_distance_calculation_counter1;
  __shared__ uint64_t local_distance_calculation_counter2;
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    local_distance_calculation_counter1 = 0;
    local_distance_calculation_counter2 = 0;
  }
  if (threadIdx.x == 0 && query_id == 0) { my_anns_v1_metrics->reset(); }
  if (METRIC_THREAD_COND()) { atomicAdd(&my_anns_v1_metrics->counter_clk_thread, 1UL); }
  __syncthreads();
#endif

#ifdef _CLK_BREAKDOWN
  std::uint64_t clk_init = 0;
  // std::uint64_t clk_compute_1st_distance = 0;
  std::uint64_t clk_topk           = 0;
  std::uint64_t clk_reset_hash     = 0;
  std::uint64_t clk_pickup_parents = 0;
  std::uint64_t clk_restore_hash   = 0;
  // std::uint64_t clk_compute_distance     = 0;
  std::uint64_t clk_final = 0;
  std::uint64_t clk_start;
#define _CLK_START() clk_start = clock64()
#define _CLK_REC(V)  V += clock64() - clk_start;
#else
#define _CLK_START()
#define _CLK_REC(V)
#endif
  _CLK_START();

  extern __shared__ uint8_t smem[];

  // Layout of result_buffer
  // +----------------------+------------------------------+---------+
  // | internal_top_k       | neighbors of internal_top_k  | padding |
  // | <internal_topk_size> | <search_width * graph_degree> | upto 32 |
  // +----------------------+------------------------------+---------+
  // |<---             result_buffer_size              --->|
  const auto result_buffer_size    = internal_topk + (search_width * graph_degree);
  const auto result_buffer_size_32 = raft::round_up_safe<uint32_t>(result_buffer_size, 32);

  // Set smem working buffer for the distance calculation
  dataset_desc = dataset_desc->setup_workspace(smem, queries_ptr, query_id);

  auto* __restrict__ result_indices_buffer =
    reinterpret_cast<INDEX_T*>(smem + dataset_desc->smem_ws_size_in_bytes());
  auto* __restrict__ result_distances_buffer =
    reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size_32);
  auto* __restrict__ parent_list_buffer = reinterpret_cast<INDEX_T*>(visited_table.setup_table(
    reinterpret_cast<INDEX_T*>(result_distances_buffer + result_buffer_size_32)));

  auto* __restrict__ topk_ws = reinterpret_cast<std::uint32_t*>(parent_list_buffer + search_width);
  auto* terminate_flag       = reinterpret_cast<std::uint32_t*>(topk_ws + 3);
  auto* __restrict__ smem_work_ptr = reinterpret_cast<std::uint32_t*>(terminate_flag + 1);

  // A flag for filtering.
  auto filter_flag = terminate_flag;

  if (threadIdx.x == 0) {
    terminate_flag[0] = 0;
    topk_ws[0]        = ~0u;
  }

  __syncthreads();
  _CLK_REC(clk_init);

  // compute distance to randomly selecting nodes
  // _CLK_START();
  if constexpr (std::is_same_v<EntryPointsPolicy, ComputeRandomEntryPoints<INDEX_T>>) {
    entry_points_policy(query_id,
                        result_indices_buffer,
                        result_distances_buffer,
                        dataset_desc,
                        result_buffer_size,
                        visited_table
#ifdef _GRAPH_QUALITY_ANALYSIS
                        ,
                        my_anns_v1_metrics,
                        &local_distance_calculation_counter1,
                        &local_distance_calculation_counter2
#endif
    );
  } else if constexpr (std::is_same_v<EntryPointsPolicy, MemcpyEntryPoints<INDEX_T, DISTANCE_T>>) {
    entry_points_policy(query_id, result_indices_buffer, result_distances_buffer, internal_topk);
    // pick up next parents
    if (threadIdx.x < 32) {
      _CLK_START();
      pickup_next_parents<TOPK_BY_BITONIC_SORT ? TopKSortType::BITONIC_SORT_MERGE
                                               : TopKSortType::RADIX_SORT,
                          INDEX_T>(
        terminate_flag, parent_list_buffer, result_indices_buffer, internal_topk, search_width);
      _CLK_REC(clk_pickup_parents);
#ifdef _GRAPH_QUALITY_ANALYSIS
      if (METRIC_THREAD_COND()) { atomicAdd(&my_anns_v1_metrics->counter_pickup_parents, 1UL); }
#endif
    }
    __syncthreads();

    // compute the norms between child nodes and query node
    // _CLK_START();
    device::compute_distance_to_child_nodes(result_indices_buffer + internal_topk,
                                            result_distances_buffer + internal_topk,
                                            *dataset_desc,
                                            knn_graph,
                                            graph_degree,
                                            visited_table,
                                            parent_list_buffer,
                                            result_indices_buffer,
                                            search_width,
                                            entry_points_policy
#ifdef _GRAPH_QUALITY_ANALYSIS
                                            ,
                                            my_anns_v1_metrics,
                                            &local_distance_calculation_counter1,
                                            &local_distance_calculation_counter2
#endif
    );
    __syncthreads();

    // Filtering
    if constexpr (!std::is_same<SAMPLE_FILTER_T,
                                cuvs::neighbors::filtering::none_sample_filter>::value) {
      if (threadIdx.x == 0) { *filter_flag = 0; }
      __syncthreads();

      constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
      const INDEX_T invalid_index        = utils::get_max_value<INDEX_T>();

      for (unsigned p = threadIdx.x; p < search_width; p += blockDim.x) {
        if (parent_list_buffer[p] != invalid_index) {
          const auto parent_id = result_indices_buffer[parent_list_buffer[p]] & ~index_msb_1_mask;
          if (!sample_filter(query_id, parent_id)) {
            // If the parent must not be in the resulting top-k list, remove from the parent list
            result_distances_buffer[parent_list_buffer[p]] = utils::get_max_value<DISTANCE_T>();
            result_indices_buffer[parent_list_buffer[p]]   = invalid_index;
            *filter_flag                                   = 1;
          }
        }
      }
      __syncthreads();
    }
  }

  __syncthreads();
  // _CLK_REC(clk_compute_1st_distance);

  std::uint32_t iter = 0;
  while (1) {
    const std::uint32_t neighbors_to_compute =
      0 < iter && iter < 10 ? graph_degree / 2 : graph_degree;
    // sort
    if constexpr (TOPK_BY_BITONIC_SORT) {
      // [Notice]
      // It is good to use multiple warps in topk_by_bitonic_sort_and_merge() when
      // batch size is small (short-latency), but it might not be always good
      // when batch size is large (high-throughput).
      // topk_by_bitonic_sort_and_merge() consists of two operations:
      // if MAX_CANDIDATES is greater than 128, the first operation uses two warps;
      // if MAX_ITOPK is greater than 256, the second operation used two warps.
      const unsigned multi_warps_1 = ((blockDim.x >= 64) && (MAX_CANDIDATES > 128)) ? 1 : 0;
      const unsigned multi_warps_2 = ((blockDim.x >= 64) && (MAX_ITOPK > 256)) ? 1 : 0;

      if constexpr (std::is_same_v<VisitedTable, visited_table::SingleMemHashtable<INDEX_T>>) {
        // reset small-hash table.
        if (visited_table.need_reset(iter)) {
          // Depending on the block size and the number of warps used in
          // topk_by_bitonic_sort_and_merge(), determine which warps are used to reset
          // the small hash and whether they are performed in overlap with
          // topk_by_bitonic_sort_and_merge().
          _CLK_START();
          unsigned hash_start_tid;
          if (blockDim.x == 32) {
            hash_start_tid = 0;
          } else if (blockDim.x == 64) {
            if (multi_warps_1 || multi_warps_2) {
              hash_start_tid = 0;
            } else {
              hash_start_tid = 32;
            }
          } else {
            if (multi_warps_1 || multi_warps_2) {
              hash_start_tid = 64;
            } else {
              hash_start_tid = 32;
            }
          }
          hashmap::init(visited_table.table, visited_table.bitlen, hash_start_tid);
          _CLK_REC(clk_reset_hash);
#ifdef _GRAPH_QUALITY_ANALYSIS
          if (METRIC_THREAD_COND()) { atomicAdd(&my_anns_v1_metrics->counter_reset_hash, 1UL); }
#endif
        }
      }

      // topk with bitonic sort
      _CLK_START();
      if (!(std::is_same<SAMPLE_FILTER_T, cuvs::neighbors::filtering::none_sample_filter>::value ||
            *filter_flag == 0)) {
        // Move the filtered out index to the end of the itopk list
        for (unsigned i = 0; i < search_width; i++) {
          move_invalid_to_end_of_list(
            result_indices_buffer, result_distances_buffer, internal_topk);
        }

        if (threadIdx.x == 0) { *terminate_flag = 0; }
      }
      bool first_iter = (iter == 0);
      if constexpr (std::is_same_v<EntryPointsPolicy, MemcpyEntryPoints<INDEX_T, DISTANCE_T>>) {
        first_iter = false;
      }
      topk_by_bitonic_sort_and_merge<MAX_ITOPK, MAX_CANDIDATES>(
        result_distances_buffer,
        result_indices_buffer,
        internal_topk,
        result_distances_buffer + internal_topk,
        result_indices_buffer + internal_topk,
        search_width * neighbors_to_compute,
        topk_ws,
        first_iter,
        multi_warps_1,
        multi_warps_2);
      __syncthreads();
      _CLK_REC(clk_topk);
#ifdef _GRAPH_QUALITY_ANALYSIS
      if (METRIC_THREAD_COND()) { atomicAdd(&my_anns_v1_metrics->counter_topk_bitonic_sort, 1UL); }
#endif
    } else {
      _CLK_START();
      // topk with radix block sort
      topk_by_radix_sort<MAX_ITOPK, INDEX_T>{}(
        internal_topk,
        gridDim.x,
        internal_topk + (search_width * graph_degree),
        reinterpret_cast<std::uint32_t*>(result_distances_buffer),
        result_indices_buffer,
        reinterpret_cast<std::uint32_t*>(result_distances_buffer),
        result_indices_buffer,
        nullptr,
        topk_ws,
        true,
        smem_work_ptr);
      _CLK_REC(clk_topk);
#ifdef _GRAPH_QUALITY_ANALYSIS
      if (METRIC_THREAD_COND()) { atomicAdd(&my_anns_v1_metrics->counter_topk_radix_sort, 1UL); }
#endif

      if constexpr (std::is_same_v<VisitedTable, visited_table::SingleMemHashtable<INDEX_T>>) {
        // reset small-hash table
        if (visited_table.need_reset(iter)) {
          _CLK_START();
          hashmap::init(visited_table.table, visited_table.bitlen);
          _CLK_REC(clk_reset_hash);
#ifdef _GRAPH_QUALITY_ANALYSIS
          if (METRIC_THREAD_COND()) { atomicAdd(&my_anns_v1_metrics->counter_reset_hash, 1UL); }
#endif
        }
      }
    }
    __syncthreads();

    if (iter + 1 == max_iteration) { break; }

    // pick up next parents
    if (threadIdx.x < 32) {
      _CLK_START();
      pickup_next_parents<TOPK_BY_BITONIC_SORT ? TopKSortType::BITONIC_SORT_MERGE
                                               : TopKSortType::RADIX_SORT,
                          INDEX_T>(
        terminate_flag, parent_list_buffer, result_indices_buffer, internal_topk, search_width);
      _CLK_REC(clk_pickup_parents);
#ifdef _GRAPH_QUALITY_ANALYSIS
      if (METRIC_THREAD_COND()) { atomicAdd(&my_anns_v1_metrics->counter_pickup_parents, 1UL); }
#endif
    }

    if constexpr (std::is_same_v<VisitedTable, visited_table::SingleMemHashtable<INDEX_T>>) {
      // restore small-hash table by putting internal-topk indices in it
      if (visited_table.need_reset(iter)) {
        const unsigned first_tid = ((blockDim.x <= 32) ? 0 : 32);
        _CLK_START();
        visited_table.restore(result_indices_buffer, internal_topk, first_tid);
        _CLK_REC(clk_restore_hash);
#ifdef _GRAPH_QUALITY_ANALYSIS
        if (METRIC_THREAD_COND()) { atomicAdd(&my_anns_v1_metrics->counter_restore_hash, 1UL); }
#endif
      }
    }
    __syncthreads();

    if (*terminate_flag && iter >= min_iteration) { break; }

    // compute the norms between child nodes and query node
    // _CLK_START();
    device::compute_distance_to_child_nodes(result_indices_buffer + internal_topk,
                                            result_distances_buffer + internal_topk,
                                            *dataset_desc,
                                            knn_graph,
                                            neighbors_to_compute,
                                            visited_table,
                                            parent_list_buffer,
                                            result_indices_buffer,
                                            search_width,
                                            entry_points_policy
#ifdef _GRAPH_QUALITY_ANALYSIS
                                            ,
                                            my_anns_v1_metrics,
                                            &local_distance_calculation_counter1,
                                            &local_distance_calculation_counter2
#endif
    );
    __syncthreads();
    // _CLK_REC(clk_compute_distance);

    // Filtering
    if constexpr (!std::is_same<SAMPLE_FILTER_T,
                                cuvs::neighbors::filtering::none_sample_filter>::value) {
      if (threadIdx.x == 0) { *filter_flag = 0; }
      __syncthreads();

      constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
      const INDEX_T invalid_index        = utils::get_max_value<INDEX_T>();

      for (unsigned p = threadIdx.x; p < search_width; p += blockDim.x) {
        if (parent_list_buffer[p] != invalid_index) {
          const auto parent_id = result_indices_buffer[parent_list_buffer[p]] & ~index_msb_1_mask;
          if (!sample_filter(query_id, parent_id)) {
            // If the parent must not be in the resulting top-k list, remove from the parent list
            result_distances_buffer[parent_list_buffer[p]] = utils::get_max_value<DISTANCE_T>();
            result_indices_buffer[parent_list_buffer[p]]   = invalid_index;
            *filter_flag                                   = 1;
          }
        }
      }
      __syncthreads();
    }

    iter++;
  }

  // Post process for filtering
  if constexpr (!std::is_same<SAMPLE_FILTER_T,
                              cuvs::neighbors::filtering::none_sample_filter>::value) {
    constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
    const INDEX_T invalid_index        = utils::get_max_value<INDEX_T>();

    for (unsigned i = threadIdx.x; i < internal_topk + search_width * graph_degree;
         i += blockDim.x) {
      const auto node_id = result_indices_buffer[i] & ~index_msb_1_mask;
      if (node_id != (invalid_index & ~index_msb_1_mask) && !sample_filter(query_id, node_id)) {
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
        result_indices_buffer[i]   = invalid_index;
      }
    }

    __syncthreads();
    // Move invalid index items to the end of the buffer without sorting the entire buffer
    using scan_op_t    = cub::WarpScan<unsigned>;
    auto& temp_storage = *reinterpret_cast<typename scan_op_t::TempStorage*>(smem_work_ptr);

    constexpr std::uint32_t warp_size = 32;
    if (threadIdx.x < warp_size) {
      std::uint32_t num_found_valid = 0;
      for (std::uint32_t buffer_offset = 0; buffer_offset < internal_topk;
           buffer_offset += warp_size) {
        // Calculate the new buffer index
        const auto src_position = buffer_offset + threadIdx.x;
        const std::uint32_t is_valid_index =
          (result_indices_buffer[src_position] & (~index_msb_1_mask)) == invalid_index ? 0 : 1;
        std::uint32_t new_position;
        scan_op_t(temp_storage).InclusiveSum(is_valid_index, new_position);
        if (is_valid_index) {
          const auto dst_position               = num_found_valid + (new_position - 1);
          result_indices_buffer[dst_position]   = result_indices_buffer[src_position];
          result_distances_buffer[dst_position] = result_distances_buffer[src_position];
        }

        // Calculate the largest valid position within a warp and bcast it for the next iteration
        num_found_valid += new_position;
        for (std::uint32_t offset = (warp_size >> 1); offset > 0; offset >>= 1) {
          const auto v = raft::shfl_xor(num_found_valid, offset);
          if ((threadIdx.x & offset) == 0) { num_found_valid = v; }
        }

        // If the enough number of items are found, do early termination
        if (num_found_valid >= top_k) { break; }
      }

      if (num_found_valid < top_k) {
        // Fill the remaining buffer with invalid values so that `topk_by_bitonic_sort_and_merge` is
        // usable in the next step
        for (std::uint32_t i = num_found_valid + threadIdx.x; i < internal_topk; i += warp_size) {
          result_indices_buffer[i]   = invalid_index;
          result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
        }
      }
    }

    // If the sufficient number of valid indexes are not in the internal topk, pick up from the
    // candidate list.
    if (top_k > internal_topk || result_indices_buffer[top_k - 1] == invalid_index) {
      __syncthreads();
      const unsigned multi_warps_1 = ((blockDim.x >= 64) && (MAX_CANDIDATES > 128)) ? 1 : 0;
      const unsigned multi_warps_2 = ((blockDim.x >= 64) && (MAX_ITOPK > 256)) ? 1 : 0;

      bool first_iter = (iter == 0);
      if constexpr (std::is_same_v<EntryPointsPolicy, MemcpyEntryPoints<INDEX_T, DISTANCE_T>>) {
        first_iter = false;
      }
      topk_by_bitonic_sort_and_merge<MAX_ITOPK, MAX_CANDIDATES>(
        result_distances_buffer,
        result_indices_buffer,
        internal_topk,
        result_distances_buffer + internal_topk,
        result_indices_buffer + internal_topk,
        search_width * graph_degree,
        topk_ws,
        first_iter,
        multi_warps_1,
        multi_warps_2);
    }
    __syncthreads();
  }

  _CLK_START();
  for (std::uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
    unsigned j  = i + (top_k * query_id);
    unsigned ii = i;
    if (TOPK_BY_BITONIC_SORT) { ii = device::swizzling(i); }
    if (result_distances_ptr != nullptr) { result_distances_ptr[j] = result_distances_buffer[ii]; }
    constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

    result_indices_ptr[j] =
      result_indices_buffer[ii] & ~index_msb_1_mask;  // clear most significant bit
  }
  if (threadIdx.x == 0 && num_executed_iterations != nullptr) {
    num_executed_iterations[query_id] = iter + 1;
  }
  _CLK_REC(clk_final);
#ifdef _CLK_BREAKDOWN
  if (METRIC_THREAD_COND()) {
    //   printf(
    //     "%s:%d "
    //     "query, %d, thread, %d"
    //     ", init, %lu"
    //     ", 1st_distance, %lu"
    //     ", topk, %lu"
    //     ", reset_hash, %lu"
    //     ", pickup_parents, %lu"
    //     ", restore_hash, %lu"
    //     ", distance, %lu"
    //     "\n",
    //     __FILE__,
    //     __LINE__,
    //     query_id,
    //     threadIdx.x,
    //     clk_init,
    //     clk_compute_1st_distance,
    //     clk_topk,
    //     clk_reset_hash,
    //     clk_pickup_parents,
    //     clk_restore_hash,
    //     clk_compute_distance);
    atomicAdd(&my_anns_v1_metrics->clk_init, clk_init);
    // atomicAdd(&my_anns_v1_metrics->clk_compute_1st_distance, clk_compute_1st_distance);
    atomicAdd(&my_anns_v1_metrics->clk_topk, clk_topk);
    atomicAdd(&my_anns_v1_metrics->clk_reset_hash, clk_reset_hash);
    atomicAdd(&my_anns_v1_metrics->clk_pickup_parents, clk_pickup_parents);
    atomicAdd(&my_anns_v1_metrics->clk_restore_hash, clk_restore_hash);
    // atomicAdd(&my_anns_v1_metrics->clk_compute_distance, clk_compute_distance);
    atomicAdd(&my_anns_v1_metrics->clk_final, clk_final);
    atomicAdd(&my_anns_v1_metrics->clk_counter, 1UL);
  }
#endif
#ifdef _GRAPH_QUALITY_ANALYSIS
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    atomicAdd(&my_anns_v1_metrics->global_distance_calculation_counter3,
              local_distance_calculation_counter1);
    atomicAdd(&my_anns_v1_metrics->global_distance_calculation_counter4,
              local_distance_calculation_counter2);
    atomicAdd(&my_anns_v1_metrics->global_distance_calculation_counter3_4_counter, 1UL);
    // printf(
    //   "GRAPH: my_anns_v1-single-cta, file: %s, line: %d, query_id: %u, num_executed_iterations:
    //   %u, " "min_iteration: %u, max_iteration: %u, local_distance_calculation_counter1: %lu, "
    //   "local_distance_calculation_counter2: %lu\n",
    //   __FILE__,
    //   __LINE__,
    //   query_id,
    //   iter + 1,
    //   min_iteration,
    //   max_iteration,
    //   local_distance_calculation_counter1,
    //   local_distance_calculation_counter2);
    // if (query_id == 0) {
    //   printf(
    //     "GRAPH: my_anns_v1-single-cta, file: %s, line: %d, global_distance_calculation_counter1:
    //     %lu, " "global_distance_calculation_counter2: %lu, num_queries: %u\n",
    //     __FILE__,
    //     __LINE__,
    //     *graph_metrics_global_distance_calculation_counter1_ptr,
    //     *graph_metrics_global_distance_calculation_counter2_ptr,
    //     gridDim.y);
    // }
  }
#endif
  DEBUG_PRINTF("finish search_core\n");
}

// #undef NDEBUG
template <unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T,
          class EntryPointsPolicy,
          class VisitedTable>
#ifndef NDEBUG
RAFT_KERNEL search_kernel(
#else
RAFT_KERNEL __launch_bounds__(1024, 1) search_kernel(
#endif
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries, top_k]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, top_k]
  const std::uint32_t top_k,
  const DATASET_DESCRIPTOR_T* dataset_desc,
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const knn_graph,   // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,  // [num_queries]
  SAMPLE_FILTER_T sample_filter,
  const EntryPointsPolicy entry_points_policy,
  VisitedTable visited_table
#ifdef _GRAPH_QUALITY_ANALYSIS
  ,
  MyAnnsV1Metrics* my_anns_v1_metrics
#endif
)
{
  const auto query_id = blockIdx.y;
  search_core<MAX_ITOPK,
              MAX_CANDIDATES,
              TOPK_BY_BITONIC_SORT,
              DATASET_DESCRIPTOR_T,
              SAMPLE_FILTER_T,
              EntryPointsPolicy>(result_indices_ptr,
                                 result_distances_ptr,
                                 top_k,
                                 dataset_desc,
                                 queries_ptr,
                                 knn_graph,
                                 graph_degree,
                                 // num_entry_points,
                                 internal_topk,
                                 search_width,
                                 min_iteration,
                                 max_iteration,
                                 num_executed_iterations,
                                 query_id,
                                 sample_filter,
                                 entry_points_policy,
                                 visited_table
#ifdef _GRAPH_QUALITY_ANALYSIS
                                 ,
                                 my_anns_v1_metrics
#endif
  );
}

// To make sure we avoid false sharing on both CPU and GPU, we enforce cache line size to the
// maximum of the two.
// This makes sync atomic significantly faster.
constexpr size_t kCacheLineBytes = 64;

constexpr uint32_t kMaxJobsNum              = 8192;
constexpr uint32_t kMaxWorkersNum           = 4096;
constexpr uint32_t kMaxWorkersPerThread     = 256;
constexpr uint32_t kSoftMaxWorkersPerThread = 16;

template <typename DATASET_DESCRIPTOR_T>
struct alignas(kCacheLineBytes) job_desc_t {
  using index_type    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using distance_type = typename DATASET_DESCRIPTOR_T::DISTANCE_T;
  using data_type     = typename DATASET_DESCRIPTOR_T::DATA_T;
  // The algorithm input parameters
  struct value_t {
    index_type* result_indices_ptr;       // [num_queries, top_k]
    distance_type* result_distances_ptr;  // [num_queries, top_k]
    const data_type* queries_ptr;         // [num_queries, dataset_dim]
    uint32_t top_k;
    uint32_t n_queries;
  };
  using blob_elem_type = uint4;
  constexpr static inline size_t kBlobSize =
    raft::div_rounding_up_safe(sizeof(value_t), sizeof(blob_elem_type));
  // Union facilitates loading the input by a warp in a single request
  union input_t {
    blob_elem_type blob[kBlobSize];  // NOLINT
    value_t value;
  } input;
  // Last thread triggers this flag.
  cuda::atomic<bool, cuda::thread_scope_system> completion_flag;
};

struct alignas(kCacheLineBytes) worker_handle_t {
  using handle_t = uint64_t;
  struct value_t {
    uint32_t desc_id;
    uint32_t query_id;
  };
  union data_t {
    handle_t handle;
    value_t value;
  };
  cuda::atomic<data_t, cuda::thread_scope_system> data;
};
static_assert(sizeof(worker_handle_t::value_t) == sizeof(worker_handle_t::handle_t));
static_assert(
  cuda::atomic<worker_handle_t::data_t, cuda::thread_scope_system>::is_always_lock_free);

constexpr worker_handle_t::handle_t kWaitForWork = std::numeric_limits<uint64_t>::max();
constexpr worker_handle_t::handle_t kNoMoreWork  = kWaitForWork - 1;

constexpr auto is_worker_busy(worker_handle_t::handle_t h) -> bool
{
  return (h != kWaitForWork) && (h != kNoMoreWork);
}

template <bool Persistent,
          unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T,
          class EntryPointsPolicy,
          class VisitedTable>
auto dispatch_kernel = []() {
  // if constexpr (Persistent) {
  //   return search_kernel_p<MAX_ITOPK,
  //                          MAX_CANDIDATES,
  //                          TOPK_BY_BITONIC_SORT,
  //                          DATASET_DESCRIPTOR_T,
  //                          SAMPLE_FILTER_T>;
  // } else {
  return search_kernel<MAX_ITOPK,
                       MAX_CANDIDATES,
                       TOPK_BY_BITONIC_SORT,
                       DATASET_DESCRIPTOR_T,
                       SAMPLE_FILTER_T,
                       EntryPointsPolicy,
                       VisitedTable>;
  // }
}();

template <bool Persistent,
          typename DATASET_DESCRIPTOR_T,
          typename SAMPLE_FILTER_T,
          typename EntryPointsPolicy,
          typename VisitedTable>
struct search_kernel_config {
  using kernel_t = decltype(dispatch_kernel<Persistent,
                                            64,
                                            64,
                                            0,
                                            DATASET_DESCRIPTOR_T,
                                            SAMPLE_FILTER_T,
                                            EntryPointsPolicy,
                                            VisitedTable>);

  template <unsigned MAX_CANDIDATES, unsigned USE_BITONIC_SORT>
  static auto choose_search_kernel(unsigned itopk_size) -> kernel_t
  {
    if (itopk_size <= 64) {
      return dispatch_kernel<Persistent,
                             64,
                             MAX_CANDIDATES,
                             USE_BITONIC_SORT,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T,
                             EntryPointsPolicy,
                             VisitedTable>;
    } else if (itopk_size <= 128) {
      return dispatch_kernel<Persistent,
                             128,
                             MAX_CANDIDATES,
                             USE_BITONIC_SORT,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T,
                             EntryPointsPolicy,
                             VisitedTable>;
    } else if (itopk_size <= 256) {
      return dispatch_kernel<Persistent,
                             256,
                             MAX_CANDIDATES,
                             USE_BITONIC_SORT,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T,
                             EntryPointsPolicy,
                             VisitedTable>;
    } else if (itopk_size <= 512) {
      return dispatch_kernel<Persistent,
                             512,
                             MAX_CANDIDATES,
                             USE_BITONIC_SORT,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T,
                             EntryPointsPolicy,
                             VisitedTable>;
    }
    THROW("No kernel for parametels itopk_size %u, max_candidates %u", itopk_size, MAX_CANDIDATES);
  }

  static auto choose_itopk_and_mx_candidates(unsigned itopk_size,
                                             unsigned num_itopk_candidates,
                                             unsigned block_size) -> kernel_t
  {
    if (num_itopk_candidates <= 64) {
      // use bitonic sort based topk
      return choose_search_kernel<64, 1>(itopk_size);
    } else if (num_itopk_candidates <= 128) {
      return choose_search_kernel<128, 1>(itopk_size);
    } else if (num_itopk_candidates <= 256) {
      return choose_search_kernel<256, 1>(itopk_size);
    } else {
      // Radix-based topk is used
      constexpr unsigned max_candidates = 32;  // to avoid build failure
      if (itopk_size <= 256) {
        return dispatch_kernel<Persistent,
                               256,
                               max_candidates,
                               0,
                               DATASET_DESCRIPTOR_T,
                               SAMPLE_FILTER_T,
                               EntryPointsPolicy,
                               VisitedTable>;
      } else if (itopk_size <= 512) {
        return dispatch_kernel<Persistent,
                               512,
                               max_candidates,
                               0,
                               DATASET_DESCRIPTOR_T,
                               SAMPLE_FILTER_T,
                               EntryPointsPolicy,
                               VisitedTable>;
      }
    }
    THROW("No kernel for parametels itopk_size %u, num_itopk_candidates %u",
          itopk_size,
          num_itopk_candidates);
  }
};

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SampleFilterT,
          typename EntryPointsPolicy,
          class VisitedTable>
void select_and_run(const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                    raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
                    IndexT* topk_indices_ptr,       // [num_queries, topk]
                    DistanceT* topk_distances_ptr,  // [num_queries, topk]
                    const DataT* queries_ptr,       // [num_queries, dataset_dim]
                    uint32_t num_queries,
                    uint32_t* num_executed_iterations,  // [num_queries,]
                    const search_params& ps,
                    uint32_t topk,
                    uint32_t num_itopk_candidates,
                    uint32_t block_size,  //
                    uint32_t smem_size,
                    SampleFilterT sample_filter,
                    const EntryPointsPolicy& entry_points_policy,
                    VisitedTable& visited_table,
#ifdef _GRAPH_QUALITY_ANALYSIS
                    MyAnnsV1Metrics* my_anns_v1_metrics,
#endif
                    cudaStream_t stream)
{
  if (ps.persistent) {
    // TODO(jiangyinzuo): implement persistent mode
    THROW("Persistent search is not supported for the my_anns_v1 kernel.");
  } else {
    using descriptor_base_type = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
    auto kernel =
      search_kernel_config<false,
                           descriptor_base_type,
                           SampleFilterT,
                           EntryPointsPolicy,
                           VisitedTable>::choose_itopk_and_mx_candidates(ps.itopk_size,
                                                                         num_itopk_candidates,
                                                                         block_size);
    RAFT_CUDA_TRY(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    dim3 thread_dims(block_size, 1, 1);
    dim3 block_dims(1, num_queries, 1);
    RAFT_LOG_DEBUG(
      "Launching kernel with %u threads, %u block %u smem", block_size, num_queries, smem_size);
    kernel<<<block_dims, thread_dims, smem_size, stream>>>(topk_indices_ptr,
                                                           topk_distances_ptr,
                                                           topk,
                                                           dataset_desc.dev_ptr(stream),
                                                           queries_ptr,
                                                           graph.data_handle(),
                                                           graph.extent(1),
                                                           ps.itopk_size,
                                                           ps.search_width,
                                                           ps.min_iterations,
                                                           ps.max_iterations,
                                                           num_executed_iterations,
                                                           sample_filter,
                                                           entry_points_policy,
                                                           visited_table
#ifdef _GRAPH_QUALITY_ANALYSIS
                                                           ,
                                                           my_anns_v1_metrics
#endif
    );
    RAFT_CUDA_TRY(cudaPeekAtLastError());
#ifdef _GRAPH_QUALITY_ANALYSIS
    // printf("GRAPH: my_anns_v1-single-cta, file: %s, line: %d, num_queries: %u\n",
    //        __FILE__,
    //        __LINE__,
    //        num_queries);
#endif
  }
}

}  // namespace single_cta_search
}  // namespace cuvs::neighbors::my_anns_v1::detail
