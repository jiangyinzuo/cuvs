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

#include "search_warp_distance_kernel.cuh"

#include "bitonic.hpp"
#include "compute_distance-ext.cuh"
#include "device_common.cuh"
#include "graph_analysis_macros.h"
#include "hashmap.hpp"
#include "pickup_next_parents.cuh"
#include "search_plan.cuh"
#include "sort.cuh"
#include "topk_by_radix.cuh"
#include "topk_for_my_anns_v1/topk.h"  // TODO replace with raft topk if possible
#include "utils.hpp"
#include "visited_table.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/distance/distance.hpp>

#include <cuvs/neighbors/common.hpp>

// TODO: This shouldn't be invoking anything from spatial/knn
#include "../ann_utils.cuh"

#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>  // RAFT_CUDA_TRY_NOT_THROW is used TODO(tfeher): consider moving this to cuda_rt_essentials.hpp

#include "entry_points_policy.cuh"
#include "kernel_debug.cuh"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <cooperative_groups.h>

namespace cuvs::neighbors::my_anns_v1::detail {
namespace warp_distance_search {

// #define _CLK_BREAKDOWN

// #undef NDEBUG

//
// a warp for a distance computation
//
template <TopKSortType sort_type,
          uint32_t MAX_ITOPK,
          uint32_t MAX_CANDIDATES,
          uint32_t MAX_ELEMENTS,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T,
          class VisitedTable>
#ifndef NDEBUG
RAFT_KERNEL search_kernel(
#else
RAFT_KERNEL __launch_bounds__(1024, 1) search_kernel(
#endif
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries, topk]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, topk]
  const uint32_t topk,
  const DATASET_DESCRIPTOR_T* dataset_desc,
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const knn_graph,   // [dataset_size, graph_degree]
  const uint32_t graph_degree,
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* candidate_distances_buffer,  // [graph_degree]
  typename DATASET_DESCRIPTOR_T::INDEX_T* candidate_indices_buffer,       // [graph_degree]
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  const uint32_t itopk_size,
  const uint32_t search_width,
  const uint32_t min_iteration,
  const uint32_t max_iteration,
  uint32_t* const num_executed_iterations, /* stats */
  SAMPLE_FILTER_T sample_filter,
  VisitedTable visited_table
#ifdef _GRAPH_QUALITY_ANALYSIS
  ,
  MyAnnsV1Metrics* my_anns_v1_metrics
#endif
)
{
  using DATA_T     = typename DATASET_DESCRIPTOR_T::DATA_T;
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;

  const auto query_id = blockIdx.y;
  // const auto cta_id   = blockIdx.x;  // local CTA ID

#ifdef _GRAPH_QUALITY_ANALYSIS
  __shared__ uint64_t local_distance_calculation_counter1;
  __shared__ uint64_t local_distance_calculation_counter2;
  __syncthreads();
  if (threadIdx.x == 0 && cta_id == 0) {
    local_distance_calculation_counter1 = 0;
    local_distance_calculation_counter2 = 0;
  }
  if (threadIdx.x == 0 && cta_id == 0 && query_id == 0) { my_anns_v1_metrics->reset(); }
  if (METRIC_THREAD_COND()) { atomicAdd(&my_anns_v1_metrics->counter_clk_thread, 1UL); }
  __syncthreads();
#endif

#ifdef _CLK_BREAKDOWN
  uint64_t clk_init = 0;
  // uint64_t clk_compute_1st_distance = 0;
  uint64_t clk_topk           = 0;
  uint64_t clk_pickup_parents = 0;
  // uint64_t clk_compute_distance     = 0;
  uint64_t clk_start;
#define _CLK_START() clk_start = clock64()
#define _CLK_REC(V)  V += clock64() - clk_start;
#else
#define _CLK_START()
#define _CLK_REC(V)
#endif
  _CLK_START();

  extern __shared__ uint8_t smem[];

  assert(itopk_size % 32 == 0);
  // Layout of result_buffer
  // +----------------+---------+------------------------------------------+
  // | internal_top_k | padding | neighbors of parent nodes                |
  // | <itopk_size>   | upto 32 | <search_width * graph_degree>            |
  // +----------------+---------+------------------------------------------+
  // |<---                 result_buffer_size_32                       --->|
  const auto result_buffer_size    = itopk_size + search_width * graph_degree;
  const auto result_buffer_size_32 = raft::round_up_safe<uint32_t>(result_buffer_size, 32);

  // Set smem working buffer for the distance calculation
  dataset_desc = dataset_desc->setup_workspace(smem, queries_ptr, query_id);

  auto* __restrict__ result_indices_buffer =
    reinterpret_cast<INDEX_T*>(smem + dataset_desc->smem_ws_size_in_bytes());
  auto* __restrict__ result_distances_buffer =
    reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size_32);
  auto* __restrict__ parent_indices_buffer = visited_table.setup_table(
    reinterpret_cast<INDEX_T*>(result_distances_buffer + result_buffer_size_32));
  auto* __restrict__ result_position = reinterpret_cast<int*>(parent_indices_buffer + search_width);
  auto* __restrict__ topk_ws         = reinterpret_cast<std::uint32_t*>(result_position + 1);
  auto* __restrict__ terminate_flag  = reinterpret_cast<INDEX_T*>(topk_ws + 3);
  auto* __restrict__ smem_work_ptr   = reinterpret_cast<std::uint32_t*>(terminate_flag + 1);

  if (threadIdx.x == 0) {
    terminate_flag[0] = 0;
    topk_ws[0]        = ~0u;
  }

  constexpr INDEX_T invalid_index    = ~static_cast<INDEX_T>(0);
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

  for (unsigned i = threadIdx.x; i < result_buffer_size_32; i += blockDim.x) {
    result_indices_buffer[i]   = invalid_index;
    result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
  }
  __syncthreads();
  _CLK_REC(clk_init);

  print_result_buffer(
    result_buffer_size_32, result_indices_buffer, result_distances_buffer, __FILE__, __LINE__, 0);

  // compute distance to randomly selecting nodes
  _CLK_START();

  // Each warp computes a distance
  INDEX_T warp_result_index;
  DISTANCE_T warp_result_distance;
  device::compute_distance_to_one_random_node_one_warp(warp_result_index,
                                                       warp_result_distance,
                                                       *dataset_desc,
                                                       graph_degree,
                                                       num_distilation,
                                                       rand_xor_mask,
                                                       seed_ptr,
                                                       num_seeds,
                                                       visited_table
#ifdef _GRAPH_QUALITY_ANALYSIS
                                                       ,
                                                       my_anns_v1_metrics,
                                                       &local_distance_calculation_counter1,
                                                       &local_distance_calculation_counter2
#endif
  );

  print_result_buffer(
    result_buffer_size_32, result_indices_buffer, result_distances_buffer, __FILE__, __LINE__, 0);

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  grid.sync();

  const INDEX_T warp_id = threadIdx.x / warp_size() + blockIdx.x * (blockDim.x / warp_size());
  uint32_t iter         = 0;
  const bool search_width_is_1 = (search_width == 1);
  while (1) {
    DEBUG_PRINTF("iter %u", iter);
    // lead lane store the result to immediate gmem buffer
    if (threadIdx.x % warp_size() == 0 && warp_id < graph_degree) {
      candidate_distances_buffer[warp_id] = warp_result_distance;
      candidate_indices_buffer[warp_id]   = warp_result_index;
    }

    grid.sync();

    // load candidate_distances_buffer and candidate_indice_buffer to shared memory
    for (unsigned i = threadIdx.x; i < graph_degree; i += blockDim.x) {
      result_distances_buffer[i + itopk_size] = candidate_distances_buffer[i];
      result_indices_buffer[i + itopk_size]   = candidate_indices_buffer[i];
    }

    __syncthreads();

    print_result_buffer(result_buffer_size_32,
                        result_indices_buffer,
                        result_distances_buffer,
                        __FILE__,
                        __LINE__,
                        iter);
    // sort
    if constexpr (sort_type == TopKSortType::BITONIC_SORT_MERGE) {
      // [Notice]
      // It is good to use multiple warps in topk_by_bitonic_sort_and_merge() when
      // batch size is small (short-latency), but it might not be always good
      // when batch size is large (high-throughput).
      // topk_by_bitonic_sort_and_merge() consists of two operations:
      // if MAX_CANDIDATES is greater than 128, the first operation uses two warps;
      // if MAX_ITOPK is greater than 256, the second operation used two warps.
      const unsigned multi_warps_1 = ((blockDim.x >= 64) && (MAX_CANDIDATES > 128)) ? 1 : 0;
      const unsigned multi_warps_2 = ((blockDim.x >= 64) && (MAX_ITOPK > 256)) ? 1 : 0;

      // topk with bitonic sort
      _CLK_START();
      bool first_iter = (iter == 0);
      topk_by_bitonic_sort_and_merge<MAX_ITOPK, MAX_CANDIDATES, INDEX_T>(
        result_distances_buffer,
        result_indices_buffer,
        itopk_size,
        result_distances_buffer + itopk_size,
        result_indices_buffer + itopk_size,
        search_width * graph_degree,
        topk_ws,
        first_iter,
        multi_warps_1,
        multi_warps_2);
      _CLK_REC(clk_topk);
#ifdef _GRAPH_QUALITY_ANALYSIS
      if (METRIC_THREAD_COND()) { atomicAdd(&my_anns_v1_metrics->counter_topk_bitonic_sort, 1UL); }
#endif
    } else if (sort_type == TopKSortType::RADIX_SORT) {
      _CLK_START();
      // topk with radix block sort
      single_cta_search::topk_by_radix_sort<MAX_ITOPK, INDEX_T>{}(
        itopk_size,
        gridDim.x,
        result_buffer_size,
        reinterpret_cast<std::uint32_t*>(result_distances_buffer),
        result_indices_buffer,
        reinterpret_cast<std::uint32_t*>(result_distances_buffer),
        result_indices_buffer,
        nullptr,
        topk_ws,
        true,
        smem_work_ptr);
      _CLK_REC(clk_topk);
    } else if (sort_type == TopKSortType::BITONIC_SORT) {
      if (threadIdx.x < 32) {
        // [1st warp] Topk with bitonic sort
        topk_by_bitonic_sort<MAX_ELEMENTS, INDEX_T>(
          result_distances_buffer, result_indices_buffer, result_buffer_size_32);
      }
    } else {
      // assert(false);
    }

    __syncthreads();

    print_result_buffer(result_buffer_size_32,
                        result_indices_buffer,
                        result_distances_buffer,
                        __FILE__,
                        __LINE__,
                        iter);
    _CLK_REC(clk_topk);
#ifdef _GRAPH_QUALITY_ANALYSIS
    if (METRIC_THREAD_COND()) { atomicAdd(&my_anns_v1_metrics->counter_topk_bitonic_sort, 1UL); }
#endif

    if (iter + 1 >= max_iteration) {
      DEBUG_PRINTF("iter %u, max iteration reached\n", iter);
      break;
    }

    DEBUG_PRINTF("iter %u, begin pickup next parent", iter);
    _CLK_START();
    if (threadIdx.x < 32) {
      if (search_width_is_1) {
        // [1st warp] Pick up a next parent
        pickup_next_parent<sort_type, INDEX_T, DISTANCE_T>(
          parent_indices_buffer, result_indices_buffer, result_distances_buffer, itopk_size);
      } else {
        pickup_next_parents<sort_type, INDEX_T>(
          terminate_flag, parent_indices_buffer, result_indices_buffer, itopk_size, search_width);
      }
    }
    __syncthreads();

    print_result_buffer(result_buffer_size_32,
                        result_indices_buffer,
                        result_distances_buffer,
                        __FILE__,
                        __LINE__,
                        iter);

    _CLK_REC(clk_pickup_parents);
#ifdef _GRAPH_QUALITY_ANALYSIS
    if (METRIC_THREAD_COND()) { atomicAdd(&my_anns_v1_metrics->counter_pickup_parents, 1UL); }
#endif

    if ((*terminate_flag) && (iter >= min_iteration)) { break; }

#ifdef _GRAPH_QUALITY_ANALYSIS
    auto clk_insert_hashmap_start = clock64();
#endif
    // Initialize buffer for compute_distance_to_child_nodes.
    if (threadIdx.x == blockDim.x - 1) { result_position[0] = result_buffer_size_32; }
    __syncthreads();

#ifdef _GRAPH_QUALITY_ANALYSIS
    if (METRIC_THREAD_COND()) {
      auto clk_insert_hashmap = clock64() - clk_insert_hashmap_start;
      atomicAdd(&my_anns_v1_metrics->clk_insert_hashmap, (uint64_t)clk_insert_hashmap);
    }
#endif

    print_result_buffer(result_buffer_size_32,
                        result_indices_buffer,
                        result_distances_buffer,
                        __FILE__,
                        __LINE__,
                        iter);
    const AlwaysUnvisited always_unvisited;
    const auto smem_parent_id = parent_indices_buffer[warp_id / graph_degree % search_width];
    if (smem_parent_id != invalid_index) {
      const auto parent_id = result_indices_buffer[smem_parent_id] & ~index_msb_1_mask;
      // Compute the norms between child nodes and query node
      device::compute_distance_to_one_child_node_one_warp<INDEX_T,
                                                          DISTANCE_T,
                                                          DATASET_DESCRIPTOR_T,
                                                          AlwaysUnvisited,
                                                          VisitedTable>(
        warp_result_index,
        warp_result_distance,
        *dataset_desc,
        knn_graph,
        graph_degree,
        visited_table,
        parent_id,
        always_unvisited
#ifdef _GRAPH_QUALITY_ANALYSIS
        ,
        my_anns_v1_metrics,
        &local_distance_calculation_counter1,
        &local_distance_calculation_counter2,
#endif
      );
    }

    __syncthreads();
    print_result_buffer(result_buffer_size_32,
                        result_indices_buffer,
                        result_distances_buffer,
                        __FILE__,
                        __LINE__,
                        iter);
    // _CLK_REC(clk_compute_distance);

    // Filtering
    if constexpr (!std::is_same<SAMPLE_FILTER_T,
                                cuvs::neighbors::filtering::none_sample_filter>::value) {
      for (unsigned p = threadIdx.x; p < 1; p += blockDim.x) {
        if (parent_indices_buffer[p] != invalid_index) {
          const auto parent_id =
            result_indices_buffer[parent_indices_buffer[p]] & ~index_msb_1_mask;
          if (!sample_filter(query_id, parent_id)) {
            // If the parent must not be in the resulting top-k list, remove from the parent list
            result_distances_buffer[parent_indices_buffer[p]] = utils::get_max_value<DISTANCE_T>();
            result_indices_buffer[parent_indices_buffer[p]]   = invalid_index;
          }
        }
      }
      __syncthreads();
    }

    iter++;
  }

  // Filtering
  if constexpr (!std::is_same<SAMPLE_FILTER_T,
                              cuvs::neighbors::filtering::none_sample_filter>::value) {
    for (uint32_t i = threadIdx.x; i < result_buffer_size_32; i += blockDim.x) {
      INDEX_T index = result_indices_buffer[i];
      if (index == invalid_index) { continue; }
      index &= ~index_msb_1_mask;
      if (!sample_filter(query_id, index)) {
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
      }
    }
    __syncthreads();
  }

  // Output search results (1st thread block only).
  if (blockIdx.x == 0) {
    for (std::uint32_t i = threadIdx.x; i < topk; i += blockDim.x) {
      unsigned ii = i;
      if (sort_type == TopKSortType::BITONIC_SORT_MERGE) { ii = device::swizzling(i); }
      if (result_distances_ptr != nullptr) {
        result_distances_ptr[i] = result_distances_buffer[ii];
      }
      result_indices_ptr[i] =
        result_indices_buffer[i] & ~index_msb_1_mask;  // clear most significant bit
    }

    if (threadIdx.x == 0 && num_executed_iterations != nullptr) {
      num_executed_iterations[query_id] = iter + 1;
    }
    print_result_buffer(result_buffer_size_32,
                        result_indices_buffer,
                        result_distances_buffer,
                        __FILE__,
                        __LINE__,
                        iter);
    print_result_buffer(topk, result_indices_ptr, result_distances_ptr, __FILE__, __LINE__, iter);
  }
  DEBUG_PRINTF("finish warp_distance_kernel\n");
}

template <class T>
RAFT_KERNEL set_value_batch_kernel(T* const dev_ptr,
                                   const std::size_t ld,
                                   const T val,
                                   const std::size_t count,
                                   const std::size_t batch_size)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= count * batch_size) { return; }
  const auto batch_id              = tid / count;
  const auto elem_id               = tid % count;
  dev_ptr[elem_id + ld * batch_id] = val;
}

template <class T>
void set_value_batch(T* const dev_ptr,
                     const std::size_t ld,
                     const T val,
                     const std::size_t count,
                     const std::size_t batch_size,
                     cudaStream_t cuda_stream)
{
  constexpr std::uint32_t block_size = 256;
  const auto grid_size               = (count * batch_size + block_size - 1) / block_size;
  set_value_batch_kernel<T>
    <<<grid_size, block_size, 0, cuda_stream>>>(dev_ptr, ld, val, count, batch_size);
}

template <typename DATASET_DESCRIPTOR_T, typename SAMPLE_FILTER_T, typename VisitedTable>
struct search_kernel_config {
  // Search kernel function type. Note that the actual values for the template value
  // parameters do not matter, because they are not part of the function signature. The
  // second to fourth value parameters will be selected by the choose_* functions below.
  using kernel_t = decltype(&search_kernel<TopKSortType::BITONIC_SORT_MERGE,
                                           32,
                                           32,
                                           0,
                                           DATASET_DESCRIPTOR_T,
                                           SAMPLE_FILTER_T,
                                           VisitedTable>);

  static auto choose_buffer_size(unsigned result_buffer_size, unsigned block_size) -> kernel_t
  {
    if (result_buffer_size <= 64) {
      return search_kernel<TopKSortType::BITONIC_SORT,
                           0,
                           0,
                           64,
                           DATASET_DESCRIPTOR_T,
                           SAMPLE_FILTER_T,
                           VisitedTable>;
    } else if (result_buffer_size <= 128) {
      return search_kernel<TopKSortType::BITONIC_SORT,
                           0,
                           0,
                           128,
                           DATASET_DESCRIPTOR_T,
                           SAMPLE_FILTER_T,
                           VisitedTable>;
    } else if (result_buffer_size <= 256) {
      return search_kernel<TopKSortType::BITONIC_SORT,
                           0,
                           0,
                           256,
                           DATASET_DESCRIPTOR_T,
                           SAMPLE_FILTER_T,
                           VisitedTable>;
    }
    THROW("Result buffer size %u larger than max buffer size %u", result_buffer_size, 256);
  }

  template <unsigned MAX_CANDIDATES, TopKSortType sort_type>
  static auto choose_search_kernel(unsigned itopk_size) -> kernel_t
  {
    if (itopk_size <= 64) {
      return search_kernel<sort_type,
                           64,
                           MAX_CANDIDATES,
                           0,
                           DATASET_DESCRIPTOR_T,
                           SAMPLE_FILTER_T,
                           VisitedTable>;
    } else if (itopk_size <= 128) {
      return search_kernel<sort_type,
                           128,
                           MAX_CANDIDATES,
                           0,
                           DATASET_DESCRIPTOR_T,
                           SAMPLE_FILTER_T,
                           VisitedTable>;
    } else if (itopk_size <= 256) {
      return search_kernel<sort_type,
                           256,
                           MAX_CANDIDATES,
                           0,
                           DATASET_DESCRIPTOR_T,
                           SAMPLE_FILTER_T,
                           VisitedTable>;
    } else if (itopk_size <= 512) {
      return search_kernel<sort_type,
                           512,
                           MAX_CANDIDATES,
                           0,
                           DATASET_DESCRIPTOR_T,
                           SAMPLE_FILTER_T,
                           VisitedTable>;
    }
    THROW("No kernel for parametels itopk_size %u, max_candidates %u", itopk_size, MAX_CANDIDATES);
  }

  static auto choose_itopk_and_max_candidates(unsigned itopk_size,
                                              unsigned num_itopk_candidates,
                                              unsigned block_size) -> kernel_t
  {
    if (num_itopk_candidates <= 64) {
      // use bitonic sort based topk
      return choose_search_kernel<64, TopKSortType::BITONIC_SORT_MERGE>(itopk_size);
    } else if (num_itopk_candidates <= 128) {
      return choose_search_kernel<128, TopKSortType::BITONIC_SORT_MERGE>(itopk_size);
    } else if (num_itopk_candidates <= 256) {
      return choose_search_kernel<256, TopKSortType::BITONIC_SORT_MERGE>(itopk_size);
    } else {
      // Radix-based topk is used
      // constexpr unsigned max_candidates = 32;  // to avoid build failure
      if (itopk_size <= 256) {
        return search_kernel<TopKSortType::RADIX_SORT,
                             256,
                             0,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T,
                             VisitedTable>;
      } else if (itopk_size <= 512) {
        return search_kernel<TopKSortType::RADIX_SORT,
                             512,
                             0,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T,
                             VisitedTable>;
      }
    }
    THROW("No kernel for parametels itopk_size %u, num_itopk_candidates %u",
          itopk_size,
          num_itopk_candidates);
  }
};

template <typename DataT, typename IndexT, typename DistanceT, typename SampleFilterT>
void select_and_run(const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                    raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
                    IndexT* topk_indices_ptr,               // [num_queries, topk]
                    DistanceT* topk_distances_ptr,          // [num_queries, topk]
                    DistanceT* candidate_distances_buffer,  // [graph_degree]
                    IndexT* candidate_indices_buffer,       // [graph_degree]
                    const DataT* queries_ptr,               // [num_queries, dataset_dim]
                    uint32_t num_queries,
                    const IndexT* dev_seed_ptr,         // [num_queries, num_seeds]
                    uint32_t* num_executed_iterations,  // [num_queries,]
                    const search_params& ps,
                    uint32_t topk,
                    uint32_t num_itopk_candidates,
                    // multi_cta_search (params struct)
                    uint32_t block_size,  //
                    uint32_t result_buffer_size,
                    uint32_t smem_size,
                    uint32_t visited_hash_bitlen,
                    int64_t traversed_hash_bitlen,
                    IndexT* traversed_hashmap_ptr,
                    uint32_t num_cta_per_query,
                    uint32_t num_seeds,
                    SampleFilterT sample_filter,
#ifdef _GRAPH_QUALITY_ANALYSIS
                    MyAnnsV1Metrics* my_anns_v1_metrics,
#endif
                    cudaStream_t stream)
{
  // auto kernel = search_kernel_config<
  //   dataset_descriptor_base_t<DataT, IndexT, DistanceT>,
  //   SampleFilterT,
  //   visited_table::GlobalBitmap<IndexT>>::choose_itopk_and_max_candidates(ps.itopk_size,
  //                                                                         num_itopk_candidates,
  //                                                                         block_size);
  auto kernel = search_kernel_config<
    dataset_descriptor_base_t<DataT, IndexT, DistanceT>,
    SampleFilterT,
    visited_table::GlobalBitmap<IndexT>>::choose_buffer_size(result_buffer_size, block_size);

  RAFT_CUDA_TRY(
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  // Initialize hash table
  // const uint32_t traversed_hash_size = hashmap::get_size(traversed_hash_bitlen);
  cudaMemsetAsync(traversed_hashmap_ptr,
                  visited_table::GlobalBitmap<IndexT>::init_value,
                  visited_table::GlobalBitmap<IndexT>::default_size * sizeof(IndexT),
                  stream);
  // set_value_batch(traversed_hashmap_ptr,
  //                 traversed_hash_size,
  //                 visited_table::GlobalBitmap<IndexT>::init_value,
  //                 traversed_hash_size,
  //                 num_queries,
  //                 stream);

  auto graph_degree = graph.extent(1);

  RAFT_EXPECTS(block_size % warp_size() == 0, "block_size must be a multiple of warp size");
  RAFT_EXPECTS(block_size > 32, "block must has more than 1 warp");
  // a warp computes a distance (explore a parent node)
  const auto parents_per_block = block_size / warp_size();
  RAFT_EXPECTS(graph_degree % parents_per_block == 0,
               "graph.extent(1) must be a multiple of block_size / 32");

  // totally `graph_degree * ps.search_width` parents to explore
  dim3 grid_dims(graph_degree * ps.search_width / parents_per_block, num_queries, 1);
  dim3 block_dims(block_size, 1, 1);
  RAFT_LOG_DEBUG("Launching kernel with %u threads, (%u, %u) blocks %u smem",
                 block_size,
                 num_cta_per_query,
                 num_queries,
                 smem_size);
  visited_table::GlobalBitmap<IndexT> visited_table{.gmem_bitmap = traversed_hashmap_ptr};

  auto* dataset_ptr = dataset_desc.dev_ptr(stream);
  auto* graph_ptr   = graph.data_handle();
  void* args[]      = {(void*)&topk_indices_ptr,
                       (void*)&topk_distances_ptr,
                       (void*)&topk,
                       (void*)&dataset_ptr,
                       (void*)&queries_ptr,
                       (void*)&graph_ptr,
                       (void*)&graph_degree,
                       (void*)&candidate_distances_buffer,
                       (void*)&candidate_indices_buffer,
                       (void*)&ps.num_random_samplings,
                       (void*)&ps.rand_xor_mask,
                       (void*)&dev_seed_ptr,
                       (void*)&num_seeds,
                       (void*)&ps.itopk_size,
                       (void*)&ps.search_width,
                       (void*)&ps.min_iterations,
                       (void*)&ps.max_iterations,
                       (void*)&num_executed_iterations,
                       (void*)&sample_filter,
                       (void*)&visited_table
#ifdef _GRAPH_QUALITY_ANALYSIS
                  ,
                  (void*)&my_anns_v1_metrics
#endif
  };
  cudaLaunchCooperativeKernel((void*)kernel, grid_dims, block_dims, args, smem_size, stream);
  //   kernel<<<grid_dims, block_dims, smem_size, stream>>>(topk_indices_ptr,
  //                                                        topk_distances_ptr,
  //                                                        topk,
  //                                                        dataset_desc.dev_ptr(stream),
  //                                                        queries_ptr,
  //                                                        graph.data_handle(),
  //                                                        graph.extent(1),
  //                                                        candidate_distances_buffer,
  //                                                        candidate_indices_buffer,
  //                                                        ps.num_random_samplings,
  //                                                        ps.rand_xor_mask,
  //                                                        dev_seed_ptr,
  //                                                        num_seeds,
  //                                                        ps.itopk_size,
  //                                                        ps.search_width,
  //                                                        ps.min_iterations,
  //                                                        ps.max_iterations,
  //                                                        num_executed_iterations,
  //                                                        sample_filter,
  //                                                        visited_table
  // #ifdef _GRAPH_QUALITY_ANALYSIS
  //                                                        ,
  //                                                        my_anns_v1_metrics
  // #endif
  //   );
  // #ifdef _GRAPH_QUALITY_ANALYSIS
  //   printf("GRAPH: my_anns_v1-multi-cta, file: %s, line: %d, num_queries: %u\n",
  //          __FILE__,
  //          __LINE__,
  //          num_queries);
  // #endif
}

}  // namespace warp_distance_search
}  // namespace cuvs::neighbors::my_anns_v1::detail
