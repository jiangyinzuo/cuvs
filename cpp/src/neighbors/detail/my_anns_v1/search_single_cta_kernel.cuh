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

#include "compute_distance-ext.cuh"

#include <cuvs/neighbors/my_anns_v1.hpp>
#include <cuvs/neighbors/my_anns_v1_metrics.cuh>

namespace cuvs::neighbors::my_anns_v1::detail::single_cta_search {

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
                    cudaStream_t stream);

}  // namespace cuvs::neighbors::my_anns_v1::detail::single_cta_search
