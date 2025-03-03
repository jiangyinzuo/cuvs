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
#include "cuvs/neighbors/my_anns_v1.hpp"
#include <raft/linalg/gemm.cuh>  // raft::linalg::gemm
#include <raft/linalg/norm.cuh>  // raft::linalg::norm

// TODO: This shouldn't be calling spatial/knn apis
#include "../ann_utils.cuh"
#include "cuvs/selection/select_k.hpp"

namespace cuvs::neighbors::my_anns_v1::detail {
namespace single_cta_search {
template <typename T, typename IdxT, typename DistanceT>
void compute_entry_point_distances(
  const index<T, IdxT>& index,
  raft::resources const& res,
  const T* const dev_queries,  // [num_queries, dataset_dim]
  const uint32_t num_queries,  // number of queries
  const uint32_t num_entry_points,
  const size_t itopk_size,
  DistanceT* dev_coarse_distance_buffer,  // [num_queries, itopk_size]
  IdxT* dev_coarse_indices_buffer         // [num_queries, itopk_size]
)
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                "Only float and half are supported for now");
  static_assert(std::is_same_v<DistanceT, float> || std::is_same_v<DistanceT, half>,
                "Only float and half are supported for now");

  auto stream                              = raft::resource::get_cuda_stream(res);
  rmm::device_async_resource_ref search_mr = raft::resource::get_workspace_resource(res);

  const DistanceT alpha = -2.0f;
  const DistanceT beta  = 1.0f;
  // The norm of query
  rmm::device_uvector<DistanceT> query_norm_dev(num_queries, stream, search_mr);
  // The distance value of cluster(list) and queries
  rmm::device_uvector<DistanceT> distance_buffer_dev(
    num_queries * num_entry_points, stream, search_mr);

  auto distance_buffer_dev_view = raft::make_device_matrix_view<DistanceT, int64_t>(
    distance_buffer_dev.data(), num_queries, num_entry_points);

  raft::linalg::rowNorm(query_norm_dev.data(),
                        dev_queries,
                        static_cast<IdxT>(index.dim()),
                        static_cast<IdxT>(num_queries),
                        raft::linalg::L2Norm,
                        true,
                        stream);
  raft::device_vector_view<const T, uint32_t> entry_point_norms_view = index.query_norms().value();
  spatial::knn::detail::utils::outer_add(query_norm_dev.data(),
                                         static_cast<IdxT>(num_queries),
                                         entry_point_norms_view.data_handle(),
                                         num_entry_points,
                                         distance_buffer_dev.data(),
                                         stream);
  RAFT_LOG_TRACE_VEC(entry_point_norms()->data_handle(), std::min<uint32_t>(20, dim()));
  RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), std::min<uint32_t>(20, num_entry_points));

  raft::device_matrix_view<const T, uint32_t, raft::row_major> entry_points_view =
    index.entry_points(num_entry_points);
  // A: Queries, [n_queries, dim]
  // B: Entry points, [num_entry_points, dim]
  // A x B^T: Queries x Entry points^T (cublas is column-major, so we need to swap A and B)
  // m: n_queries, n: num_entry_points, k: dim
  raft::linalg::gemm(res,
                     true,
                     false,
                     num_entry_points,
                     num_queries,
                     index.dim(),
                     &alpha,
                     entry_points_view.data_handle(),
                     // entry points have padding, so we need to use the stride
                     entry_points_view.stride(0),
                     dev_queries,
                     index.dim(),
                     &beta,
                     distance_buffer_dev.data(),
                     num_entry_points,
                     stream);
  RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), std::min<uint32_t>(20, num_entry_points));

  // select `itopk_size` smallest distances from each row
  cuvs::selection::select_k(res,
                            raft::make_const_mdspan(distance_buffer_dev_view),
                            std::nullopt,
                            raft::make_device_matrix_view<DistanceT, int64_t>(
                              dev_coarse_distance_buffer, num_queries, itopk_size),
                            raft::make_device_matrix_view<uint32_t, int64_t>(
                              dev_coarse_indices_buffer, num_queries, itopk_size),
                            true);

  RAFT_LOG_TRACE_VEC(dev_coarse_distance_buffer, num_entry_points);
  RAFT_LOG_TRACE_VEC(dev_coarse_indices_buffer, num_entry_points);
}
}  // namespace single_cta_search

}  // namespace cuvs::neighbors::my_anns_v1::detail
