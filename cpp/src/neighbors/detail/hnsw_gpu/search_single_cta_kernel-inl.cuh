#pragma once

#include <cstdint>
#include <raft/core/detail/macros.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

namespace cuvs::neighbors::experimental::hnsw_gpu::detail {

template <unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T>
RAFT_KERNEL __launch_bounds__(1024, 1) search_kernel(
  typename DATASET_DESCRIPTOR_T::INDEX_T* const result_indices_ptr,       // [num_queries, top_k]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, top_k]
  const std::uint32_t top_k,
  const DATASET_DESCRIPTOR_T* dataset_desc,
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const knn_graph,   // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
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
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  SAMPLE_FILTER_T sample_filter)
{
  extern __shared__ uint8_t smem[];
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
}  // namespace cuvs::neighbors::experimental::hnsw_gpu::detail
