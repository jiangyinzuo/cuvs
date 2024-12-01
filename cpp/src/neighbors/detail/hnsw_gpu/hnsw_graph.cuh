#pragma once
#include <cstdint>

namespace cuvs::neighbors::experimental::hnsw_gpu::detail {

template <class DATASET_DESCRIPTOR_T>
class HNSWGraph {
 public:
  HNSWGraph(typename DATASET_DESCRIPTOR_T::INDEX_T* const inline_linked_lists,
            std::uint32_t max_level,
            std::uint32_t graph_degree,
            std::uint32_t graph_degree_layer0)
    : inline_linked_lists_(inline_linked_lists),
      max_level_(max_level),
      graph_degree_(graph_degree),
      graph_degree_layer0_(graph_degree_layer0)
  {
  }

 public:
  const typename DATASET_DESCRIPTOR_T::INDEX_T* inline_linked_lists_;
  std::uint32_t max_level_;
  std::uint32_t graph_degree_;
  std::uint32_t graph_degree_layer0_;
};

}  // namespace cuvs::neighbors::experimental::hnsw_gpu::detail
