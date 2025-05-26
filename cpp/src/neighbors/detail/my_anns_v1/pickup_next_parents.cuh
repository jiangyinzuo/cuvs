/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <cstdint>
#include <raft/core/detail/macros.hpp>
#include "sort.cuh"
#include "utils.hpp"
#include "device_common.cuh"
#include <raft/util/cudart_utils.hpp>  // RAFT_CUDA_TRY_NOT_THROW is used TODO(tfeher): consider moving this to cuda_rt_essentials.hpp

namespace cuvs::neighbors::my_anns_v1::detail {

template <TopKSortType sort_type , class INDEX_T>
RAFT_DEVICE_INLINE_FUNCTION void pickup_next_parents(std::uint32_t* const terminate_flag,
                                                     INDEX_T* const next_parent_indices,
                                                     INDEX_T* const internal_topk_indices,
                                                     const std::size_t internal_topk_size,
                                                     const std::uint32_t search_width)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  // if (threadIdx.x >= 32) return;

  for (std::uint32_t i = threadIdx.x; i < search_width; i += 32) {
    next_parent_indices[i] = utils::get_max_value<INDEX_T>();
  }
  std::uint32_t itopk_max = internal_topk_size;
  if (itopk_max % 32) { itopk_max += 32 - (itopk_max % 32); }
  std::uint32_t num_new_parents = 0;
  for (std::uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
    std::uint32_t jj = j;
    if (sort_type == TopKSortType::BITONIC_SORT_MERGE) { jj = device::swizzling(j); }
    INDEX_T index;
    int new_parent = 0;
    if (j < internal_topk_size) {
      index = internal_topk_indices[jj];
      if ((index & index_msb_1_mask) == 0) {  // check if most significant bit is set
        new_parent = 1;
      }
    }
    const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
    if (new_parent) {
      const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_new_parents;
      if (i < search_width) {
        next_parent_indices[i] = jj;
        // set most significant bit as used node
        internal_topk_indices[jj] |= index_msb_1_mask;
      }
    }
    num_new_parents += __popc(ballot_mask);
    if (num_new_parents >= search_width) { break; }
  }
  if (threadIdx.x == 0 && (num_new_parents == 0)) { *terminate_flag = 1; }
}

template <TopKSortType sort_type, class INDEX_T, class DISTANCE_T>
RAFT_DEVICE_INLINE_FUNCTION void pickup_next_parent(
  INDEX_T* const next_parent_indices,
  INDEX_T* const itopk_indices,       // [internal_topk_size]
  DISTANCE_T* const itopk_distances,  // [internal_topk_size]
  const std::size_t internal_topk_size)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  constexpr INDEX_T invalid_index    = ~static_cast<INDEX_T>(0);

  const unsigned warp_id = threadIdx.x / warp_size();
  if (warp_id > 0) { return; }
  if (threadIdx.x == 0) { next_parent_indices[0] = invalid_index; }
  __syncwarp();

  int j = -1;
  for (unsigned i = threadIdx.x; i < internal_topk_size; i += warp_size()) {
    std::uint32_t ii = i;
    if (sort_type == TopKSortType::BITONIC_SORT_MERGE) { ii = device::swizzling(i); }
    INDEX_T index    = itopk_indices[ii];
    int is_invalid   = 0;
    int is_candidate = 0;
    if (index == invalid_index) {
      is_invalid = 1;
    } else if (index & index_msb_1_mask) {
    } else {
      is_candidate = 1;
    }

    const auto ballot_mask  = __ballot_sync(0xffffffff, is_candidate);
    const auto candidate_id = __popc(ballot_mask & ((1 << threadIdx.x) - 1));
    for (int k = 0; k < __popc(ballot_mask); k++) {
      int flag_done = 0;
      if (is_candidate && candidate_id == k) {
        is_candidate = 0;
        // Use this candidate as next parent
        index |= index_msb_1_mask;  // set most significant bit as used node
        next_parent_indices[0] = i;
        itopk_indices[ii]      = index;
        flag_done              = 1;
      }
      if (__any_sync(0xffffffff, (flag_done > 0))) { return; }
    }
    j = 31 - __clz(__ballot_sync(0xffffffff, is_invalid));
    if (j < 0) { return; }
  }
}

}  // namespace cuvs::neighbors::my_anns_v1::detail
