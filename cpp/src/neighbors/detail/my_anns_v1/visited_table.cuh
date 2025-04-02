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

// TODO: This shouldn't be invoking anything from detail outside of neighbors/
#include "device_common.cuh"
#include <raft/core/detail/macros.hpp>

namespace cuvs::neighbors::my_anns_v1::detail {
namespace visited_table {

RAFT_INLINE_FUNCTION uint32_t get_size(const uint32_t bitlen) { return 1U << bitlen; }

// CAGRA single-CTA mode, can be in shared memory or global memory
template <typename IdxT>
struct SingleMemHashtable {
  /**
   * @return true if the key already exists in the table, false otherwise
   */
  RAFT_DEVICE_INLINE_FUNCTION bool search_and_try_insert(const IdxT key)
  {
    return hashmap::insert(table, bitlen, key) == 0;
  }

  RAFT_DEVICE_INLINE_FUNCTION IdxT* setup_table(IdxT* smem)
  {
    IdxT* result = nullptr;
    if (table != nullptr) {
      // global memory
      table  = table + (hashmap::get_size(bitlen) * blockIdx.y);
      result = smem;
    } else {
      // shared memory
      table  = smem;
      result = smem + hashmap::get_size(bitlen);
    }
    hashmap::init(table, bitlen, 0);
    return result;
  }

  RAFT_DEVICE_INLINE_FUNCTION bool need_reset(std::uint32_t iter) const
  {
    return (iter + 1) % small_hash_reset_interval == 0;
  }

  RAFT_DEVICE_INLINE_FUNCTION void restore(const IdxT* itopk_indices,
                                           const uint32_t itopk_size,
                                           const uint32_t first_tid = 0)
  {
    constexpr IdxT index_msb_1_mask = utils::gen_index_msb_1_mask<IdxT>::value;
    if (threadIdx.x < first_tid) return;
    for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid) {
      auto key = itopk_indices[i] & ~index_msb_1_mask;  // clear most significant bit
      hashmap::insert(table, bitlen, key);
    }
  }

  IdxT* table;
  uint32_t bitlen;
  uint32_t small_hash_reset_interval;
};

// CAGRA multi-CTA mode
template <typename IdxT>
struct SharedGlobalMemHashtable {
  /**
   * @return true if the key already exists in the table, false otherwise
   */
  RAFT_DEVICE_INLINE_FUNCTION bool search_and_try_insert(const IdxT key)
  {
    return (hashmap::insert(smem_table, smem_bitlen, key) == 0) ||
           hashmap::search<IdxT, 1>(gmem_table, gmem_bitlen, key);
  }

  RAFT_DEVICE_INLINE_FUNCTION IdxT* setup_table(IdxT* smem)
  {
    smem_table = smem;
    hashmap::init<IdxT>(smem_table, smem_bitlen);
    gmem_table = gmem_table + (hashmap::get_size(gmem_bitlen) * blockIdx.y);
    return smem + hashmap::get_size(smem_bitlen);
  }


  IdxT* smem_table;
  uint32_t smem_bitlen;
  IdxT* gmem_table;
  uint32_t gmem_bitlen;
};

template <typename IdxT>
struct Cache {
  /**
   * @return true if the key already exists in the table, false otherwise
   */
  RAFT_DEVICE_INLINE_FUNCTION bool search_and_try_insert(const IdxT key)
  {
    static_assert(sizeof(IdxT) == 4);
    constexpr IdxT hashval_empty = ~static_cast<IdxT>(0);
    const uint32_t size          = get_size(bitlen);
    const uint32_t bit_mask      = size / 4 - 1;
    IdxT index                   = key & bit_mask;
    uint4* group                 = reinterpret_cast<uint4*>(&s_cache[index * 4]);

    uint4 keys;
    device::lds(keys, group);

    bool exists = (keys.x == key) | (keys.y == key) | (keys.z == key) | (keys.w == key);
    if (exists) return true;

    int replace_pos = threadIdx.x % 4;
    replace_pos     = (keys.x == hashval_empty) ? 0 : replace_pos;
    replace_pos     = (keys.y == hashval_empty) ? 1 : replace_pos;
    replace_pos     = (keys.z == hashval_empty) ? 2 : replace_pos;
    replace_pos     = (keys.w == hashval_empty) ? 3 : replace_pos;

    reinterpret_cast<uint32_t*>(group)[replace_pos] = key;
    return false;
  }

  RAFT_DEVICE_INLINE_FUNCTION IdxT* setup_table(IdxT* smem)
  {
    // shared memory
    s_cache = smem;
    hashmap::init(s_cache, bitlen, 0);
    return smem + hashmap::get_size(bitlen);
  }

  IdxT* s_cache;
  uint32_t bitlen;
};

}  // namespace visited_table
}  // namespace cuvs::neighbors::my_anns_v1::detail
