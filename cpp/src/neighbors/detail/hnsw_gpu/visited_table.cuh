#pragma once
#include "device_atomic_functions.h"
#include <cassert>
#include <cstdint>

#include <raft/core/detail/macros.hpp>

namespace cuvs::neighbors::experimental::hnsw_gpu::detail {

template <class INDEX_T>
class VisitedBitset {
 public:
  RAFT_DEVICE_INLINE_FUNCTION VisitedBitset(std::uint32_t* visited_bitset,
                                            std::uint32_t bitset_byte_count)
    : visited_bitset_(visited_bitset), bitset_byte_count_(bitset_byte_count)
  {
  }

  RAFT_DEVICE_INLINE_FUNCTION bool is_visited(const INDEX_T internal_id)
  {
    assert(internal_id < bitset_byte_count_ * 8);
    const std::uint32_t bit_index  = internal_id % bitset_byte_count_;
    const std::uint32_t bit_offset = internal_id / bitset_byte_count_;
    return (visited_bitset_[bit_index] & (1 << bit_offset)) != 0;
  }

  RAFT_DEVICE_INLINE_FUNCTION void set_visited(const INDEX_T internal_id)
  {
    assert(internal_id < bitset_byte_count_ * 8);
    const std::uint32_t bit_index  = internal_id % bitset_byte_count_;
    const std::uint32_t bit_offset = internal_id / bitset_byte_count_;
    atomicOr(&visited_bitset_[bit_index], 1 << bit_offset);
  }

 private:
  std::uint32_t* visited_bitset_;
  std::uint32_t bitset_byte_count_;
};

/*
 * Memory Usage: M
 * Number of Data Points: N
 * Size of Assoicated Set: C
 * Tag Size: T
 * M = N / 2^{8T} * C * T
 */
template <class INDEX_T>
class VisitedSetAssociativeCache {
 public:
  RAFT_DEVICE_INLINE_FUNCTION bool is_visited_8threads(const INDEX_T internal_id)
  {
    const std::uint64_t set_index       = internal_id % associative_set_count_;
    const std::uint8_t set_id           = internal_id / associative_set_count_;
    const std::uint64_t associative_set = associative_sets_[set_index];

    const std::uint8_t mask = 0xFF << (threadIdx.x % 8 * 8);
    const std::uint8_t tag  = associative_set & mask;

    bool visited                   = tag == set_id;
    const unsigned int ballot_mask = 0xFF << (threadIdx.x / 8 * 8);
    const unsigned int ballot      = __ballot_sync(ballot_mask, visited);
    return ballot > 0;
  }

  RAFT_DEVICE_INLINE_FUNCTION void set_visited_8threads(const INDEX_T internal_id)
  {
    const std::uint64_t set_index = internal_id % associative_set_count_;
    const std::uint8_t set_id     = internal_id / associative_set_count_;
    std::uint64_t associative_set = associative_sets_[set_index];

    const std::uint8_t mask            = 0xFF << (threadIdx.x % 8 * 8);
    const std::uint8_t tag             = associative_set & mask;
    bool is_blank                      = tag == 0;
    const unsigned int ballot_mask     = 0xFF << (threadIdx.x / 8 * 8);
    const unsigned int ballot          = __ballot_sync(ballot_mask, is_blank);

    const bool has_blank               = ballot > 0;
    const unsigned int update_tag_mask = ((1 << (threadIdx.x % 8)) - 1) << (threadIdx.x / 8 * 8);
    if ((has_blank && (is_blank && (ballot & update_tag_mask))) ||
        (!has_blank && (threadIdx.x % 8 == threadIdx.x % 64 / 8))) {
      // threadIdx.x % 64 = 0, 8+1, 16+2, 24+3, 32+4, 40+5, 48+6, 56+7
      // may lost update if 2 threads happen to update the same associative set
      associative_sets_[set_index] = (associative_set & ~mask) | set_id;
    }
  }

 private:
  std::uint64_t* associative_sets_;
  std::uint32_t associative_set_count_;
};
}  // namespace cuvs::neighbors::experimental::hnsw_gpu::detail
