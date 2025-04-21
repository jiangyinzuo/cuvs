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

#include "utils.hpp"

#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>

constexpr std::uint32_t PRINT_BLOCK_IDX_X = 1;
#ifndef NDEBUG
#define DEBUG_PRINTF(str, ...)                                     \
  if (threadIdx.x == 0 && blockIdx.x == PRINT_BLOCK_IDX_X && blockIdx.y == 0) {    \
    printf("%s:%d (%u, %u) (%u): " str "\n", __FILE__, __LINE__, blockIdx.x, blockIdx.y, threadIdx.x, ##__VA_ARGS__); \
  }

template <typename T>
__device__ void print_buffer(const T* buffer, uint32_t size)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
                  std::is_same_v<T, half> || std::is_same_v<T, int>,
                "T must be uint32_t, float, half or int");
  if (blockIdx.x == PRINT_BLOCK_IDX_X && blockIdx.y == 0 && threadIdx.x == 0) {
    printf("%s:%d (size=%u): ", __FILE__, __LINE__, size);
    for (uint32_t i = 0; i < size; ++i) {
      if constexpr (std::is_same_v<T, uint32_t>) {
        printf("%u ", buffer[i]);
      } else if constexpr (std::is_same_v<T, float>) {
        printf("%f ", buffer[i]);
      } else if constexpr (std::is_same_v<T, half>) {
        printf("%f ", __half_to_float(buffer[i]));
      } else if constexpr (std::is_same_v<T, int>) {
        printf("%d ", buffer[i]);
      }
    }
    printf("\n");
  }
}

#else
#define DEBUG_PRINTF(str, ...) ((void)0)

template <typename T>
__device__ __forceinline__ void print_buffer(const T* buffer, uint32_t size)
{
}

#endif

namespace cuvs::neighbors::my_anns_v1::detail {
template <typename INDEX_T, typename DISTANCE_T>
__device__ void print_result_buffer(uint32_t topk,
                                    const INDEX_T* result_indices_ptr,
                                    const DISTANCE_T* result_distances_ptr,
                                    const char* file,
                                    const unsigned line,
                                    uint32_t iter)
{
#ifndef NDEBUG
  constexpr INDEX_T invalid_index    = ~static_cast<INDEX_T>(0);
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  if (blockIdx.x == PRINT_BLOCK_IDX_X && threadIdx.x == 0 && std::is_same_v<float, DISTANCE_T> &&
      result_distances_ptr != nullptr) {
    printf("%s:%u iter=%u", file, line, iter);
    for (std::uint32_t i = 0; i < topk; ++i) {
      if (i % 16 == 0) { printf("\n"); }
      if (result_indices_ptr[i] == invalid_index) {
        printf("         ? ");
      } else {
        if (result_indices_ptr[i] & index_msb_1_mask) {
          printf("!%9u ", result_indices_ptr[i] & ~index_msb_1_mask);
        } else {
          printf("%10u ", result_indices_ptr[i] & ~index_msb_1_mask);
        }
      }
    }
    printf("\ndistance:");
    for (std::uint32_t i = 0; i < topk; ++i) {
      if (i % 16 == 0) { printf("\n"); }
      if (result_indices_ptr[i] == invalid_index) {
        printf("         ? ");
      } else {
        printf("%10f ", result_distances_ptr[i]);
      }
    }
    printf("\n");
  }
#endif
}
}  // namespace cuvs::neighbors::my_anns_v1::detail
