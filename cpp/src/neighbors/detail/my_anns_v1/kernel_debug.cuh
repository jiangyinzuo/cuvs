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

#include <cuda_fp16.h>
#ifndef NDEBUG
#define DEBUG_PRINTF(str, ...)                                     \
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 1) {    \
    printf("%s:%d: " str "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
  }

template <typename T>
__device__ void print_buffer(const T* buffer, uint32_t size)
{
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
                  std::is_same_v<T, half> || std::is_same_v<T, int>,
                "T must be uint32_t, float, half or int");
  if (blockIdx.x == 0 && blockIdx.y == 1 && threadIdx.x == 0) {
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
