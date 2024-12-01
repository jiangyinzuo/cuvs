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

#include "../../src/neighbors/detail/hnsw_gpu/visited_table.cuh"
#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include <cuda_runtime_api.h>
#include <raft/core/detail/macros.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/itertools.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <cstdio>

namespace cuvs::neighbors::experimental::hnsw_gpu::detail {

RAFT_KERNEL test_visited_table()
{
  __shared__ std::uint32_t visited_bitset[32];
  static_assert(sizeof(visited_bitset) == 32 * 4);
  for (unsigned i = threadIdx.x; i < 32; i += blockDim.x) {
    if (i < 32) { visited_bitset[i] = 0; }
  }
  VisitedBitset<std::uint32_t> vt(visited_bitset, sizeof(visited_bitset));

  CUDA_KERNEL_ASSERT(!vt.is_visited(0));
  vt.set_visited(0);
  CUDA_KERNEL_ASSERT(vt.is_visited(0));
  vt.set_visited(threadIdx.x);
  CUDA_KERNEL_ASSERT(vt.is_visited(threadIdx.x));
  for (unsigned i = 0; i < threadIdx.x; ++i) {
    CUDA_KERNEL_ASSERT(vt.is_visited(i));
  }
  for (unsigned i = blockDim.x; i < sizeof(visited_bitset) * 8; ++i) {
    CUDA_KERNEL_ASSERT_MSG(!vt.is_visited(i), "i=%u", i);
  }
}

TEST(VisitedBitset, GetSet)
{
  test_visited_table<<<1, 8>>>();
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}
}  // namespace cuvs::neighbors::experimental::hnsw_gpu::detail
