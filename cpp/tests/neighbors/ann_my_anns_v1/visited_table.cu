/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "../ann_my_anns_v1.cuh"
#include <gtest/gtest.h>

#include "../../../src/neighbors/detail/my_anns_v1/visited_table.cuh"

namespace cuvs::neighbors::my_anns_v1::detail {
namespace visited_table {

__global__ void test_cache_kernel(uint32_t* output_buffer)
{
  __shared__ uint32_t smem[1024];
  smem[threadIdx.x] = UINT32_MAX;
  Cache<uint32_t> cache;
  cache.s_cache = smem;
  cache.bitlen  = 10;

  if (cache.search_and_try_insert(threadIdx.x)) {
    output_buffer[threadIdx.x] = 12345;
  } else {
    output_buffer[threadIdx.x] = cache.s_cache[threadIdx.x];
  }
}

class CacheTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Initialize test data
    cudaMallocManaged(&output_buffer, 1024 * sizeof(uint32_t));
  }

  void TearDown() override
  {
    // Free test data
    cudaFree(output_buffer);
  }

  uint32_t* output_buffer;
};

TEST_F(CacheTest, TestCacheKernel)
{
  test_cache_kernel<<<1, 1024>>>(output_buffer);
  cudaDeviceSynchronize();

  // Check the output buffer
  uint32_t* h_output_buffer = new uint32_t[1024];
  cudaMemcpy(h_output_buffer, output_buffer, 1024 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 1024; ++i) {
    EXPECT_EQ(h_output_buffer[i], i);
  }

  delete[] h_output_buffer;
}

}  // namespace visited_table
}  // namespace cuvs::neighbors::my_anns_v1::detail
