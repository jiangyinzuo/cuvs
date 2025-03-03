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

#include "../../../src/neighbors/detail/my_anns_v1/entry_points_policy.cuh"

namespace cuvs::neighbors::my_anns_v1::detail {

constexpr uint32_t kInternalTopKSize = 128;
class MemcpyEntryPointsTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Initialize test data
    num_entry_points = kInternalTopKSize;
    num_queries      = 16;
    cudaMallocManaged(&global_mem_result_indices_buffer,
                      num_entry_points * num_queries * sizeof(int));
    cudaMallocManaged(&global_mem_result_distances_buffer, num_entry_points * sizeof(float));
    cudaMallocManaged(&global_mem_output_indices_buffer,
                      num_entry_points * num_queries * sizeof(int));
    cudaMallocManaged(&global_mem_output_distances_buffer,
            num_entry_points * num_queries * sizeof(float));
  }

  void TearDown() override
  {
    // Free test data
    cudaFree(global_mem_result_indices_buffer);
    cudaFree(global_mem_result_distances_buffer);
  }

  uint32_t num_entry_points;
  uint32_t num_queries;
  int* global_mem_result_indices_buffer;
  float* global_mem_result_distances_buffer;

  int* global_mem_output_indices_buffer;
  float* global_mem_output_distances_buffer;
};

__global__ void test_memcpy_entry_points(MemcpyEntryPoints<int, float> entry_points,
    float* global_mem_result_distances_buffer,
    int* global_mem_result_indices_buffer)
{
  __shared__ int shared_mem_result_indices_buffer[kInternalTopKSize];
  __shared__ float shared_mem_result_distances_buffer[kInternalTopKSize];
  auto query_id = blockIdx.x;
  entry_points(query_id,
               shared_mem_result_indices_buffer,
               shared_mem_result_distances_buffer,
               kInternalTopKSize);
  __syncwarp();
  auto buffer_size = kInternalTopKSize * blockDim.x;
  for (int i = threadIdx.x; i < buffer_size; i += blockDim.x) {
    auto shared_mem_idx = i % kInternalTopKSize;
    global_mem_result_indices_buffer[i] = shared_mem_result_indices_buffer[shared_mem_idx];
    global_mem_result_distances_buffer[i] = shared_mem_result_distances_buffer[shared_mem_idx];
  }
}

TEST_F(MemcpyEntryPointsTest, OperatorCopiesMemory)
{
  // init entry points
  for (uint32_t i = 0; i < num_entry_points * num_queries; i++) {
    global_mem_result_indices_buffer[i] = i;
    global_mem_result_distances_buffer[i] = (float)i;
  }

  MemcpyEntryPoints<int, float> entry_points(
    global_mem_result_indices_buffer, global_mem_result_distances_buffer, num_entry_points);
  test_memcpy_entry_points<<<num_queries, 32>>>(entry_points, global_mem_result_distances_buffer,
                                                global_mem_result_indices_buffer);
  cudaDeviceSynchronize();

  // check results
  for (uint32_t i = 0; i < num_entry_points * num_queries; i++) {
    ASSERT_EQ(global_mem_result_indices_buffer[i], global_mem_output_indices_buffer[i]);
    ASSERT_NEAR(global_mem_result_distances_buffer[i], global_mem_output_distances_buffer[i], 1e-6);
  }
}

}  // namespace cuvs::neighbors::my_anns_v1::detail
