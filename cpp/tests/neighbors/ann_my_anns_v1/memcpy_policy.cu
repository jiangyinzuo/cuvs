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
    num_queries      = 16;
    cudaMallocManaged(&global_mem_result_indices_buffer,
                      kInternalTopKSize * num_queries * sizeof(int));
    cudaMallocManaged(&global_mem_result_distances_buffer,
                      kInternalTopKSize * num_queries * sizeof(float));
    cudaMallocManaged(&global_mem_output_indices_buffer,
                      kInternalTopKSize * num_queries * sizeof(int));
    cudaMallocManaged(&global_mem_output_distances_buffer,
                      kInternalTopKSize * num_queries * sizeof(float));
    cudaMemset(global_mem_output_indices_buffer, 0, kInternalTopKSize * num_queries * sizeof(int));
    cudaMemset(
      global_mem_output_distances_buffer, 0, kInternalTopKSize * num_queries * sizeof(float));
  }

  void TearDown() override
  {
    // Free test data
    cudaFree(global_mem_result_indices_buffer);
    cudaFree(global_mem_result_distances_buffer);
  }

  uint32_t num_queries;
  int* global_mem_result_indices_buffer;
  float* global_mem_result_distances_buffer;

  int* global_mem_output_indices_buffer;
  float* global_mem_output_distances_buffer;
};

struct MyMemcpy {
  __device__ __forceinline__ void operator()(const std::uint32_t query_id,
                                                int* __restrict__ smem_result_indices_buffer,
                                                float* __restrict__ smem_result_distances_buffer,
                                                const std::uint32_t itopk_size)
  {
    assert(itopk_size % 4 == 0);
    // async copy from global memory to shared memory
    // every thread copies 16 bytes
    constexpr int BYTES_PER_THREAD               = 16;
    constexpr int WARP_SIZE                      = 32;
    const int indices_buffer_elements_per_thread = BYTES_PER_THREAD / sizeof(int);
    auto global_buffer_offset                    = query_id * kInternalTopKSize;
    for (uint32_t i = 0; i < itopk_size; i += WARP_SIZE * indices_buffer_elements_per_thread) {
      uint32_t idx = i + threadIdx.x * indices_buffer_elements_per_thread;
      bool pred    = idx < itopk_size;
      cp_async4_pred(smem_result_indices_buffer + idx,
                     global_mem_result_indices_buffer + global_buffer_offset + idx,
                     pred);
    }

    const int distances_buffer_elements_per_thread = BYTES_PER_THREAD / sizeof(float);
    for (uint32_t i = 0; i < itopk_size; i += WARP_SIZE * distances_buffer_elements_per_thread) {
      uint32_t idx = i + threadIdx.x * distances_buffer_elements_per_thread;
      bool pred    = idx < itopk_size;
      cp_async4_pred(smem_result_distances_buffer + idx,
                     global_mem_result_distances_buffer + global_buffer_offset + idx,
                     pred);
    }
    cp_async_wait_all();
    if (blockIdx.y == 0 && threadIdx.x == 0) {
      printf("operator() end itopk_size=%d!!\n", itopk_size);
      for (uint32_t i = 0; i < itopk_size; ++i) {
        printf("smem_result_indices_buffer[%d]=%u\n", i, smem_result_indices_buffer[i]);
        printf("smem_result_distances_buffer[%d]=%f\n", i, smem_result_distances_buffer[i]);
      }
      printf("operator() end itopk_size=%d!!\n", itopk_size);
    }
  }

  const int* __restrict__ global_mem_result_indices_buffer;      // [num_queries, itopk_size]
  const float* __restrict__ global_mem_result_distances_buffer;  // [num_queries, itopk_size]
  uint32_t kInternalTopkSize;
};

template<typename MemCpyOp>
__global__ void test_memcpy_entry_points(MemCpyOp entry_points,
                                         float* global_mem_output_distances_buffer,
                                         int* global_mem_output_indices_buffer)
{
  __shared__ int shared_mem_result_indices_buffer[kInternalTopKSize];
  __shared__ float shared_mem_result_distances_buffer[kInternalTopKSize];
  auto query_id = blockIdx.x;
  entry_points(query_id,
               shared_mem_result_indices_buffer,
               shared_mem_result_distances_buffer,
               kInternalTopKSize);
  // for (int i = threadIdx.x; i < kInternalTopKSize; i += blockDim.x) {
  //   shared_mem_result_indices_buffer[i]   =
  //   entry_points.global_mem_result_indices_buffer[query_id * kInternalTopKSize + i];
  //   shared_mem_result_distances_buffer[i] =
  //   entry_points.global_mem_result_distances_buffer[query_id * kInternalTopKSize + i]; if
  //   (threadIdx.x == 3 && query_id == 0) {
  //     for (int i = 0; i < kInternalTopKSize; ++i)
  //       printf("%d ", shared_mem_result_indices_buffer[i]);
  //   }
  // }
  __syncthreads();
  for (int i = threadIdx.x; i < kInternalTopKSize; i += blockDim.x) {
    global_mem_output_indices_buffer[query_id * kInternalTopKSize + i] =
      shared_mem_result_indices_buffer[i];
    global_mem_output_distances_buffer[query_id * kInternalTopKSize + i] =
      shared_mem_result_distances_buffer[i];
  }
}

TEST_F(MemcpyEntryPointsTest, OperatorCopiesMemory)
{
  // init entry points
  for (uint32_t i = 0; i < kInternalTopKSize * num_queries; i++) {
    global_mem_result_indices_buffer[i]   = i;
    global_mem_result_distances_buffer[i] = (float)i;
  }
  ASSERT_EQ(global_mem_result_distances_buffer[128], 128);

  MemcpyEntryPoints<int, float> entry_points(
    global_mem_result_indices_buffer, global_mem_result_distances_buffer, kInternalTopKSize);
  // MyMemcpy my_memcpy{
  //   global_mem_result_indices_buffer, global_mem_result_distances_buffer, kInternalTopkSize};

  test_memcpy_entry_points<<<num_queries, 32>>>(
    entry_points, global_mem_output_distances_buffer, global_mem_output_indices_buffer);
  cudaDeviceSynchronize();

  // check results
  for (uint32_t i = 0; i < kInternalTopKSize * num_queries; i++) {
    EXPECT_NEAR(global_mem_result_distances_buffer[i], global_mem_output_distances_buffer[i], 1e-6)
      << "i=" << i;
    ASSERT_EQ(global_mem_result_indices_buffer[i], global_mem_output_indices_buffer[i])
      << "i=" << i;
    if (i < 10) {
      std::cout << "global_mem_output_indices_buffer[" << i
                << "] = " << global_mem_output_indices_buffer[i] << std::endl;
      std::cout << "global_mem_output_distances_buffer[" << i
                << "] = " << global_mem_output_distances_buffer[i] << std::endl;
    }
  }
}

}  // namespace cuvs::neighbors::my_anns_v1::detail
