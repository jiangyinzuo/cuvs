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
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/my_anns_v1.hpp>
#include <gtest/gtest.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>

#include <cstdint>

namespace cuvs::neighbors::my_anns_v1 {

template <typename T>
float to_float(T v)
{
  if constexpr (std::is_same_v<T, half>) {
    return __half2float(v);
  } else {
    return static_cast<float>(v);
  }
}

// half type may produces many same values
// 15.25 15.75 15.75 16 16.75 17.25 17.75 18 18 18.25 18.75 19 19.25 19.75 20
// 20 20.25 20.25 20.25 20.5 20.5 20.75 21 21.25 22 22 22 22.25 22.25 22.75 22.75 22.75 22.75
// 23 23.25 23.25 23.5 23.5 23.5 23.5 23.75 23.75 23.75 23.75 24 24
// 24 24.25 24.25 24.25 24.25 24.25 24.25 24.25 24.25 24.5 24.5 24.5 24.5 24.5 24.5 24.75 24.75 24.75
// 24.75 24.75 25 25 25 25 25 25
// 25 25.25 25.25 25.25 25.25 25.5 25.5 25.5 25.5 25.5 25.5 25.75 25.75 26 26 26 26
// 26 26.25 26.25 26.25 26.25 26.5 26.5
#define assert_float_lt(a, b) \
  ASSERT_GT(to_float(a), 0);  \
  ASSERT_GT(to_float(b), 0);  \
  ASSERT_LE(to_float(a), to_float(b))

struct AnnMyAnnsV1TestParams {
  my_anns_v1::search_params search_params;
  uint32_t n_queries;
};

template <typename data_type>
class AnnMyAnnsV1SingleNeighborListMultiCTAV2 : public ::testing::TestWithParam<AnnMyAnnsV1TestParams> {
 public:
  using distance_type = float;

 protected:
  raft::host_matrix<int64_t, int64_t> get_exact_neighbors()
  {
    // get exact neighbors
    auto brute_force_neighbors_host = raft::make_host_matrix<int64_t, int64_t>(res, n_queries, k);
    auto brute_force_distances_host =
      raft::make_host_matrix<distance_type, int64_t>(res, n_queries, k);
    {
      cuvs::neighbors::brute_force::index_params brute_force_index_params;
      const cuvs::neighbors::brute_force::search_params brute_force_search_params;
      const cuvs::neighbors::filtering::none_sample_filter filter{};
      const auto brute_force_index = cuvs::neighbors::brute_force::build(
        res, brute_force_index_params, raft::make_const_mdspan(dataset->view()));
      auto brute_force_distances =
        raft::make_device_matrix<distance_type, int64_t>(res, n_queries, k);
      auto brute_force_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k);
      cuvs::neighbors::brute_force::search(res,
                                           brute_force_search_params,
                                           brute_force_index,
                                           raft::make_const_mdspan(queries->view()),
                                           brute_force_neighbors.view(),
                                           brute_force_distances.view(),
                                           filter);
      raft::copy(brute_force_neighbors_host.data_handle(),
                 brute_force_neighbors.data_handle(),
                 brute_force_neighbors.size(),
                 raft::resource::get_cuda_stream(res));
      raft::copy(brute_force_distances_host.data_handle(),
                 brute_force_distances.data_handle(),
                 brute_force_distances.size(),
                 raft::resource::get_cuda_stream(res));
      raft::resource::sync_stream(res);
    }
    std::cout << "brute force neighbors & distances: " << std::endl;
    for (int i = 0; i < n_queries; ++i) {
      for (size_t j = 0; j < k; ++j) {
        std::cout << brute_force_neighbors_host(i, j) << " ";
      }
      for (size_t j = 0; j < k; ++j) {
        std::cout << brute_force_distances_host(i, j) << " ";
      }
    }
    return brute_force_neighbors_host;
  }

  void compute_recall(raft::host_matrix_view<uint32_t, int64_t> my_anns_v1_neighbors_host,
                      raft::host_matrix_view<int64_t, int64_t> brute_force_neighbors_host)
  {
    // compute recall
    uint64_t num_correct = 0;
    for (int i = 0; i < n_queries; ++i) {
      for (size_t j = 0; j < k; ++j) {
        for (size_t l = 0; l < k; ++l) {
          if (my_anns_v1_neighbors_host(i, j) == brute_force_neighbors_host(i, l)) {
            num_correct++;
            break;
          }
        }
      }
    }
    double recall = (double)num_correct / (n_queries * k);
    std::cout << "recall: " << recall << std::endl;
    if constexpr (std::is_same_v<data_type, half>) {
      ASSERT_GE(recall, 0.05) << "Recall is too low: " << recall;
    } else {
      ASSERT_GE(recall, 0.10) << "Recall is too low: " << recall;
    }
  }

  void run_warp_distance()
  {
    my_anns_v1::index_params my_anns_v1_index_params;
    my_anns_v1_index_params.graph_degree              = 64;
    my_anns_v1_index_params.intermediate_graph_degree = 96;

    my_anns_v1::index<data_type, uint32_t> my_anns_v1_index =
      my_anns_v1::build(res, my_anns_v1_index_params, raft::make_const_mdspan(dataset->view()));
    raft::resource::sync_stream(res);

    auto test_params = ::testing::TestWithParam<AnnMyAnnsV1TestParams>::GetParam();
    my_anns_v1::search_params my_anns_v1_search_params = test_params.search_params;

    std::cout << "start search" << std::endl;
    my_anns_v1::search(res,
                       my_anns_v1_search_params,
                       my_anns_v1_index,
                       raft::make_const_mdspan(queries->view()),
                       neighbors->view(),
                       distances->view());
    std::cout << "neighbors size: " << neighbors->size() << std::endl;

    auto last_error = cudaPeekAtLastError();
    raft::resource::sync_stream(res);
    std::cout << "end search" << std::endl;

    ASSERT_EQ(last_error, cudaSuccess)
      << "Error in my_anns_v1::search: " << cudaGetErrorString(last_error);

    auto my_anns_v1_neighbors_host = raft::make_host_matrix<uint32_t, size_t>(res, n_queries, k);
    {
      raft::copy(my_anns_v1_neighbors_host.data_handle(),
                 neighbors->data_handle(),
                 neighbors->size(),
                 raft::resource::get_cuda_stream(res));
      auto distances_host = raft::make_host_matrix<distance_type, size_t>(res, n_queries, k);
      raft::copy(distances_host.data_handle(),
                 distances->data_handle(),
                 distances->size(),
                 raft::resource::get_cuda_stream(res));
      raft::resource::sync_stream(res);
      std::cout << "result:" << std::endl;
      for (int i = 0; i < n_queries; ++i) {
        for (size_t j = 0; j < k; ++j) {
          std::cout << my_anns_v1_neighbors_host(i, j) << " ";
        }
        std::cout << std::endl;
        for (size_t j = 0; j < k; ++j) {
          std::cout << distances_host(i, j) << " ";
        }
        std::cout << std::endl;
        for (size_t j = 0; j < k - 1; ++j) {
          assert_float_lt(distances_host(i, j), distances_host(i, j + 1));
        }
      }
    }
    auto brute_force_neighbors_host = get_exact_neighbors();
    compute_recall(my_anns_v1_neighbors_host.view(), brute_force_neighbors_host.view());
  }

  void run()
  {
    auto test_params = ::testing::TestWithParam<AnnMyAnnsV1TestParams>::GetParam();
    my_anns_v1::search_params my_anns_v1_search_params = test_params.search_params;
    if (my_anns_v1_search_params.algo == search_algo::AUTO) {
    } else if (my_anns_v1_search_params.algo == search_algo::SINGLE_NEIGHBOR_LIST_MULTI_CTA_V1 ||
               my_anns_v1_search_params.algo == search_algo::SINGLE_NEIGHBOR_LIST_MULTI_CTA_V2) {
      run_warp_distance();
    } else {
      throw std::runtime_error("Untested search algorithm");
    }
  }

  void SetUp() override
  {
    n_queries = GetParam().n_queries;

    dataset.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_samples, n_dim));
    queries.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_queries, n_dim));
    neighbors.emplace(raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k));
    distances.emplace(raft::make_device_matrix<distance_type, int64_t>(res, n_queries, k));
    raft::random::RngState r(1244ULL);
    InitDataset(res, dataset->data_handle(), n_samples, n_dim, metric, r);
    InitDataset(res, queries->data_handle(), n_queries, n_dim, metric, r);
    raft::resource::sync_stream(res);
  }

  void TearDown() override
  {
    dataset.reset();
    queries.reset();
    neighbors.reset();
    distances.reset();
    raft::resource::sync_stream(res);
  }

 private:
  int64_t n_queries;

  raft::resources res;
  std::optional<raft::device_matrix<data_type, int64_t>> dataset       = std::nullopt;
  std::optional<raft::device_matrix<data_type, int64_t>> queries       = std::nullopt;
  std::optional<raft::device_matrix<uint32_t, int64_t>> neighbors      = std::nullopt;
  std::optional<raft::device_matrix<distance_type, int64_t>> distances = std::nullopt;

  constexpr static int64_t n_samples                   = 1183514;
  constexpr static int64_t n_dim                       = 100;
  constexpr static int64_t k                           = 10;
  constexpr static cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
};

using AnnMyAnnsV1_float = AnnMyAnnsV1SingleNeighborListMultiCTAV2<float>;

TEST_P(AnnMyAnnsV1_float, Test) { this->run(); }

static auto generate_search_params()
{
  std::vector<AnnMyAnnsV1TestParams> params_vec;
  for (auto itopk_size : {32, 48, 64, 96, 128, 256}) {
    my_anns_v1::search_params search_params;
    search_params.itopk_size        = itopk_size;
    search_params.thread_block_size = 256;
    search_params.search_width      = 1;
    search_params.max_iterations    = 0;
    search_params.algo              = search_algo::SINGLE_NEIGHBOR_LIST_MULTI_CTA_V2;
    search_params.num_entry_points  = 0;

    params_vec.push_back({search_params, 1});
  }
  return params_vec;
}

INSTANTIATE_TEST_CASE_P(AnnMyAnnsV1_SingleNeighborListMultiCTAV2_float,
                        AnnMyAnnsV1_float,
                        ::testing::ValuesIn(generate_search_params()));
}  // namespace cuvs::neighbors::my_anns_v1
