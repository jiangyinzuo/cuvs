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

#include "../../../src/neighbors/detail/my_anns_v1/compute_entry_points_distance.cuh"
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
class AnnMyAnnsV1 : public ::testing::TestWithParam<AnnMyAnnsV1TestParams> {
 public:
  using distance_type = float;

 protected:
  void test_norm(uint32_t num_entry_points,
                 my_anns_v1::index<data_type, uint32_t>& index,
                 distance_type* dev_coarse_distance_buffer,  // [num_queries, itopk_size]
                 uint32_t* dev_coarse_indices_buffer,        // [num_queries, itopk_size]
                 uint32_t itopk_size)
  {
    auto stream                              = raft::resource::get_cuda_stream(res);
    rmm::device_async_resource_ref search_mr = raft::resource::get_workspace_resource(res);
    // The norm of query
    rmm::device_uvector<data_type> query_norm_dev(n_queries, stream, search_mr);
    // The distance value of cluster(list) and queries
    rmm::device_uvector<distance_type> distance_buffer_dev(
      n_queries * num_entry_points, stream, search_mr);

    ASSERT_EQ(index.dim(), n_dim);
    raft::linalg::rowNorm(query_norm_dev.data(),
                          queries->data_handle(),
                          static_cast<uint32_t>(index.dim()),
                          static_cast<uint32_t>(n_queries),
                          raft::linalg::L2Norm,
                          true,
                          stream);
    auto query_norm_host = raft::make_host_vector<data_type, int64_t>(res, n_queries);
    raft::copy(query_norm_host.data_handle(), query_norm_dev.data(), query_norm_dev.size(), stream);
    raft::resource::sync_stream(res);
    for (int i = 0; i < n_queries; ++i) {
      float temp = to_float(query_norm_host(i));
      ASSERT_GT(temp, 0) << "Norm is zero for query " << i;
    }

    raft::device_vector_view<const data_type, uint32_t> entry_point_norms_view =
      index.dataset_norms().value();
    spatial::knn::detail::utils::outer_add(query_norm_dev.data(),
                                           static_cast<uint32_t>(n_queries),
                                           entry_point_norms_view.data_handle(),
                                           num_entry_points,
                                           distance_buffer_dev.data(),
                                           stream);

    // check distance_buffer_dev
    auto distance_buffer_host =
      raft::make_host_matrix<distance_type, int64_t>(res, n_queries, num_entry_points);
    raft::copy(distance_buffer_host.data_handle(),
               distance_buffer_dev.data(),
               distance_buffer_dev.size(),
               stream);

    for (int i = 0; i < n_queries; ++i) {
      for (size_t j = 0; j < num_entry_points; ++j) {
        ASSERT_GT(distance_buffer_host(i, j), 0)
          << "Distance is zero for query " << i << " and index " << j;
      }
    }

    const float alpha = -2.0f;
    const float beta  = 1.0f;
    raft::device_matrix_view<const data_type, uint32_t, raft::layout_stride> entry_points_view =
      index.entry_points(num_entry_points);
    std::cout << "stride: " << entry_points_view.stride(0) << "x" << entry_points_view.stride(1)
              << std::endl;
    std::cout << "m, n, k: " << num_entry_points << "x" << n_queries << "x" << index.dim()
              << std::endl;
    // A: Queries, [n_queries, dim]
    // B: Entry points, [num_entry_points, dim (stride)]
    // C = A x B^T: Queries x Entry points^T (cublas is column-major, so we need to swap A and B)
    // C^T = (B^T)^T x A^T
    // m: n_queries, n: num_entry_points, k: dim
    if constexpr (std::is_same_v<data_type, float> && std::is_same_v<distance_type, float>) {
      raft::linalg::detail::cublasgemm(
        index.cublas_handle(),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        num_entry_points,
        n_queries,
        index.dim(),
        &alpha,
        entry_points_view.data_handle(),  // B^T: [dim (stride), num_entry_points]
        entry_points_view.stride(0),
        queries->data_handle(),  // A^T: [n_dim, n_queries]
        index.dim(),
        &beta,
        distance_buffer_dev.data(),  // C^T: [num_entry_points, n_queries]
        num_entry_points,
        stream.value());
    } else {
      raft::linalg::gemm(res,
                         true,
                         false,
                         num_entry_points,
                         n_queries,
                         index.dim(),
                         &alpha,
                         entry_points_view.data_handle(),
                         // entry points have padding, so we need to use the stride
                         entry_points_view.stride(0),
                         queries->data_handle(),
                         n_dim,
                         &beta,
                         distance_buffer_dev.data(),
                         num_entry_points,
                         stream);
    }
    // select `itopk_size` smallest distances from each row
    auto distance_buffer_dev_view = raft::make_device_matrix_view<distance_type, int64_t>(
      distance_buffer_dev.data(), n_queries, num_entry_points);
    cuvs::selection::select_k(res,
                              raft::make_const_mdspan(distance_buffer_dev_view),
                              std::nullopt,
                              raft::make_device_matrix_view<distance_type, int64_t>(
                                dev_coarse_distance_buffer, n_queries, itopk_size),
                              raft::make_device_matrix_view<uint32_t, int64_t>(
                                dev_coarse_indices_buffer, n_queries, itopk_size),
                              true);

    auto coarse_distance_buffer_host =
      raft::make_host_matrix<distance_type, int64_t>(res, n_queries, itopk_size);
    raft::copy(coarse_distance_buffer_host.data_handle(),
               dev_coarse_distance_buffer,
               n_queries * itopk_size,
               stream);
    auto coarse_indices_buffer_host =
      raft::make_host_matrix<uint32_t, int64_t>(res, n_queries, itopk_size);
    raft::copy(coarse_indices_buffer_host.data_handle(),
               dev_coarse_indices_buffer,
               n_queries * itopk_size,
               stream);

    raft::resource::sync_stream(res);
    for (int i = 0; i < n_queries; ++i) {
      if (i < 10) {
        for (size_t j = 0; j < itopk_size; ++j) {
          std::cout << coarse_distance_buffer_host(i, j) << " ";
        }
        std::cout << std::endl;
      }
      for (size_t j = 0; j < itopk_size - 1; ++j) {
        assert_float_lt(coarse_distance_buffer_host(i, j), coarse_distance_buffer_host(i, j + 1));
      }
    }
    std::cout << "indices: " << std::endl;
    for (int i = 0; i < n_queries; ++i) {
      if (i < 10) {
        for (size_t j = 0; j < itopk_size; ++j) {
          std::cout << coarse_indices_buffer_host(i, j) << " ";
        }
        std::cout << std::endl;
      }
      for (size_t j = 0; j < itopk_size; ++j) {
        ASSERT_LT(coarse_indices_buffer_host(i, j), n_samples)
          << "Index is out of bounds for query " << i << " and index " << j
          << ", value: " << coarse_indices_buffer_host(i, j);
      }
    }
  }

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

  void run_gemm()
  {
    my_anns_v1::index_params my_anns_v1_index_params;
    my_anns_v1_index_params.graph_degree              = 64;
    my_anns_v1_index_params.intermediate_graph_degree = 96;

    my_anns_v1::index<data_type, uint32_t> my_anns_v1_index =
      my_anns_v1::build(res, my_anns_v1_index_params, raft::make_const_mdspan(dataset->view()));
    raft::resource::sync_stream(res);
    my_anns_v1_index.precompute_dataset_norms(res);
    auto dataset_norms = my_anns_v1_index.dataset_norms();
    ASSERT_TRUE(dataset_norms.has_value());
    ASSERT_EQ(dataset_norms->size(), n_samples);
    auto dataset_norms_host = raft::make_host_vector<data_type, size_t>(res, n_samples);
    raft::copy(dataset_norms_host.data_handle(),
               dataset_norms->data_handle(),
               dataset_norms->size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    for (int i = 0; i < n_samples; ++i) {
      float temp = to_float(dataset_norms_host(i));
      ASSERT_GT(temp, 0) << "Norm is zero for index " << i;
      if (i < 10) { std::cout << temp << " "; }
    }
    std::cout << std::endl;

    auto test_params = ::testing::TestWithParam<AnnMyAnnsV1TestParams>::GetParam();

    my_anns_v1::search_params my_anns_v1_search_params = test_params.search_params;

    auto brute_force_neighbors_host = get_exact_neighbors();

    for (uint32_t num_entry_points : {0, 32, 64, 128, 256, 512, 1024}) {
      if (num_entry_points < my_anns_v1_search_params.itopk_size) {
        std::cout << "num_entry_points=" << num_entry_points
                  << " is less than itopk_size=" << my_anns_v1_search_params.itopk_size
                  << ", skipping test" << std::endl;
        continue;
      }
      my_anns_v1_search_params.num_entry_points = num_entry_points;
      auto coarse_distance_buffer               = raft::make_device_matrix<distance_type, size_t>(
        res, n_queries, my_anns_v1_search_params.itopk_size);
      auto coarse_indices_buffer = raft::make_device_matrix<uint32_t, size_t>(
        res, n_queries, my_anns_v1_search_params.itopk_size);

      if (my_anns_v1_search_params.num_entry_points > 0) {
        test_norm(my_anns_v1_search_params.num_entry_points,
                  my_anns_v1_index,
                  coarse_distance_buffer.data_handle(),
                  coarse_indices_buffer.data_handle(),
                  my_anns_v1_search_params.itopk_size);
        detail::single_cta_search::compute_entry_point_distances(
          my_anns_v1_index,
          res,
          queries->data_handle(),
          n_queries,
          my_anns_v1_search_params.num_entry_points,
          my_anns_v1_search_params.itopk_size,
          coarse_distance_buffer.data_handle(),
          coarse_indices_buffer.data_handle());
        raft::resource::sync_stream(res);
        auto coarse_distance_buffer_host = raft::make_host_matrix<distance_type, size_t>(
          res, n_queries, my_anns_v1_search_params.itopk_size);
        raft::copy(coarse_distance_buffer_host.data_handle(),
                   coarse_distance_buffer.data_handle(),
                   coarse_distance_buffer.size(),
                   raft::resource::get_cuda_stream(res));
        auto coarse_indices_buffer_host = raft::make_host_matrix<uint32_t, size_t>(
          res, n_queries, my_anns_v1_search_params.itopk_size);
        raft::copy(coarse_indices_buffer_host.data_handle(),
                   coarse_indices_buffer.data_handle(),
                   coarse_indices_buffer.size(),
                   raft::resource::get_cuda_stream(res));
        raft::resource::sync_stream(res);
        for (int i = 0; i < n_queries; ++i) {
          for (size_t j = 0; j < my_anns_v1_search_params.itopk_size - 1; ++j) {
            assert_float_lt(coarse_distance_buffer_host(i, j),
                            coarse_distance_buffer_host(i, j + 1));
          }
        }

        std::cout << "distances: " << std::endl;
        for (int i = 0; i < n_queries; ++i) {
          for (size_t j = 0; j < my_anns_v1_search_params.itopk_size - 1; ++j) {
            assert_float_lt(coarse_distance_buffer_host(i, j),
                            coarse_distance_buffer_host(i, j + 1));
          }
          if (i < 10) {
            for (size_t j = 0; j < my_anns_v1_search_params.itopk_size; ++j) {
              std::cout << to_float(coarse_distance_buffer_host(i, j)) << " ";
            }
            std::cout << std::endl;
          }
        }
        std::cout << "indices: " << std::endl;
        for (int i = 0; i < n_queries; ++i) {
          for (size_t j = 0; j < my_anns_v1_search_params.itopk_size; ++j) {
            ASSERT_LT(coarse_indices_buffer_host(i, j), n_samples)
              << "Index is out of bounds for query " << i << " and index " << j
              << ", value: " << coarse_indices_buffer_host(i, j);
          }
          if (i < 10) {
            for (size_t j = 0; j < my_anns_v1_search_params.itopk_size; ++j) {
              std::cout << coarse_indices_buffer_host(i, j) << " ";
            }
            std::cout << std::endl;
          }
        }
      }

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
        for (int i = 0; i < n_queries; ++i) {
          for (size_t j = 0; j < k - 1; ++j) {
            assert_float_lt(distances_host(i, j), distances_host(i, j + 1))
              << "num_entry_points=" << num_entry_points;
          }
        }
      }

      compute_recall(my_anns_v1_neighbors_host.view(), brute_force_neighbors_host.view());
    }
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
      run_gemm();
    } else if (my_anns_v1_search_params.algo == search_algo::WARP_DISTANCE) {
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

using AnnMyAnnsV1_half  = AnnMyAnnsV1<half>;
using AnnMyAnnsV1_float = AnnMyAnnsV1<float>;

TEST_P(AnnMyAnnsV1_half, Test) { this->run(); }
TEST_P(AnnMyAnnsV1_float, Test) { this->run(); }

static auto generate_gemm_search_params()
{
  std::vector<AnnMyAnnsV1TestParams> params_vec;
  for (auto itopk_size : {32, 48, 64, 96, 128, 256}) {
    my_anns_v1::search_params search_params;
    search_params.itopk_size        = itopk_size;
    search_params.thread_block_size = 256;
    search_params.search_width      = 1;
    search_params.max_iterations    = 0;

    params_vec.push_back({search_params, 30});
  }
  return params_vec;
}

INSTANTIATE_TEST_CASE_P(AnnMyAnnsV1PreComputeNorm_half,
                        AnnMyAnnsV1_half,
                        ::testing::ValuesIn(generate_gemm_search_params()));
INSTANTIATE_TEST_CASE_P(AnnMyAnnsV1PreComputeNorm_float,
                        AnnMyAnnsV1_float,
                        ::testing::ValuesIn(generate_gemm_search_params()));

static auto generate_warp_distance_search_params()
{
  std::vector<AnnMyAnnsV1TestParams> params_vec;
  for (auto itopk_size : {32, 48, 64, 96, 128, 256}) {
    my_anns_v1::search_params search_params;
    search_params.itopk_size        = itopk_size;
    search_params.thread_block_size = 256;
    search_params.search_width      = 1;
    search_params.max_iterations    = 0;
    search_params.algo              = search_algo::WARP_DISTANCE;

    params_vec.push_back({search_params, 1});
  }
  return params_vec;
}

INSTANTIATE_TEST_CASE_P(AnnMyAnnsV1_WarpDistance_float,
                        AnnMyAnnsV1_float,
                        ::testing::ValuesIn(generate_warp_distance_search_params()));
}  // namespace cuvs::neighbors::my_anns_v1
