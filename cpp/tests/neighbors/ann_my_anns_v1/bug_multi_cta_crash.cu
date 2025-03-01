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

#include <gtest/gtest.h>

#include "../ann_my_anns_v1.cuh"

#include <cuvs/neighbors/my_anns_v1.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>

#include <cstdint>

namespace cuvs::neighbors::my_anns_v1 {

class AnnMyAnnsV1BugMultiCTACrash : public ::testing::TestWithParam<my_anns_v1::search_algo> {
 public:
  using data_type = half;

 protected:
  void run()
  {
    my_anns_v1::index_params my_anns_v1_index_params;
    my_anns_v1_index_params.graph_degree              = 32;
    my_anns_v1_index_params.intermediate_graph_degree = 48;

    auto my_anns_v1_index =
      my_anns_v1::build(res, my_anns_v1_index_params, raft::make_const_mdspan(dataset->view()));
    raft::resource::sync_stream(res);

    my_anns_v1::search_params my_anns_v1_search_params;
    my_anns_v1_search_params.itopk_size        = 32;
    my_anns_v1_search_params.thread_block_size = 256;
    my_anns_v1_search_params.search_width      = 1;
    my_anns_v1_search_params.max_iterations    = 0;
    my_anns_v1_search_params.algo = ::testing::TestWithParam<my_anns_v1::search_algo>::GetParam();

    // NOTE: when using one resource/stream for everything, the bug is NOT reproducible
    raft::resources res_search;
    my_anns_v1::search(res_search,
                  my_anns_v1_search_params,
                  my_anns_v1_index,
                  raft::make_const_mdspan(queries->view()),
                  neighbors->view(),
                  distances->view());

    raft::resource::sync_stream(res_search);
  }

  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_samples, n_dim));
    queries.emplace(raft::make_device_matrix<data_type, int64_t>(res, n_queries, n_dim));
    neighbors.emplace(raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k));
    distances.emplace(raft::make_device_matrix<float, int64_t>(res, n_queries, k));
    raft::random::RngState r(1234ULL);
    InitDataset(res, dataset->data_handle(), n_samples, n_dim, metric, r);
    // NOTE: when initializing queries with "normal" data, the bug is NOT reproducible
    raft::linalg::map(
      res, queries->view(), raft::const_op<data_type>{raft::upper_bound<data_type>()});
    // InitDataset(res, queries->data_handle(), n_queries, n_dim, metric, r);
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
  raft::resources res;
  std::optional<raft::device_matrix<data_type, int64_t>> dataset  = std::nullopt;
  std::optional<raft::device_matrix<data_type, int64_t>> queries  = std::nullopt;
  std::optional<raft::device_matrix<uint32_t, int64_t>> neighbors = std::nullopt;
  std::optional<raft::device_matrix<float, int64_t>> distances    = std::nullopt;

  constexpr static int64_t n_samples                   = 1183514;
  constexpr static int64_t n_dim                       = 100;
  constexpr static int64_t n_queries                   = 30;
  constexpr static int64_t k                           = 10;
  constexpr static cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
};

TEST_P(AnnMyAnnsV1BugMultiCTACrash, AnnMyAnnsV1BugMultiCTACrash) { this->run(); }

INSTANTIATE_TEST_CASE_P(AnnMyAnnsV1BugMultiCTACrashReproducer,
                        AnnMyAnnsV1BugMultiCTACrash,
                        ::testing::Values(my_anns_v1::search_algo::MULTI_CTA));

}  // namespace cuvs::neighbors::my_anns_v1
