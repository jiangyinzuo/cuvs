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

class AnnMyAnnsV1PreComputeNorm : public ::testing::TestWithParam<my_anns_v1::search_algo> {
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
    my_anns_v1_index.precompute_entry_point_norms(res);
    auto entry_point_norms = my_anns_v1_index.query_norms();
    ASSERT_TRUE(entry_point_norms.has_value());
    ASSERT_EQ(entry_point_norms->size(), n_samples);
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

TEST_P(AnnMyAnnsV1PreComputeNorm, TestPreComputeNorm) { this->run(); }

INSTANTIATE_TEST_CASE_P(AnnMyAnnsV1PreComputeNorm,
                        AnnMyAnnsV1PreComputeNorm,
                        ::testing::Values(my_anns_v1::search_algo::SINGLE_CTA));
}
