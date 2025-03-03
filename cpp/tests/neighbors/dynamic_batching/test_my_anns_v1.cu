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

#include "../dynamic_batching.cuh"

#include <cuvs/neighbors/my_anns_v1.hpp>

namespace cuvs::neighbors::dynamic_batching {

using my_anns_v1_F32 = dynamic_batching_test<float,
                                        uint32_t,
                                        my_anns_v1::index<float, uint32_t>,
                                        my_anns_v1::build,
                                        my_anns_v1::search>;

// using my_anns_v1_U8 = dynamic_batching_test<uint8_t,
//                                        uint32_t,
//                                        my_anns_v1::index<uint8_t, uint32_t>,
//                                        my_anns_v1::build,
//                                        my_anns_v1::search>;

template <typename fixture>
static void set_default_my_anns_v1_params(fixture& that)
{
  that.build_params_upsm.intermediate_graph_degree = 128;
  that.build_params_upsm.graph_degree              = 64;
  that.search_params_upsm.itopk_size =
    std::clamp<int64_t>(raft::bound_by_power_of_two(that.ps.k) * 16, 128, 512);
}

TEST_P(my_anns_v1_F32, single_cta)
{
  set_default_my_anns_v1_params(*this);
  search_params_upsm.algo = my_anns_v1::search_algo::SINGLE_CTA;
  build_all();
  search_all();
  check_neighbors();
}

TEST_P(my_anns_v1_F32, multi_cta)
{
  set_default_my_anns_v1_params(*this);
  search_params_upsm.algo = my_anns_v1::search_algo::MULTI_CTA;
  build_all();
  search_all();
  check_neighbors();
}

TEST_P(my_anns_v1_F32, multi_kernel)
{
  set_default_my_anns_v1_params(*this);
  search_params_upsm.algo = my_anns_v1::search_algo::MULTI_KERNEL;
  build_all();
  search_all();
  check_neighbors();
}

// TEST_P(my_anns_v1_U8, defaults)
// {
//   set_default_my_anns_v1_params(*this);
//   build_all();
//   search_all();
//   check_neighbors();
// }

INSTANTIATE_TEST_CASE_P(dynamic_batching, my_anns_v1_F32, ::testing::ValuesIn(inputs));
// INSTANTIATE_TEST_CASE_P(dynamic_batching, my_anns_v1_U8, ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::dynamic_batching
