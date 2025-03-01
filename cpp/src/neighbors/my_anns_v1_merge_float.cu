/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "my_anns_v1.cuh"
#include <cuvs/neighbors/my_anns_v1.hpp>

namespace cuvs::neighbors::my_anns_v1 {

#define RAFT_INST_my_anns_v1_MERGE(T, IdxT)                                      \
  auto merge(raft::resources const& handle,                                 \
             const my_anns_v1::merge_params& params,            \
             std::vector<my_anns_v1::index<T, IdxT>*>& indices) \
    ->my_anns_v1::index<T, IdxT>                                \
  {                                                                         \
    return my_anns_v1::merge<T, IdxT>(handle, params, indices); \
  }

RAFT_INST_my_anns_v1_MERGE(float, uint32_t);

#undef RAFT_INST_my_anns_v1_MERGE

}  // namespace cuvs::neighbors::my_anns_v1
