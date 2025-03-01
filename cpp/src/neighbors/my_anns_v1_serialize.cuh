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

#pragma once

#include "detail/my_anns_v1/my_anns_v1_serialize.cuh"

namespace cuvs::neighbors::my_anns_v1 {

#define CUVS_INST_my_anns_v1_SERIALIZE(DTYPE)                                                   \
  void serialize(raft::resources const& handle,                                            \
                 const std::string& filename,                                              \
                 const my_anns_v1::index<DTYPE, uint32_t>& index,              \
                 bool include_dataset)                                                     \
  {                                                                                        \
    my_anns_v1::detail::serialize<DTYPE, uint32_t>(                            \
      handle, filename, index, include_dataset);                                           \
  };                                                                                       \
                                                                                           \
  void deserialize(raft::resources const& handle,                                          \
                   const std::string& filename,                                            \
                   my_anns_v1::index<DTYPE, uint32_t>* index)                  \
  {                                                                                        \
    my_anns_v1::detail::deserialize<DTYPE, uint32_t>(handle, filename, index); \
  };                                                                                       \
  void serialize(raft::resources const& handle,                                            \
                 std::ostream& os,                                                         \
                 const my_anns_v1::index<DTYPE, uint32_t>& index,              \
                 bool include_dataset)                                                     \
  {                                                                                        \
    my_anns_v1::detail::serialize<DTYPE, uint32_t>(                            \
      handle, os, index, include_dataset);                                                 \
  }                                                                                        \
                                                                                           \
  void deserialize(raft::resources const& handle,                                          \
                   std::istream& is,                                                       \
                   my_anns_v1::index<DTYPE, uint32_t>* index)                  \
  {                                                                                        \
    my_anns_v1::detail::deserialize<DTYPE, uint32_t>(handle, is, index);       \
  }                                                                                        \
                                                                                           \
  void serialize_to_hnswlib(                                                               \
    raft::resources const& handle,                                                         \
    std::ostream& os,                                                                      \
    const my_anns_v1::index<DTYPE, uint32_t>& index,                           \
    std::optional<raft::host_matrix_view<const DTYPE, int64_t, raft::row_major>> dataset)  \
  {                                                                                        \
    my_anns_v1::detail::serialize_to_hnswlib<DTYPE, uint32_t>(                 \
      handle, os, index, dataset);                                                         \
  }                                                                                        \
                                                                                           \
  void serialize_to_hnswlib(                                                               \
    raft::resources const& handle,                                                         \
    const std::string& filename,                                                           \
    const my_anns_v1::index<DTYPE, uint32_t>& index,                           \
    std::optional<raft::host_matrix_view<const DTYPE, int64_t, raft::row_major>> dataset)  \
  {                                                                                        \
    my_anns_v1::detail::serialize_to_hnswlib<DTYPE, uint32_t>(                 \
      handle, filename, index, dataset);                                                   \
  }

}  // namespace cuvs::neighbors::my_anns_v1
