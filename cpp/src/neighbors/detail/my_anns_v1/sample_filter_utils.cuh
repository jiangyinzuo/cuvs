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

#pragma once

#include "../../sample_filter.cuh"

#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::my_anns_v1::detail {

template <class my_anns_v1SampleFilterT>
struct my_anns_v1SampleFilterWithQueryIdOffset {
  const uint32_t offset;
  my_anns_v1SampleFilterT filter;

  my_anns_v1SampleFilterWithQueryIdOffset(const uint32_t offset, const my_anns_v1SampleFilterT filter)
    : offset(offset), filter(filter)
  {
  }

  _RAFT_DEVICE auto operator()(const uint32_t query_id, const uint32_t sample_id)
  {
    return filter(query_id + offset, sample_id);
  }
};

template <class my_anns_v1SampleFilterT>
struct my_anns_v1SampleFilterT_Selector {
  using type = my_anns_v1SampleFilterWithQueryIdOffset<my_anns_v1SampleFilterT>;
};
template <>
struct my_anns_v1SampleFilterT_Selector<cuvs::neighbors::filtering::none_sample_filter> {
  using type = cuvs::neighbors::filtering::none_sample_filter;
};

// A helper function to set a query id offset
template <class my_anns_v1SampleFilterT>
inline typename my_anns_v1SampleFilterT_Selector<my_anns_v1SampleFilterT>::type set_offset(
  my_anns_v1SampleFilterT filter, const uint32_t offset)
{
  typename my_anns_v1SampleFilterT_Selector<my_anns_v1SampleFilterT>::type new_filter(offset, filter);
  return new_filter;
}
template <>
inline typename my_anns_v1SampleFilterT_Selector<cuvs::neighbors::filtering::none_sample_filter>::type
set_offset<cuvs::neighbors::filtering::none_sample_filter>(
  cuvs::neighbors::filtering::none_sample_filter filter, const uint32_t)
{
  return filter;
}
}  // namespace cuvs::neighbors::my_anns_v1::detail
