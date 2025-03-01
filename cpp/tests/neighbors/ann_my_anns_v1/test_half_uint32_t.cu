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

#include <gtest/gtest.h>

#include "../ann_my_anns_v1.cuh"

namespace cuvs::neighbors::my_anns_v1 {

typedef AnnMyAnnsV1Test<float, half, std::uint32_t> AnnMyAnnsV1TestF16_U32;
TEST_P(AnnMyAnnsV1TestF16_U32, AnnMyAnnsV1) { this->testMyAnnsV1(); }

typedef AnnMyAnnsV1IndexMergeTest<float, half, std::uint32_t> AnnMyAnnsV1IndexMergeTestF16_U32;
TEST_P(AnnMyAnnsV1IndexMergeTestF16_U32, AnnMyAnnsV1IndexMerge) { this->testMyAnnsV1(); }

INSTANTIATE_TEST_CASE_P(AnnMyAnnsV1Test, AnnMyAnnsV1TestF16_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnMyAnnsV1IndexMergeTest,
                        AnnMyAnnsV1IndexMergeTestF16_U32,
                        ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::my_anns_v1
