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

typedef AnnMyAnnsV1Test<float, std::uint8_t, std::uint32_t> AnnMyAnnsV1TestU8_U32;
TEST_P(AnnMyAnnsV1TestU8_U32, AnnMyAnnsV1) { this->testMyAnnsV1(); }
typedef AnnMyAnnsV1AddNodesTest<float, std::uint8_t, std::uint32_t> AnnMyAnnsV1AddNodesTestU8_U32;
TEST_P(AnnMyAnnsV1AddNodesTestU8_U32, AnnMyAnnsV1) { this->testMyAnnsV1(); }
typedef AnnMyAnnsV1FilterTest<float, std::uint8_t, std::uint32_t> AnnMyAnnsV1FilterTestU8_U32;
TEST_P(AnnMyAnnsV1FilterTestU8_U32, AnnMyAnnsV1) { this->testMyAnnsV1(); }
typedef AnnMyAnnsV1IndexMergeTest<float, std::uint8_t, std::uint32_t> AnnMyAnnsV1IndexMergeTestU8_U32;
TEST_P(AnnMyAnnsV1IndexMergeTestU8_U32, AnnMyAnnsV1) { this->testMyAnnsV1(); }

INSTANTIATE_TEST_CASE_P(AnnMyAnnsV1Test, AnnMyAnnsV1TestU8_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnMyAnnsV1AddNodesTest,
                        AnnMyAnnsV1AddNodesTestU8_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnMyAnnsV1FilterTest,
                        AnnMyAnnsV1FilterTestU8_U32,
                        ::testing::ValuesIn(inputs_filtering));
INSTANTIATE_TEST_CASE_P(AnnMyAnnsV1IndexMergeTest,
                        AnnMyAnnsV1IndexMergeTestU8_U32,
                        ::testing::ValuesIn(inputs));

}  // namespace cuvs::neighbors::my_anns_v1
