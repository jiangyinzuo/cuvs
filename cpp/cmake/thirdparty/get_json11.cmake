#=============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_json11)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    rapids_cpm_find(json11      ${PKG_VERSION}
            GLOBAL_TARGETS      json11
            CPM_ARGS
            GIT_REPOSITORY         https://github.com/${PKG_FORK}/json11.git
            GIT_TAG                ${PKG_PINNED_TAG}
            EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
            )

    if(json11_ADDED)
        message(VERBOSE "cuVS: Using json11 located in ${json11_SOURCE_DIR}")
    else()
        message(VERBOSE "cuVS: Using json11 located in ${json11_DIR}")
    endif()

endfunction()

find_and_configure_json11(VERSION  1.0.0
        FORK             dropbox
        PINNED_TAG       v1.0.0
        EXCLUDE_FROM_ALL ON
        )
