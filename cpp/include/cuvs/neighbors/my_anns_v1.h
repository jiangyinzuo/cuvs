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

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/common.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup my_anns_v1_c_index_params C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Enum to denote which ANN algorithm is used to build my_anns_v1 graph
 *
 */
enum MyAnnsV1GraphBuildAlgo {
  /* Select build algorithm automatically */
  AUTO_SELECT,
  /* Use IVF-PQ to build all-neighbors knn graph */
  IVF_PQ,
  /* Experimental, use NN-Descent to build all-neighbors knn graph */
  NN_DESCENT,
  /* Experimental, use iterative my_anns_v1 search and optimize to build the knn graph */
  ITERATIVE_my_anns_v1_SEARCH
};

/** Parameters for VPQ compression. */
struct cuvsmy_anns_v1CompressionParams {
  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible values: [4, 5, 6, 7, 8].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the better the search
   * performance, but the lower the recall.
   */
  uint32_t pq_bits;
  /**
   * The dimensionality of the vector after compression by PQ.
   * When zero, an optimal value is selected using a heuristic.
   *
   * TODO: at the moment `dim` must be a multiple `pq_dim`.
   */
  uint32_t pq_dim;
  /**
   * Vector Quantization (VQ) codebook size - number of "coarse cluster centers".
   * When zero, an optimal value is selected using a heuristic.
   */
  uint32_t vq_n_centers;
  /** The number of iterations searching for kmeans centers (both VQ & PQ phases). */
  uint32_t kmeans_n_iters;
  /**
   * The fraction of data to use during iterative kmeans building (VQ phase).
   * When zero, an optimal value is selected using a heuristic.
   */
  double vq_kmeans_trainset_fraction;
  /**
   * The fraction of data to use during iterative kmeans building (PQ phase).
   * When zero, an optimal value is selected using a heuristic.
   */
  double pq_kmeans_trainset_fraction;
};

typedef struct cuvsmy_anns_v1CompressionParams* cuvsmy_anns_v1CompressionParams_t;

/**
 * @brief Supplemental parameters to build my_anns_v1 Index
 *
 */
struct cuvsmy_anns_v1IndexParams {
  /** Distance type. */
  cuvsDistanceType metric;
  /** Degree of input graph for pruning. */
  size_t intermediate_graph_degree;
  /** Degree of output graph. */
  size_t graph_degree;
  /** ANN algorithm to build knn graph. */
  enum MyAnnsV1GraphBuildAlgo build_algo;
  /** Number of Iterations to run if building with NN_DESCENT */
  size_t nn_descent_niter;
  /**
   * Optional: specify compression parameters if compression is desired.
   *
   * NOTE: this is experimental new API, consider it unsafe.
   */
  cuvsmy_anns_v1CompressionParams_t compression;
};

typedef struct cuvsmy_anns_v1IndexParams* cuvsmy_anns_v1IndexParams_t;

/**
 * @brief Allocate my_anns_v1 Index params, and populate with default values
 *
 * @param[in] params cuvsmy_anns_v1IndexParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsmy_anns_v1IndexParamsCreate(cuvsmy_anns_v1IndexParams_t* params);

/**
 * @brief De-allocate my_anns_v1 Index params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsmy_anns_v1IndexParamsDestroy(cuvsmy_anns_v1IndexParams_t params);

/**
 * @brief Allocate my_anns_v1 Compression params, and populate with default values
 *
 * @param[in] params cuvsmy_anns_v1CompressionParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsmy_anns_v1CompressionParamsCreate(cuvsmy_anns_v1CompressionParams_t* params);

/**
 * @brief De-allocate my_anns_v1 Compression params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsmy_anns_v1CompressionParamsDestroy(cuvsmy_anns_v1CompressionParams_t params);

/**
 * @}
 */

/**
 * @defgroup my_anns_v1_c_extend_params C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */
/**
 * @brief Supplemental parameters to extend my_anns_v1 Index
 *
 */
struct cuvsmy_anns_v1ExtendParams {
  /** The additional dataset is divided into chunks and added to the graph. This is the knob to
   * adjust the tradeoff between the recall and operation throughput. Large chunk sizes can result
   * in high throughput, but use more working memory (O(max_chunk_size*degree^2)). This can also
   * degrade recall because no edges are added between the nodes in the same chunk. Auto select when
   * 0. */
  uint32_t max_chunk_size;
};

typedef struct cuvsmy_anns_v1ExtendParams* cuvsmy_anns_v1ExtendParams_t;

/**
 * @brief Allocate my_anns_v1 Extend params, and populate with default values
 *
 * @param[in] params cuvsmy_anns_v1ExtendParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsmy_anns_v1ExtendParamsCreate(cuvsmy_anns_v1ExtendParams_t* params);

/**
 * @brief De-allocate my_anns_v1 Extend params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsmy_anns_v1ExtendParamsDestroy(cuvsmy_anns_v1ExtendParams_t params);

/**
 * @}
 */

/**
 * @defgroup my_anns_v1_c_search_params C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Enum to denote algorithm used to search my_anns_v1 Index
 *
 */
enum cuvsmy_anns_v1SearchAlgo {
  /** For large batch sizes. */
  SINGLE_CTA,
  /** For small batch sizes. */
  MULTI_CTA,
  MULTI_KERNEL,
  AUTO
};

/**
 * @brief Enum to denote Hash Mode used while searching my_anns_v1 index
 *
 */
enum cuvsmy_anns_v1HashMode { HASH, SMALL, AUTO_HASH };

/**
 * @brief Supplemental parameters to search my_anns_v1 index
 *
 */
struct cuvsmy_anns_v1SearchParams {
  /** Maximum number of queries to search at the same time (batch size). Auto select when 0.*/
  size_t max_queries;

  /** Number of intermediate search results retained during the search.
   *
   *  This is the main knob to adjust trade off between accuracy and search speed.
   *  Higher values improve the search accuracy.
   */
  size_t itopk_size;

  /** Upper limit of search iterations. Auto select when 0.*/
  size_t max_iterations;

  // In the following we list additional search parameters for fine tuning.
  // Reasonable default values are automatically chosen.

  /** Which search implementation to use. */
  enum cuvsmy_anns_v1SearchAlgo algo;

  /** Number of threads used to calculate a single distance. 4, 8, 16, or 32. */
  size_t team_size;

  /** Number of graph nodes to select as the starting point for the search in each iteration. aka
   * search width?*/
  size_t search_width;
  /** Lower limit of search iterations. */
  size_t min_iterations;

  /** Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0. */
  size_t thread_block_size;
  /** Hashmap type. Auto selection when AUTO. */
  enum cuvsmy_anns_v1HashMode hashmap_mode;
  /** Lower limit of hashmap bit length. More than 8. */
  size_t hashmap_min_bitlen;
  /** Upper limit of hashmap fill rate. More than 0.1, less than 0.9.*/
  float hashmap_max_fill_rate;

  /** Number of iterations of initial random seed node selection. 1 or more. */
  uint32_t num_random_samplings;
  /** Bit mask used for initial random seed node selection. */
  uint64_t rand_xor_mask;
};

typedef struct cuvsmy_anns_v1SearchParams* cuvsmy_anns_v1SearchParams_t;

/**
 * @brief Allocate my_anns_v1 search params, and populate with default values
 *
 * @param[in] params cuvsmy_anns_v1SearchParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsmy_anns_v1SearchParamsCreate(cuvsmy_anns_v1SearchParams_t* params);

/**
 * @brief De-allocate my_anns_v1 search params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsmy_anns_v1SearchParamsDestroy(cuvsmy_anns_v1SearchParams_t params);

/**
 * @}
 */

/**
 * @defgroup my_anns_v1_c_index C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Struct to hold address of my_anns_v1::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;

} cuvsmy_anns_v1Index;

typedef cuvsmy_anns_v1Index* cuvsmy_anns_v1Index_t;

/**
 * @brief Allocate my_anns_v1 index
 *
 * @param[in] index cuvsmy_anns_v1Index_t to allocate
 * @return my_anns_v1Error_t
 */
cuvsError_t cuvsmy_anns_v1IndexCreate(cuvsmy_anns_v1Index_t* index);

/**
 * @brief De-allocate my_anns_v1 index
 *
 * @param[in] index cuvsmy_anns_v1Index_t to de-allocate
 */
cuvsError_t cuvsmy_anns_v1IndexDestroy(cuvsmy_anns_v1Index_t index);

/**
 * @brief Get dimension of the my_anns_v1 index
 *
 * @param[in] index my_anns_v1 index
 * @param[out] dim return dimension of the index
 * @return cuvsError_t
 */
cuvsError_t cuvsmy_anns_v1IndexGetDims(cuvsmy_anns_v1Index_t index, int* dim);

/**
 * @}
 */

/**
 * @defgroup my_anns_v1_c_index_build C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Build a my_anns_v1 index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *        3. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *        4. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/my_anns_v1.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create default index params
 * cuvsmy_anns_v1IndexParams_t params;
 * cuvsError_t params_create_status = cuvsmy_anns_v1IndexParamsCreate(&params);
 *
 * // Create my_anns_v1 index
 * cuvsmy_anns_v1Index_t index;
 * cuvsError_t index_create_status = cuvsmy_anns_v1IndexCreate(&index);
 *
 * // Build the my_anns_v1 Index
 * cuvsError_t build_status = cuvsmy_anns_v1Build(res, params, &dataset, index);
 *
 * // de-allocate `params`, `index` and `res`
 * cuvsError_t params_destroy_status = cuvsmy_anns_v1IndexParamsDestroy(params);
 * cuvsError_t index_destroy_status = cuvsmy_anns_v1IndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsmy_anns_v1IndexParams_t used to build my_anns_v1 index
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[out] index cuvsmy_anns_v1Index_t Newly built my_anns_v1 index
 * @return cuvsError_t
 */
cuvsError_t cuvsmy_anns_v1Build(cuvsResources_t res,
                           cuvsmy_anns_v1IndexParams_t params,
                           DLManagedTensor* dataset,
                           cuvsmy_anns_v1Index_t index);

/**
 * @}
 */

/**
 * @defgroup my_anns_v1_c_extend_params C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Extend a my_anns_v1 index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *        3. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsmy_anns_v1ExtendParams_t used to extend my_anns_v1 index
 * @param[in] additional_dataset DLManagedTensor* additional dataset
 * @param[in,out] index cuvsmy_anns_v1Index_t my_anns_v1 index
 * @param[out] return_dataset DLManagedTensor* extended dataset
 * @return cuvsError_t
 */
cuvsError_t cuvsmy_anns_v1Extend(cuvsResources_t res,
                            cuvsmy_anns_v1ExtendParams_t params,
                            DLManagedTensor* additional_dataset,
                            cuvsmy_anns_v1Index_t index,
                            DLManagedTensor* return_dataset);

/**
 * @}
 */

/**
 * @defgroup my_anns_v1_c_index_search C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */
/**
 * @brief Search a my_anns_v1 index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`.
 *        It is also important to note that the my_anns_v1 Index must have been built
 *        with the same type of `queries`, such that `index.dtype.code ==
 * queries.dl_tensor.dtype.code` Types for input are:
 *        1. `queries`:
 *          a. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *          b. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *          c. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *          d. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *        2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/my_anns_v1.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 * DLManagedTensor queries;
 * DLManagedTensor neighbors;
 *
 * // Create default search params
 * cuvsmy_anns_v1SearchParams_t params;
 * cuvsError_t params_create_status = cuvsmy_anns_v1SearchParamsCreate(&params);
 *
 * // Search the `index` built using `cuvsmy_anns_v1Build`
 * cuvsError_t search_status = cuvsmy_anns_v1Search(res, params, index, &queries, &neighbors,
 * &distances);
 *
 * // de-allocate `params` and `res`
 * cuvsError_t params_destroy_status = cuvsmy_anns_v1SearchParamsDestroy(params);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsmy_anns_v1SearchParams_t used to search my_anns_v1 index
 * @param[in] index cuvsmy_anns_v1Index which has been returned by `cuvsmy_anns_v1Build`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 * @param[in] filter cuvsFilter input filter that can be used
              to filter queries and neighbors based on the given bitset.
 */
cuvsError_t cuvsmy_anns_v1Search(cuvsResources_t res,
                            cuvsmy_anns_v1SearchParams_t params,
                            cuvsmy_anns_v1Index_t index,
                            DLManagedTensor* queries,
                            DLManagedTensor* neighbors,
                            DLManagedTensor* distances,
                            cuvsFilter filter);

/**
 * @}
 */

/**
 * @defgroup my_anns_v1_c_serialize my_anns_v1 C-API serialize functions
 * @{
 */
/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.c}
 * #include <cuvs/neighbors/my_anns_v1.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create an index with `cuvsmy_anns_v1Build`
 * cuvsmy_anns_v1Serialize(res, "/path/to/index", index, true);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the file name for saving the index
 * @param[in] index my_anns_v1 index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 *
 */
cuvsError_t cuvsmy_anns_v1Serialize(cuvsResources_t res,
                               const char* filename,
                               cuvsmy_anns_v1Index_t index,
                               bool include_dataset);

/**
 * Save the my_anns_v1 index to file in hnswlib format.
 * NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
 *       as the serialization format is not compatible with the original hnswlib.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/my_anns_v1.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // create an index with `cuvsmy_anns_v1Build`
 * cuvsmy_anns_v1SerializeHnswlib(res, "/path/to/index", index);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the file name for saving the index
 * @param[in] index my_anns_v1 index
 *
 */
cuvsError_t cuvsmy_anns_v1SerializeToHnswlib(cuvsResources_t res,
                                        const char* filename,
                                        cuvsmy_anns_v1Index_t index);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index my_anns_v1 index loaded disk
 */
cuvsError_t cuvsmy_anns_v1Deserialize(cuvsResources_t res, const char* filename, cuvsmy_anns_v1Index_t index);
/**
 * @}
 */
#ifdef __cplusplus
}
#endif
