/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cstdint>
#include <cuvs/neighbors/cagra.hpp>
#include <iostream>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <unordered_set>
#include <vector>

template <typename IdxT>
void check_hop_size(const std::unordered_set<IdxT> &hop, const int i, const int degree) {
    if (hop.size() != degree - 1) {
      std::cerr << __LINE__ << ": node " << i
                << " has less than (degree - 1) 1-hop "
                << "neighbors" << std::endl;
      for (auto &neighbor : hop) {
        std::cerr << neighbor << " ";
      }
      std::cerr << std::endl;
      return;
    }
}

template <typename IdxT>
void two_hop_analysis(const IdxT *graph, const int degree, const int num_nodes) {
  std::vector<std::unordered_set<IdxT>> one_hop;
  one_hop.resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    if (graph[i * degree] != 0) {
      std::cerr << "graph's first element is invalid" << std::endl;
      return;
    }
    // cagra's first element is invalid
    for (int j = 1; j < degree; ++j) {
      if (graph[i * degree + j] >= num_nodes) {
        std::cout << "starts from 1!!!" << std::endl;
      } else if (graph[i * degree + j] == num_nodes - 1) {
        std::cout << "found max Idx" << std::endl;
      }
      one_hop[i].insert(graph[i * degree + j]);
    }
  }
  std::vector<uint64_t> two_hop_count(num_nodes, degree - 1);
  std::map<uint64_t, uint64_t> two_hop_count_hist;
  uint64_t two_hop_count_sum = num_nodes * (degree - 1);

  bool show_a_min_2hop_example = false;
  for (int i = 0; i < num_nodes; ++i) {
    std::unordered_set<IdxT> two_hop_neighbors;
    two_hop_neighbors.insert(one_hop[i].begin(), one_hop[i].end());
    check_hop_size(two_hop_neighbors, i, degree);
    for (int j = 1; j < degree; ++j) {
      IdxT one_hop_neighbor = graph[i * degree + j];
      two_hop_neighbors.insert(one_hop[one_hop_neighbor].begin(), one_hop[one_hop_neighbor].end());
    }
    two_hop_count[i] += two_hop_neighbors.size();
    two_hop_count_hist[two_hop_count[i]]++;
    two_hop_count_sum += two_hop_neighbors.size();
    if (!show_a_min_2hop_example && two_hop_count[i] == degree - 1) {
      std::cout << "node " << i << " has (degree - 1) 2-hop neighbors: ";
      for (auto &neighbor : two_hop_neighbors) {
        std::cout << neighbor << " ";
      }
      std::cout << std::endl;
      show_a_min_2hop_example = true;
    }
  }

  std::cout << "two_hop_count_avg: " << two_hop_count_sum << "/" << num_nodes
            << "=" << ((double)two_hop_count_sum) / num_nodes
            << ", upper: " << (degree - 1) + (degree - 1) * (degree - 1)
            << ", lower: " << (degree - 1) << std::endl;

  std::cout << "two_hop_count_hist: " << std::endl;
  for (auto &kv : two_hop_count_hist) {
    std::cout << kv.first << ": " << kv.second << std::endl;
  }
}

int main() {
  raft::resources handle;
  cuvs::neighbors::cagra::index<float, uint32_t> index(handle);
  cuvs::neighbors::cagra::deserialize(
      handle,
      "/usr3/jiangyinzuo_data/cuvs-bench-dataset/sift-128-euclidean/index/"
      "cuvs_cagra.graph_degree32.intermediate_graph_degree128.graph_build_"
      "algoNN_DESCENT",
      &index);
  std::cout << "graph degree: " << index.graph_degree() << std::endl;
  auto graph_view = index.graph();
  // std::cout << graph_view.accessor().is_host_accessible << std::endl;
  // std::cout << graph_view.accessor().is_device_accessible << std::endl;
  // std::cout << graph_view.accessor().is_managed_accessible << std::endl;
  std::cout << "graph size: " << graph_view.extent(0) << std::endl;

  auto graph_host = raft::make_host_matrix<uint32_t, int64_t>(
      graph_view.extent(0), graph_view.extent(1));
  raft::copy(graph_host.data_handle(), graph_view.data_handle(),
             graph_view.size(), raft::resource::get_cuda_stream(handle));
  for (size_t i = 0; i < 10; ++i) {
    for (size_t j = 0; j < graph_view.extent(1); ++j) {
      std::cout << graph_host(i, j) << " ";
    }
    std::cout << std::endl;
  }
  two_hop_analysis(graph_host.data_handle(), graph_view.extent(1),
                   graph_view.extent(0));
  return 0;
}
