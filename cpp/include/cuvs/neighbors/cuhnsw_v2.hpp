// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

#include <cmath>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <string>
#include <utility>
#include <vector>

#include "cuvs/neighbors/cuhnsw_v2_types.hpp"

namespace cuhnsw_v2 {

// for the compatibility with hnswlib
// following two functions refer to
// https://github.com/nmslib/hnswlib/blob/
// 2571bdb6ef3f91d6f4c2e59178fde49055d2f980/hnswlib/hnswlib.h
template <typename T>
static void writeBinaryPOD(std::ostream& out, const T& podRef)
{
  out.write(reinterpret_cast<const char*>(&podRef), sizeof(T));
}
template <typename T>
static void readBinaryPOD(std::istream& in, T& podRef)
{
  in.read(reinterpret_cast<char*>(&podRef), sizeof(T));
}

class LevelGraph {
 public:
  void SetNodes(std::vector<int>& nodes, int num_data, int ef_construction)
  {
    nodes_     = nodes;
    num_nodes_ = nodes_.size();
    neighbors_.clear();
    neighbors_.resize(num_nodes_);
    nodes_idmap_.resize(num_data);
    std::fill(nodes_idmap_.begin(), nodes_idmap_.end(), -1);
    for (int i = 0; i < num_nodes_; ++i)
      nodes_idmap_[nodes[i]] = i;
  }

  const std::vector<std::pair<float, int>>& GetNeighbors(int node) const
  {
    int nodeid = GetNodeId(node);
    return neighbors_[nodeid];
  }

  const std::vector<int>& GetNodes() const { return nodes_; }

  void ClearEdges(int node) { neighbors_[GetNodeId(node)].clear(); }

  void AddEdge(int src, int dst, float dist)
  {
    if (src == dst) return;
    int srcid = GetNodeId(src);
    neighbors_[srcid].emplace_back(dist, dst);
  }

  inline int GetNodeId(int node) const
  {
    int nodeid = nodes_idmap_.at(node);
    if (not(nodeid >= 0 and nodeid < num_nodes_)) {
      RAFT_LOG_ERROR("invalid nodeid: %d, node: %d, num_nodes: %d", nodeid, node, num_nodes_);
    }
    return nodeid;
  }

  void ShowGraph()
  {
    for (int i = 0; i < num_nodes_; ++i) {
      std::cout << std::string(50, '=') << std::endl;
      printf("nodeid %d: %d\n", i, nodes_[i]);
      for (auto& nb : GetNeighbors(nodes_[i])) {
        printf("neighbor id: %d, dist: %f\n", nb.second, nb.first);
      }
      std::cout << std::string(50, '=') << std::endl;
    }
  }

 private:
  std::vector<int> nodes_;
  std::vector<std::vector<std::pair<float, int>>> neighbors_;
  int num_nodes_ = 0;
  std::vector<int> nodes_idmap_;
};  // class LevelGraph

class CuHNSW {
 public:
  using NeighborIdxT = int;
  CuHNSW();
  ~CuHNSW();

  bool Init(int max_m,
            int max_m0,
            bool save_remains,
            int ef_construction,
            // float level_mult,
            // int batch_size,
            int block_dim,
            int hyper_threads,
            int visited_table_size,
            int visited_list_size,
            float heuristic_coef,
            enum DIST_TYPE dist_type,
            bool reverse_cand,
            int log_level);
  void SetData(const float* data, int num_data, int num_dims);
  void SetRandomLevels(const int* levels);
  void BuildGraph();
  void SaveIndex(std::string fpath);
  void LoadIndex(std::string fpath);
  void SearchGraph(raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
                   const int num_queries,
                   const int topk,
                   const int ef_search,
                   raft::device_matrix_view<NeighborIdxT, int64_t, raft::row_major> neighbors,
                   float* distances,
                   raft::device_vector_view<int> found_cnt);

 private:
  void GetEntryPoints(const cuda_scalar* qdata,
                      const std::vector<int>& nodes,
                      std::vector<int>& entries);
  void SearchAtLayer(const std::vector<int>& queries,
                     std::vector<std::deque<std::pair<float, int>>>& entries,
                     int level,
                     int max_m);
  void SearchHeuristicAtLayer(const std::vector<int>& queries,
                              int level,
                              int max_m,
                              bool postprocess);
  void BuildLevelGraph(int level);
  std::vector<LevelGraph> level_graphs_;
  std::vector<int> levels_;

  // int num_data_, num_dims_, batch_size_;
  int num_data_, num_dims_;
  thrust::device_vector<cuda_scalar> device_data_;
  const float* data_;
  std::vector<int> labels_;
  bool labelled_     = false;
  bool reverse_cand_ = false;

  int cores_;
  int block_cnt_, block_dim_, hyper_threads_;
  int visited_table_size_, visited_list_size_;
  int max_level_, max_m_, max_m0_;
  int enter_point_, ef_construction_;
  int dist_type_;
  bool save_remains_;
  double heuristic_coef_;
};

}  // namespace cuhnsw_v2
