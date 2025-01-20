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

#include "../common/ann_types.hpp"
#include "../common/util.hpp"

#include "song/config.h"
#include "song/kernelgraph.h"
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <utility>
#include <vector>

namespace cuvs::bench {
template <SongMetricType metric_type, int DIM>
class song_impl;

class song : public algo<data_value_t>, public algo_gpu {
 public:
  struct build_param {
    int num_threads{32};
    int degree{31};
  };

  using search_param_base = typename algo<data_value_t>::search_param;

  struct search_param : public search_param_base {
    size_t finish_cnt;
    int pq_size;
    [[nodiscard]] virtual auto needs_dataset() const -> bool override { return true; };
  };

  song(Metric metric, int dim, const build_param& param);
  void build(const float* dataset, size_t nrow) override { impl_->build(dataset, nrow); }

  void set_search_param(const search_param_base& param) override { impl_->set_search_param(param); }
  void set_search_dataset(const data_value_t* dataset, size_t nrow) override
  {
    impl_->set_search_dataset(dataset, nrow);
  };
  void search(const float* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override
  {
    impl_->search(queries, batch_size, k, neighbors, distances);
  }
  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return dynamic_cast<algo_gpu*>(impl_.get())->get_sync_stream();
  }

  void save(const std::string& file) const override { impl_->save(file); }

  void load(const std::string& file) override { impl_->load(file); }
  std::unique_ptr<algo<data_value_t>> copy() override { return std::make_unique<song>(*this); };

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    return impl_->get_preference();
  }

 private:
  std::shared_ptr<algo<data_value_t>> impl_;
};

template <SongMetricType metric_type, int DIM>
class song_impl : public algo<data_value_t>, public algo_gpu {
 public:
  using search_param_base = typename algo<data_value_t>::search_param;

  song_impl(Metric metric, int dim, const typename song::build_param& param)
    : algo<data_value_t>(metric, dim),
      build_param_(param),
      stream_(cuvs::bench::get_stream_from_global_pool())
  {
  }

  void build(const float* dataset, size_t nrow) override;

  void set_search_param(const search_param_base& param) override;
  void search(const float* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;
  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override { return stream_; }

  void save(const std::string& file) const override;
  void load(const std::string& file) override;
  void set_search_dataset(const data_value_t* dataset, size_t nrow) override;

  std::unique_ptr<algo<data_value_t>> copy() override
  {
    auto r = std::make_unique<song_impl>(*this);
    // set the thread-local stream to the copied handle.
    r->stream_ = cuvs::bench::get_stream_from_global_pool();
    return r;
  };

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

 private:
  void add_data(const float* dataset, size_t nrow, bool add_vertex);

  using algo<data_value_t>::dim_;

  using song_instance = KernelFixedDegreeGraph<metric_type, DIM>;
  std::shared_ptr<song_instance> song_;
  std::shared_ptr<Data> data_;
  std::string file_;
  typename song::build_param build_param_;
  typename song::search_param search_param_;
  cudaStream_t stream_;
  const float* base_dataset_ = nullptr;
  size_t base_n_rows_        = 0;
};

song::song(Metric metric, int dim, const build_param& param) : algo<float>(metric, dim)
{
  if (metric == Metric::kEuclidean) {
    if (dim == 100) {
      impl_ = std::make_shared<song_impl<SongMetricType::L2, 100>>(metric, dim, param);
    } else if (dim == 128) {
      impl_ = std::make_shared<song_impl<SongMetricType::L2, 128>>(metric, dim, param);
    } else if (dim == 960) {
      impl_ = std::make_shared<song_impl<SongMetricType::L2, 960>>(metric, dim, param);
    }
  } else if (metric == Metric::kInnerProduct) {
    if (dim == 100) {
      impl_ = std::make_shared<song_impl<SongMetricType::IP, 100>>(metric, dim, param);
    } else if (dim == 128) {
      impl_ = std::make_shared<song_impl<SongMetricType::IP, 128>>(metric, dim, param);
    } else if (dim == 960) {
      impl_ = std::make_shared<song_impl<SongMetricType::IP, 960>>(metric, dim, param);
    }
  }
  if (!impl_) { throw std::runtime_error("unsupported metric or dim"); }
}

template <SongMetricType metric_type, int DIM>
void song_impl<metric_type, DIM>::add_data(const float* dataset, size_t nrow, bool add_vertex)
{
  std::vector<std::thread> threads;
  size_t step = nrow / build_param_.num_threads;
  for (int i = 0; i < build_param_.num_threads; ++i) {
    size_t start = i * step;
    size_t end   = (i == build_param_.num_threads - 1) ? nrow : (i + 1) * step;
    threads.push_back(std::thread([this, dataset, start, end, add_vertex] {
      for (size_t row = start; row < end; ++row) {
        std::vector<std::pair<int, value_t>> vec;
        vec.reserve(dim_);
        const value_t* ptr = dataset + dim_ * row;
        for (int col = 0; col < dim_; ++col) {
          vec.emplace_back(col, ptr[col]);
        }
        // load data
        data_->add(row, vec);
        // build graph
        if (add_vertex) { song_->add_vertex(row, vec); }
      }
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

template <SongMetricType metric_type, int DIM>
void song_impl<metric_type, DIM>::build(const float* dataset, size_t nrow)
{
  base_dataset_ = dataset;
  base_n_rows_  = nrow;
  data_         = std::make_shared<Data>(nrow, dim_);
  song_         = std::make_shared<song_instance>(data_.get(), build_param_.degree);

  add_data(dataset, nrow, true);
}

template <SongMetricType metric_type, int DIM>
void song_impl<metric_type, DIM>::set_search_param(const search_param_base& param)
{
  search_param_ = dynamic_cast<const typename song::search_param&>(param);
}

template <SongMetricType metric_type, int DIM>
void song_impl<metric_type, DIM>::search(const float* queries,
                                         int batch_size,
                                         int k,
                                         algo_base::index_type* neighbors,
                                         float* distances) const
{
  std::vector<std::vector<std::pair<int, value_t>>> queries_vec;
  std::vector<std::vector<idx_t>> results;
  queries_vec.reserve(batch_size);

  for (int row = 0; row < batch_size; ++row) {
    const value_t* ptr = queries + row * dim_;
    std::vector<std::pair<int, value_t>> vec_row;
    vec_row.reserve(dim_);
    for (int col = 0; col < dim_; ++col) {
      vec_row.emplace_back(col, ptr[col]);
    }
    queries_vec.push_back(vec_row);
  }
  results.reserve(batch_size);
  switch (search_param_.pq_size) {
    case 10:
      song_->template search_top_k_batch<10>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 20:
      song_->template search_top_k_batch<20>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 30:
      song_->template search_top_k_batch<30>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 40:
      song_->template search_top_k_batch<40>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 50:
      song_->template search_top_k_batch<50>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 60:
      song_->template search_top_k_batch<60>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 70:
      song_->template search_top_k_batch<70>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 80:
      song_->template search_top_k_batch<80>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 90:
      song_->template search_top_k_batch<90>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 100:
      song_->template search_top_k_batch<100>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 150:
      song_->template search_top_k_batch<150>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 200:
      song_->template search_top_k_batch<200>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 400:
      song_->template search_top_k_batch<400>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 800:
      song_->template search_top_k_batch<800>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 1600:
      song_->template search_top_k_batch<1600>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 3200:
      song_->template search_top_k_batch<3200>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    default: throw std::runtime_error("unsupported pq_size");
  }
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < k; ++j) {
      // if (i < 10) { std::cout << results[i][j] << " "; }
      neighbors[i * k + j] = results[i][j];
    }
    // if (i < 10) { std::cout << std::endl; }
  }
}

template <SongMetricType metric_type, int DIM>
void song_impl<metric_type, DIM>::save(const std::string& file) const
{
  song_->dump(file);
}

template <SongMetricType metric_type, int DIM>
void song_impl<metric_type, DIM>::load(const std::string& file)
{
  file_ = file;
}

template <SongMetricType metric_type, int DIM>
void song_impl<metric_type, DIM>::set_search_dataset(const data_value_t* dataset, size_t nrow)
{
  if (file_.empty()) { throw std::runtime_error("file is not set"); }
  data_ = std::make_shared<Data>(nrow, dim_);
  add_data(dataset, nrow, false);
  song_ = std::make_shared<song_instance>(data_.get(), build_param_.degree);
  song_->load(file_);
};
}  // namespace cuvs::bench
