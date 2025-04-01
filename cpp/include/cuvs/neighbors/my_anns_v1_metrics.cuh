#pragma once
#include <cstdint>
#include <iostream>

namespace cuvs::neighbors::my_anns_v1::detail {

struct MyAnnsV1Metrics {
  std::uint64_t counter_clk_thread;
  std::uint64_t clk_init;
  std::uint64_t clk_compute_1st_distance;
  std::uint64_t clk_topk;
  std::uint64_t counter_topk_bitonic_sort;
  std::uint64_t counter_topk_radix_sort;
  std::uint64_t clk_reset_hash;
  std::uint64_t counter_reset_hash;
  std::uint64_t clk_pickup_parents;
  std::uint64_t counter_pickup_parents;
  // 1 restore hashop = internal_topk insert hashmap
  std::uint64_t clk_restore_hash;
  std::uint64_t counter_restore_hash;
  std::uint64_t clk_insert_hashmap;
  std::uint64_t clk_load_gmem_graph;
  std::uint64_t counter_insert_hashmap;
  std::uint64_t clk_compute_distance;
  std::uint64_t clk_final;

  uint64_t clk_counter;

  uint64_t global_distance_calculation_counter1;
  uint64_t global_distance_calculation_counter2;
  uint64_t global_distance_calculation_counter3;
  uint64_t global_distance_calculation_counter4;
  uint64_t global_distance_calculation_counter3_4_counter;

  uint32_t param_smem_size;
  uint32_t param_min_iterations;
  uint32_t param_max_iterations;
  uint32_t param_hash_bitlen;
  uint32_t param_small_hash_bitlen;
  uint32_t param_small_hash_reset_interval;

  __host__ __device__ void reset()
  {
    counter_clk_thread = 0;
    clk_init           = 0;
    clk_compute_1st_distance = 0;
    clk_topk                 = 0;
    counter_topk_bitonic_sort = 0;
    counter_topk_radix_sort   = 0;
    clk_reset_hash           = 0;
    counter_reset_hash        = 0;
    clk_pickup_parents        = 0;
    counter_pickup_parents    = 0;
    clk_restore_hash          = 0;
    counter_restore_hash      = 0;
    clk_insert_hashmap        = 0;
    clk_load_gmem_graph       = 0;
    counter_insert_hashmap    = 0;
    clk_compute_distance      = 0;
    clk_final                 = 0;

    clk_counter              = 0;

    global_distance_calculation_counter1           = 0;
    global_distance_calculation_counter2           = 0;
    global_distance_calculation_counter3           = 0;
    global_distance_calculation_counter4           = 0;
    global_distance_calculation_counter3_4_counter = 0;

    param_smem_size = 0;
    param_min_iterations = 0;
    param_max_iterations = 0;
    param_hash_bitlen = 0;
    param_small_hash_bitlen = 0;
    param_small_hash_reset_interval = 0;
  }
};

enum class MyAnnsV1KernelType : int {
  kUnknown   = 0,
  kSingleCta = 1,
  kMultiCta  = 2,
};

struct MyAnnsV1MetricsAccumulator {
 private:
  MyAnnsV1MetricsAccumulator() { metrics.reset(); }

 public:
  MyAnnsV1MetricsAccumulator(const MyAnnsV1MetricsAccumulator&)            = delete;
  MyAnnsV1MetricsAccumulator(MyAnnsV1MetricsAccumulator&&)                 = delete;
  MyAnnsV1MetricsAccumulator& operator=(const MyAnnsV1MetricsAccumulator&) = delete;
  MyAnnsV1MetricsAccumulator& operator=(MyAnnsV1MetricsAccumulator&&)      = delete;

  static MyAnnsV1MetricsAccumulator& get_instance()
  {
    static MyAnnsV1MetricsAccumulator instance;
    return instance;
  }

  MyAnnsV1Metrics metrics{};
  uint64_t num_executed_iterations{};
  uint64_t num_queries{};
  MyAnnsV1KernelType kernel_type{MyAnnsV1KernelType::kUnknown};

  void accumulate(const MyAnnsV1Metrics& m,
                  uint32_t* const num_executed_iterations,
                  const uint32_t num_queries,
                  MyAnnsV1KernelType kernel_type)
  {
    metrics.counter_clk_thread += m.counter_clk_thread;
    metrics.clk_init += m.clk_init;
    metrics.clk_compute_1st_distance += m.clk_compute_1st_distance;
    metrics.clk_topk += m.clk_topk;
    metrics.counter_topk_bitonic_sort += m.counter_topk_bitonic_sort;
    metrics.counter_topk_radix_sort += m.counter_topk_radix_sort;
    metrics.clk_reset_hash += m.clk_reset_hash;
    metrics.counter_reset_hash += m.counter_reset_hash;
    metrics.clk_pickup_parents += m.clk_pickup_parents;
    metrics.counter_pickup_parents += m.counter_pickup_parents;
    metrics.clk_restore_hash += m.clk_restore_hash;
    metrics.counter_restore_hash += m.counter_restore_hash;
    metrics.clk_insert_hashmap += m.clk_insert_hashmap;
    metrics.clk_load_gmem_graph += m.clk_load_gmem_graph;
    metrics.counter_insert_hashmap += m.counter_insert_hashmap;
    metrics.clk_compute_distance += m.clk_compute_distance;
    metrics.clk_final += m.clk_final;

    metrics.clk_counter += m.clk_counter;

    metrics.global_distance_calculation_counter1 += m.global_distance_calculation_counter1;
    metrics.global_distance_calculation_counter2 += m.global_distance_calculation_counter2;
    metrics.global_distance_calculation_counter3 += m.global_distance_calculation_counter3;
    metrics.global_distance_calculation_counter4 += m.global_distance_calculation_counter4;
    metrics.global_distance_calculation_counter3_4_counter +=
      m.global_distance_calculation_counter3_4_counter;

    for (uint32_t i = 0; i < num_queries; ++i) {
      this->num_executed_iterations += num_executed_iterations[i];
    }
    this->num_queries += num_queries;
    this->kernel_type = kernel_type;
  }

  void reset()
  {
    metrics.reset();
    num_executed_iterations = 0;
    num_queries             = 0;
    kernel_type             = MyAnnsV1KernelType::kUnknown;
  }

  void print_metrics() const {
    std::cout << "counter_clk_thread: " << metrics.counter_clk_thread << std::endl;
    std::cout << "clk_init: " << metrics.clk_init << std::endl;
    std::cout << "clk_compute_1st_distance: " << metrics.clk_compute_1st_distance << std::endl;
    std::cout << "clk_topk: " << metrics.clk_topk << std::endl;
    std::cout << "counter_topk_bitonic_sort: " << metrics.counter_topk_bitonic_sort << std::endl;
    std::cout << "counter_topk_radix_sort: " << metrics.counter_topk_radix_sort << std::endl;
    std::cout << "clk_reset_hash: " << metrics.clk_reset_hash << std::endl;
    std::cout << "counter_reset_hash: " << metrics.counter_reset_hash << std::endl;
    std::cout << "clk_pickup_parents: " << metrics.clk_pickup_parents << std::endl;
    std::cout << "counter_pickup_parents: " << metrics.counter_pickup_parents << std::endl;
    std::cout << "clk_restore_hash: " << metrics.clk_restore_hash << std::endl;
    std::cout << "counter_restore_hash: " << metrics.counter_restore_hash << std::endl;
    std::cout << "clk_insert_hashmap: " << metrics.clk_insert_hashmap << std::endl;
    std::cout << "clk_load_gmem_graph: " << metrics.clk_load_gmem_graph << std::endl;
    std::cout << "counter_insert_hashmap: " << metrics.counter_insert_hashmap << std::endl;
    std::cout << "clk_compute_distance: " << metrics.clk_compute_distance << std::endl;
    std::cout << "clk_final: " << metrics.clk_final << std::endl;

    std::cout << "clk_counter: " << metrics.clk_counter << std::endl;

    std::cout << "global_distance_calculation_counter1: " << metrics.global_distance_calculation_counter1 << std::endl;
    std::cout << "global_distance_calculation_counter2: " << metrics.global_distance_calculation_counter2 << std::endl;
    std::cout << "global_distance_calculation_counter3: " << metrics.global_distance_calculation_counter3 << std::endl;
    std::cout << "global_distance_calculation_counter4: " << metrics.global_distance_calculation_counter4 << std::endl;
    std::cout << "global_distance_calculation_counter3_4_counter: " << metrics.global_distance_calculation_counter3_4_counter << std::endl;

    std::cout << "param_smem_size: " << metrics.param_smem_size << std::endl;
    std::cout << "param_min_iterations: " << metrics.param_min_iterations << std::endl;
    std::cout << "param_max_iterations: " << metrics.param_max_iterations << std::endl;
    std::cout << "param_hash_bitlen: " << metrics.param_hash_bitlen << std::endl;
    std::cout << "param_small_hash_bitlen: " << metrics.param_small_hash_bitlen << std::endl;
    std::cout << "param_small_hash_reset_interval: " << metrics.param_small_hash_reset_interval << std::endl;

    std::cout << "num_executed_iterations: " << num_executed_iterations << std::endl;
    std::cout << "num_queries: " << num_queries << std::endl;
    std::cout << "kernel_type: " << static_cast<int>(kernel_type) << std::endl;
  }
};

}  // namespace cuvs::neighbors::cagra::detail
