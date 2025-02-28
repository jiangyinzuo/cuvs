#include <hnswlib/hnswlib.h>
#include <iostream>
#include <map>
#include <memory>

void graph_node_count(hnswlib::HierarchicalNSW<float, true>& hnsw_alg)
{
  std::map<int, int> level_node_count;
  for (int i = 0; i < hnsw_alg.element_levels_.size(); ++i) {
    level_node_count[hnsw_alg.element_levels_[i]]++;
  }
  for (int level = 0; level < 10; ++level) {
    int total_count = 0;
    for (auto [l, count] : level_node_count) {
      if (l >= level) { total_count += count; }
    }
    std::cout << total_count << ",";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <index_file>" << std::endl;
    return 1;
  }
  int dim                = 100;
  std::string index_file = argv[1];
  auto space             = std::make_shared<hnswlib::L2Space>(dim);
  hnswlib::HierarchicalNSW<float, true> hnsw_alg(space.get());
  hnsw_alg.loadIndex(index_file, space.get());
  hnsw_alg.two_hop_analysis_each_layer();
  // graph_node_count(hnsw_alg);
  return 0;
}
