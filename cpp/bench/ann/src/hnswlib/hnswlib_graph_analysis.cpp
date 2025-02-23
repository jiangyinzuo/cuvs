#include <hnswlib/hnswlib.h>
#include <memory>

int main()
{
  int dim = 100;
  std::string index_file =
    "/usr3/jiangyinzuo_data/cuvs-bench-dataset/glove-100-inner/index/hnswlib.M24.efConstruction512";
  auto space = std::make_shared<hnswlib::L2Space>(dim);
  hnswlib::HierarchicalNSW<float, true> hnsw_alg(space.get());
  hnsw_alg.loadIndex(index_file, space.get());
  hnsw_alg.two_hop_analysis_each_layer();
  return 0;
}
