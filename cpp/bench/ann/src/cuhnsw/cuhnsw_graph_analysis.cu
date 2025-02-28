#include "cuhnsw.hpp"
#include <iostream>
#include <spdlog/common.h>
#include "spdlog/spdlog.h"
int main(int argc, char** argv)
{
  spdlog::set_level(spdlog::level::err);
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <index_path>" << std::endl;
    return 1;
  }
  cuhnsw::CuHNSW cuhnsw;
  cuhnsw.LoadIndex(argv[1]);

  for (int i = 0; i < cuhnsw.level_graphs_.size(); ++i) {
    std::cout << cuhnsw.level_graphs_[i].NumNodes() << ",";
  }
  for (auto i = cuhnsw.level_graphs_.size(); i < 10; ++i) {
    std::cout << "0,";
  }
  std::cout << std::endl;
  return 0;
}
