#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/resources.hpp>

int main() {
  using namespace cuvs::neighbors;
  raft::resources handle;
  brute_force::index_params index_params;
  auto index = brute_force::build(handle, index_params, "");
  return 0;
}
