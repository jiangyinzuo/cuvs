#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <iostream>

__global__ void init_kernel(float *data) { data[threadIdx.x] = threadIdx.x; }

template <typename LayoutPolicy>
static void
print_dev_matrix(raft::device_resources &dev_resources,
                 raft::device_matrix_view<float, int64_t, LayoutPolicy> dev_matrix) {
  std::cout << "size: " << dev_matrix.size() << ", extent: " << dev_matrix.extent(0) << "x" << dev_matrix.extent(1) << std::endl;
  std::cout << "stride: " << dev_matrix.stride(0) << "x" << dev_matrix.stride(1) << std::endl;
  auto host_matrix = raft::make_host_matrix<float>(dev_resources, dev_matrix.extent(0), dev_matrix.stride(0));
  auto stream = raft::resource::get_cuda_stream(dev_resources);
  raft::copy(host_matrix.data_handle(), dev_matrix.data_handle(), host_matrix.size(), stream);
  for (int i = 0; i < dev_matrix.extent(0); i++) {
    for (int j = 0; j < dev_matrix.extent(1); j++) {
      printf("%.1f ", host_matrix(i, j));
    }
    printf("\n");
  }
}

int main() {
  raft::device_resources dev_resources;
  auto dev_matrix = raft::make_device_matrix<float>(dev_resources, 10, 7);
  init_kernel<<<1, 70, 0, dev_resources.get_stream()>>>(dev_matrix.data_handle());

  print_dev_matrix<decltype(dev_matrix)::layout_type>(dev_resources, dev_matrix.view());

  auto strided_dev_matrix = raft::make_device_strided_matrix_view<float, int64_t>(
    dev_matrix.data_handle(), 10, 4, 7);
  print_dev_matrix(dev_resources, strided_dev_matrix);
  return 0;
}
