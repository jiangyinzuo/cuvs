#pragma once

#include "data.h"
#include "metric_type.h"
#include <string>

class GraphWrapper;

namespace ganns {
class GANNS {

public:
  GANNS() : graph_{nullptr} {}
  ~GANNS();
  void Load(std::string graph_path);
  void Dump(std::string graph_name);
  void Establishment(int num_of_initial_neighbors, int num_of_candidates);
  void SearchTopKonDevice(float *queries, int num_of_topk, int *&results,
                          int num_of_query_points, int num_of_candidates);
  void DisplayGraphParameters(int num_of_candidates);
  void DisplaySearchParameters(int num_of_topk, int num_of_candidates);

  template <ganns::MetricType metric_type, int DIM>
  void AddGraph(std::string graph_type, Data *points);
  
  void FreeResults(int* results);

private:
  GraphWrapper *graph_;
};

extern template void
GANNS::AddGraph<ganns::MetricType::L2, 128>(std::string graph_type,
                                            Data *points);
extern template void
GANNS::AddGraph<ganns::MetricType::L2, 100>(std::string graph_type,
                                            Data *points);
extern template void
GANNS::AddGraph<ganns::MetricType::L2, 96>(std::string graph_type,
                                            Data *points);
extern template void
GANNS::AddGraph<ganns::MetricType::IP, 100>(std::string graph_type,
                                            Data *points);
extern template void
GANNS::AddGraph<ganns::MetricType::IP, 96>(std::string graph_type,
                                            Data *points);
} // namespace ganns
