#pragma once
#include "structure_on_device.h"
#include "metric_type.h"

template <ganns::MetricType metric_type, int DIM>
__global__
void DistanceMatrixComputation(float* d_data, int total_num_of_points, int num_of_points_one_batch, KernelPair<float, int>* distance_matrix){
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;

    for (int i = 0; i < num_of_points_one_batch; i++) {
        int crt_point_id = b_id * num_of_points_one_batch + i;

        if (crt_point_id >= total_num_of_points) {
            break;
        }

        KernelPair<float, int>* crt_distance = distance_matrix + crt_point_id * num_of_points_one_batch;

        // static_assert(DIM % 32 == 0);
        static_assert(DIM <= 960);
        float q[DIM / 32];
#pragma unroll
        for (int k = 0; k < DIM / 32; k++) {
            q[k] = 0;
            if (t_id + k * 32 < DIM) {
                q[k] = d_data[crt_point_id * DIM + t_id + k * 32];
            }
        }

        for (int j = i + 1; j < num_of_points_one_batch; j++) {

            int target_point_id = b_id * num_of_points_one_batch + j;

            if(target_point_id >= total_num_of_points){
                break;
            }

            float p[DIM / 32];
            float delta[DIM / 32];
            float p_l2[DIM / 32];
            float q_l2[DIM / 32];
            float p_l2_sum = 0;
            float q_l2_sum = 0;
#pragma unroll
            for (int k = 0; k < DIM / 32; k++) {
                p[k] = 0;
                if (t_id + k * 32 < DIM) {
                    p[k] = d_data[target_point_id * DIM + t_id + k * 32];
                }
            }
            if constexpr (metric_type == ganns::MetricType::L2) {
#pragma unroll
                for (int k = 0; k < DIM / 32; k++) {
                    delta[k] = (p[k] - q[k]) * (p[k] - q[k]);
                }
            } else if constexpr (metric_type == ganns::MetricType::IP) {
#pragma unroll
                for (int k = 0; k < DIM / 32; k++) {
                    delta[k] = p[k] * q[k];
                }
            } else if constexpr (metric_type == ganns::MetricType::COS) {
#pragma unroll
                for (int k = 0; k < DIM / 32; k++) {
                    delta[k] = p[k] * q[k];
                    p_l2[k] = p[k] * p[k];
                    q_l2[k] = q[k] * q[k];
                }
            }
            float dist = 0;
            if constexpr (metric_type == ganns::MetricType::L2) {
#pragma unroll
                for (int k = 0; k < DIM / 32 ; k++) {
                    dist += delta[k];
                }
            } else if constexpr (metric_type == ganns::MetricType::IP) {
#pragma unroll
                for (int  k = 0; k < DIM / 32; k++) {
                    dist += delta[k];
                }
            } else if constexpr (metric_type == ganns::MetricType::COS) {
#pragma unroll
                for (int k = 0; k < DIM / 32; k++) {
                    dist += delta[k];
                    p_l2_sum += p_l2[k];
                    q_l2_sum += q_l2[k];
                }
            }

            if constexpr (metric_type == ganns::MetricType::L2) {
                dist += __shfl_down_sync(FULL_MASK, dist, 16);
                dist += __shfl_down_sync(FULL_MASK, dist, 8);
                dist += __shfl_down_sync(FULL_MASK, dist, 4);
                dist += __shfl_down_sync(FULL_MASK, dist, 2);
                dist += __shfl_down_sync(FULL_MASK, dist, 1);
            } else if constexpr (metric_type == ganns::MetricType::IP) {
                dist += __shfl_down_sync(FULL_MASK, dist, 16);
                dist += __shfl_down_sync(FULL_MASK, dist, 8);
                dist += __shfl_down_sync(FULL_MASK, dist, 4);
                dist += __shfl_down_sync(FULL_MASK, dist, 2);
                dist += __shfl_down_sync(FULL_MASK, dist, 1);
            } else if constexpr (metric_type == ganns::MetricType::COS) {
                dist += __shfl_down_sync(FULL_MASK, dist, 16);
                dist += __shfl_down_sync(FULL_MASK, dist, 8);
                dist += __shfl_down_sync(FULL_MASK, dist, 4);
                dist += __shfl_down_sync(FULL_MASK, dist, 2);
                dist += __shfl_down_sync(FULL_MASK, dist, 1);

                p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 16);
                p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 8);
                p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 4);
                p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 2);
                p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 1);

                q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 16);
                q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 8);
                q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 4);
                q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 2);
                q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 1);
            }

            if constexpr (metric_type == ganns::MetricType::L2) {
                dist = sqrt(dist);
            } else if constexpr (metric_type == ganns::MetricType::IP) {
                dist = -dist;
            } else if constexpr (metric_type == ganns::MetricType::COS) {
                p_l2 = sqrt(p_l2);
                q_l2 = sqrt(q_l2);
                dist = dist / (p_l2 * q_l2);
                if (t_id == 0) {
                    if(!(dist == dist)){
                        dist = 2;
                    } else {
                        dist = 1 - dist;
                    }
                }
            }

            if(t_id == 0){
                crt_distance[j].first = dist;
                crt_distance[j].second = target_point_id;

                (distance_matrix + (b_id * num_of_points_one_batch + j) * num_of_points_one_batch)[i].first = dist;
                (distance_matrix + (b_id * num_of_points_one_batch + j) * num_of_points_one_batch)[i].second = crt_point_id;
            }

        }

        if(t_id == 0){
            crt_distance[i].first = Max;
            crt_distance[i].second = crt_point_id;
        }

    }

}
