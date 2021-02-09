#include<stdio.h>
#include<math.h>
#include<stdlib.h>
void phase1_distance_computation(
    float* base_map,
    int NUM_FEATURES,
    float* overall_distances,
    int TOTAL_NUM_NEURONS,
    int NUM_FS,
    int NUM_NETS,
    int* neurons_per_nets,
    int* list_num_features_featurespaces,
    int* active_centers,
    int tid,
    int leraning_rate_index)
{
    int partition_number = 0; // Calculate which partition this feature is in.
    int part_offset = 0;


    for(int i = 0; i<NUM_FS; ++i)
    {
        if(tid < part_offset + list_num_features_featurespaces[i])
            break;
        partition_number ++;
        part_offset += list_num_features_featurespaces[i];
    }
    int base_map_position = leraning_rate_index * NUM_FEATURES * TOTAL_NUM_NEURONS; // (50 * 3 ) + (50 * 4) = (50 * 7)
    int feature_offset = 0;

    for(int i =0; i < partition_number; i++)
    {
        base_map_position += list_num_features_featurespaces[i]*TOTAL_NUM_NEURONS;
        feature_offset += list_num_features_featurespaces[i];
    }
    int neuronA = base_map_position;
    float avg_dist = 0;

    for(int i = 0; i<NUM_NETS; ++i)
    {
        avg_dist = 0;

        for(int j = 0; j< neurons_per_nets[i]; ++j)
        {

            for(int k = j+1; k < neurons_per_nets[i]; ++k)
            {
                if((active_centers[(leraning_rate_index) * TOTAL_NUM_NEURONS * NUM_FS + TOTAL_NUM_NEURONS * partition_number + j] >= 2) && (active_centers[(leraning_rate_index) * TOTAL_NUM_NEURONS * NUM_FS + TOTAL_NUM_NEURONS * partition_number + k] >= 2))
                {
                    avg_dist += fabs(base_map[neuronA + j * list_num_features_featurespaces[partition_number] + (NUM_FEATURES-feature_offset)] - base_map[neuronA + k * list_num_features_featurespaces[partition_number] + (NUM_FEATURES-feature_offset)]);
                }
            }
        }
        overall_distances[(leraning_rate_index) * NUM_FEATURES + tid] += (avg_dist/((neurons_per_nets[i]*(neurons_per_nets[i]-1))/2));
    }
}
