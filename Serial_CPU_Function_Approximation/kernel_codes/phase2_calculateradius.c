/*
This code is licensed and documented by Teuvonet Technologies.
Any use of this code, proprietary or personal, needs approval from the company.

*/
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
void phase2_calculateradius(
                        int TOTAL_NUM_NEURONS, //Total Number of Neurons 50 in case of [3,4,5] nets
                        int NUM_NETS,//[3,4,5] = 3 , [4,5] = 2
                        int Input_Data_Size, //Number of Data Points
                        int NUM_FEATURES, //Number of Features
                        int NUM_FS, //Number of Feature Spaces
                        int total_num_features_all_featurespaces, //Cumulative Sum of total number of features in all feature spaces
                        int *neurons_per_nets,// number of neurons per each net .i.e [9,16,25]
                        int *list_num_features_featurespaces, //list of features in features paces - [1,2,3]
                        int *cumulative_no_features_featurespace, //Cumulative sum of feature space - Easy for Indexing the base map - [1,3,6,10]
                        float *map, //Base Map of the Neurons
                        int *min_dist_pos, //Minimum distance array position
                        float *min_dist_array,//Minimum distance array
                        float *distance_map, //Distance Map of each neuro. calculated form each datapoint
                        float *radius_map, //Radius around teh Neuron
                        int *active_centers, //array to hold active centers count
                        float *Input_Data, //Input Data Size
                        int thread_id,
                        int feature_space_blockID, // [3,4,5]
                        int learning_rate_Index
                    )
{


     //const int thread_id = get_global_id(0) + get_global_id(1)*get_global_size(0);
    // Loop for each data point
    int datapoint = 0;
    for(; datapoint<Input_Data_Size; datapoint++)
    {
        //Initial declaration
        //int feature_space_blockID = get_global_id(0)/TOTAL_NUM_NEURONS; //Feature Space Number if Grid X direction
        int no_of_features_featurespace = list_num_features_featurespaces[feature_space_blockID]; //Get the Number of features that in each feature space
        //int learning_rate_Index = get_global_id(1); //Learning Rate Index for every thread
        int map_start_index = learning_rate_Index * TOTAL_NUM_NEURONS * total_num_features_all_featurespaces; // skip one entire learning rate neurons
        map_start_index += TOTAL_NUM_NEURONS * cumulative_no_features_featurespace[feature_space_blockID]; // skip all blocks in same learning space before it
        map_start_index += no_of_features_featurespace * (thread_id % TOTAL_NUM_NEURONS); // skip all neurons before it in the same learning rate and block
        //printf("Item %4d %2d %2d\n", thread_id, learning_rate_Index, datapoint);


        //______________________________________________________________________________________________________________________________________________________
        // Step 1: Find Cumulative distance between neuron center and input data for each feature
        //------------------------------------------------------------------------------------------------------------------------------------------------------


        float sum=0;

        //Find the corresponding Map Array Index


        for (int feature = 0; feature < no_of_features_featurespace; feature++){
            // printf("Item ThreadId :%2d |  Feature_Space :%2d  |   Learning_rate :%2d  |  Input data :%2f |  Map-data :%2f\n", thread_id, feature_space_blockID, learning_rate_Index, chunk_inp[chunck_index * CHUNK_SIZE + feature], map[map_start_index + feature]);
            sum += fabs(Input_Data[datapoint * NUM_FEATURES + feature] - map[map_start_index + feature]);
            //printf("\n map %f", map[map_start_index + feature]);
        }

        //printf("%f %d %d %d\n", sum, thread_id, chunck_index, learning_rate_Index);
        distance_map[thread_id] = sum;
        //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        //______________________________________________________________________________________________________________________________________________________
        //END OF STEP 1
        //----------------------------------------------------------------------------------------------------------------------------------------------------------






        //_____________________________________________________________________________________________________________________________________________________
        //STEP 2: Calculating minium distance Neuron and it's index in that Net
        //----------------------------------------------------------------------------------------------------------------------------------------------------

        if((NUM_FS*TOTAL_NUM_NEURONS)%TOTAL_NUM_NEURONS == 0)
        {
            //Distance Map index adjusting
            int distance_map_Index = thread_id;
            int min_pos_Index = (thread_id/TOTAL_NUM_NEURONS)*NUM_NETS;
            float min_dist = 0;

            //Loop through each net
            int net = 0;
            for(; net < NUM_NETS; net++)
            {
                min_dist_array[min_pos_Index + net] = 2147483647;
                //Loop through every neuron in that net
                for(int neuron = 0; neuron < neurons_per_nets[net]; neuron++)
                {
                    if(distance_map[distance_map_Index + neuron] < min_dist_array[min_pos_Index + net])
                    {
                        //Capture min dist array neuron and it's Index
                        min_dist_array[min_pos_Index + net] = distance_map[distance_map_Index + neuron];
                        min_dist_pos[min_pos_Index+net] = neuron;
                        min_dist = distance_map[distance_map_Index + neuron];
                    }
                    // mp_index += no_of_features_featurespace;
                }
                // if (blockIdx.x == 1)
                    // printf("\nNeuron %3d Input %3.3f %3.3f Distance %3.3f", distance_map_Index+min_dist_pos[min_pos_Index+net], Input_Data[datapoint*NUM_FEATURES],Input_Data[datapoint*NUM_FEATURES+1], distance_map[distance_map_Index+min_dist_pos[min_pos_Index+net]]);

                active_centers[distance_map_Index+min_dist_pos[min_pos_Index+net]] += 1;
                radius_map[distance_map_Index+min_dist_pos[min_pos_Index+net]] = (radius_map[distance_map_Index+min_dist_pos[min_pos_Index+net]] > min_dist) ? radius_map[distance_map_Index+min_dist_pos[min_pos_Index+net]] : min_dist;
                //Adjust distance map after every net
                distance_map_Index += neurons_per_nets[net];
            }
        }
        //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        //______________________________________________________________________________________________________________________________________________________
        //END OF STEP 2
        //----------------------------------------------------------------------------------------------------------------------------------------------------------
    }
}
