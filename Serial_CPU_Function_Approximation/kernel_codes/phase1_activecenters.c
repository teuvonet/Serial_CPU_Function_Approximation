
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
void phase1_active_centers(
                        int TOTAL_NUM_NEURONS, // [3,4,5] = 50
                        int NUM_NETS, //[3,4,5] = 3 , [4,5] = 2
                        int NUM_FEATURES, // dataset features
                        int CHUNK_SIZE, // datapoints to collect in each straming operation
                        int NUM_FS, //Number of feature spaces
                        int total_num_features_all_featurespaces, //Cumulative sum of total features in all feature spaces
                        int *list_num_features_featurespaces, //List of features in each feature spaces
                        int *cumulative_no_features_featurespace, //Cumulative sum of feature til particulat feature space
                        int *neurons_per_nets, // number of neurons per each net .i.e [9,16,25]
                        float *map, // the centers of all neurons across all nets
                        int *active_centers, //array to hold active centers count
                        int *min_dist_pos, //Minimum distance array position
                        float *min_dist_array,//Minimum distance array
                        float *Input_Data, // input data
                        float *distance_map, // the distance of each neuron from the data points
                        int *net_sizes,
                        int thread_id,
                        int feature_space_blockID, // [3,4,5]
                        int learning_rate_Index
                    )
{
    // Finding the total number of threads in the program
    //const int block_skip = blockDim.z*blockDim.y*blockDim.x;
    // const int grid_skip = gridDim.z*gridDim.y*gridDim.x;

    // Finding thread index using block and thread index
    //const int thread_id = blockIdx.x*block_skip + blockIdx.y*block_skip*gridDim.x + blockIdx.z*block_skip*gridDim.x*gridDim.y + threadIdx.x;
    //const int thread_id = get_global_id(0) + get_global_id(1)*get_global_size(0) ;
    //Initial declaration
    //int feature_space_blockID = get_global_id(0)/TOTAL_NUM_NEURONS; //Feature Space Number if Grid X direction
    int no_of_features_featurespace = list_num_features_featurespaces[feature_space_blockID]; //Get the Number of features that in each feature space
    //int learning_rate_Index = get_global_id(1); //Learning Rate Index for every thread

    //Find the corresponding Map Array Index
    //Find the corresponding Map Array Index
    int map_start_index = learning_rate_Index * TOTAL_NUM_NEURONS * total_num_features_all_featurespaces; // skip one entire learning rate neurons
    map_start_index += TOTAL_NUM_NEURONS * cumulative_no_features_featurespace[feature_space_blockID]; // skip all blocks in same learning space before it
    map_start_index += no_of_features_featurespace * (thread_id % TOTAL_NUM_NEURONS); // skip all neurons before it in the same learning rate and block

    // Loop for each data point
    int datapoint = 0;
    for(; datapoint<CHUNK_SIZE; datapoint++)
    {
        //______________________________________________________________________________________________________________________________________________________
        // Step 1: Find Cumulative distance between neuron center and input data for each feature
        //------------------------------------------------------------------------------------------------------------------------------------------------------
        float sum=0;

        for (int feature = 0; feature < no_of_features_featurespace; feature++){
             //printf("Item ThreadId :%2d |  Feature_Space :%2d  |   Learning_rate :%2d  |  Input data :%2f |  Map-data :%2f\n", thread_id, feature_space_blockID, learning_rate_Index, Input_Data[datapoint * CHUNK_SIZE + feature], map[map_start_index + feature]);
            sum += fabs(Input_Data[datapoint * NUM_FEATURES + feature] - map[map_start_index + feature]);
        }

        //printf("%f %d %d %d\n", sum, thread_id, datapoint, learning_rate_Index);
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

            //Loop through each net

            for(int net = 0; net < NUM_NETS; net++)
            {
                min_dist_array[min_pos_Index + net] = INT_MAX;
                    //Loop through every neuron in that net
                for(int neuron = 0; neuron < neurons_per_nets[net]; neuron++)
                {
                    //printf("thread_id %d neuron %d net %d min %d\n", thread_id, neuron, net, min_pos_Index);
                    if(distance_map[distance_map_Index + neuron] < min_dist_array[min_pos_Index + net])
                    {
                        //Capture min dist array neuron and it's Index
                        min_dist_array[min_pos_Index + net] = distance_map[distance_map_Index + neuron];
                        min_dist_pos[min_pos_Index+net] = neuron;
                    }
                }
                //printf("complete active centers");
                // if (blockIdx.x == 0 && blockIdx.y == 0)
                    // printf("\n%2d",distance_map_Index+min_dist_pos[min_pos_Index+net]);
                active_centers[distance_map_Index+min_dist_pos[min_pos_Index+net]] += 1;
                //printf("\n active_centers %d ",active_centers[distance_map_Index+min_dist_pos[min_pos_Index+net]] );
                //Adjust distance map after every net
                distance_map_Index += neurons_per_nets[net];

            }
        }
        //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

}




