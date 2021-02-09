/*
This code is licensed and documented by Teuvonet Technologies.
Any use of this code, proprietary or personal, needs approval from the company.
*/


int neighbourhood(
	int winner_index,
	int map_side_size,
    int current_id,
    int logic)
{
	int a_x, a_y, b_x, b_y;
	a_x = current_id % map_side_size;
	a_y = current_id / map_side_size;
	b_x = winner_index % map_side_size;
    b_y = winner_index / map_side_size;
    if(logic == 0){
        return (abs(a_x-b_x) + abs(a_y-b_y));
    }
    else{
        return max(abs(a_x-b_x) , abs(a_y-b_y));
    }
}

void loop_print(float *a, int n){
    int i;
    // printf("\n%d %d",a[0],a[1]);
    for(i=0; i<n; i++){
        printf(" %3.3f ",a[i]);
    }
}

__kernel void phase1(
                        int TOTAL_NUM_NEURONS, // [3,4,5] = 50
                        int NUM_NETS, //[3,4,5] = 3 , [4,5] = 2
                        int CHUNK_SIZE, // datapoints to collect in each streaming operation
                        int NUM_FS, //Number of feature spaces
                        int NUM_FEATURES, //Number of features in given dataset
                        int total_num_features_all_featurespaces, //Cumulative sum of total features in all feature spaces
                        int no_Passes_Phase1, //Number of passes through the data
                        __global int *list_num_featurespaces_start,
                        __global int *list_num_featurespaces_features, //List of features in each feature spaces
                        __global int *cumulative_no_features_featurespace, //Cumulative sum of feature til particulat feature space
                        __global int *neurons_per_nets,// number of neurons per each net .i.e [9,16,25]
                        __global float *map, // the centers of all neurons across all nets
                        __global int *min_dist_pos, //Minimum distance array position
                        __global float *min_dist_array,//Minimum distance array
                        __global float *chunk_inp, // input data
                        __global float *distance_map, // the distance of each neuron from the data points
                        __global float *neigh_rate,
                        __global float *learning_rate_list,
                        __global float *cumulative_weight_change_per_neuron,
                        __global int *net_sizes // [3,4,5]
                    )
{

    const int thread_id = get_global_id(0) + get_global_id(1)*get_global_size(0) ;
    int feature_space_blockID = get_global_id(0)/TOTAL_NUM_NEURONS; //Feature Space Number if Grid X direction
    int no_of_features_featurespace_start = list_num_featurespaces_start[feature_space_blockID];
    int no_of_features_featurespace_end = list_num_featurespaces_features[feature_space_blockID]; //Get the Number of features that in each feature space
    int learning_rate_Index = get_global_id(1); //Learning Rate Index for every thread

    //Find the corresponding Map Array Index
    int map_start_index = learning_rate_Index * TOTAL_NUM_NEURONS * total_num_features_all_featurespaces; // skip one entire learning rate neurons
    map_start_index += TOTAL_NUM_NEURONS * cumulative_no_features_featurespace[feature_space_blockID]; // skip all blocks in same learning space before it
    map_start_index += no_of_features_featurespace_end * (thread_id % TOTAL_NUM_NEURONS); // skip all neurons before it in the same learning rate and block

    //Finding Thread details - Which net it belongs to?
    int which_net_index = 0;
    int which_net = 0;
    int net_end_index = learning_rate_Index * NUM_FS *  TOTAL_NUM_NEURONS + feature_space_blockID * TOTAL_NUM_NEURONS;
    int net_start_index = learning_rate_Index * NUM_FS * TOTAL_NUM_NEURONS + feature_space_blockID * TOTAL_NUM_NEURONS;
    int thread_position_in_map = 0;
    for(int net = 0; net < NUM_NETS; net++)
    {
        net_end_index += neurons_per_nets[net];
        if(thread_id >= net_start_index && thread_id < net_end_index)
        {
            which_net_index = net;
            which_net = net_sizes[net];
            thread_position_in_map = thread_id - net_start_index;
        }
        net_start_index = net_end_index;
    }

    // if (thread_id == 0) loop_print(chunk_inp, NUM_FEATURES*CHUNK_SIZE);

    for(int pass = 0; pass < no_Passes_Phase1; pass++){

        // Loop for each data point
        int chunck_index = 0;
        for(; chunck_index<CHUNK_SIZE; chunck_index++)
        {
            //______________________________________________________________________________________________________________________________________________________
            // Step 1: Find Cumulative distance between neuron center and input data for each feature
            //------------------------------------------------------------------------------------------------------------------------------------------------------

            float sum=0;

            for (int feature = 0; feature < no_of_features_featurespace_end-no_of_features_featurespace_start; feature++){
                sum += fabs(chunk_inp[chunck_index * NUM_FEATURES + feature + no_of_features_featurespace_start] - map[map_start_index + feature]);
                // if(thread_id==0)
                    // printf("\n %2d %2d %3.3f %3.3f", chunck_index, feature, chunk_inp[chunck_index * NUM_FEATURES + feature], map[map_start_index + feature]);
            }

            distance_map[thread_id] = sum;

            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);



            if(get_global_id(0)%TOTAL_NUM_NEURONS == 0)
            {
                //Distance Map index adjusting
                int distance_map_Index = thread_id;
                int min_pos_Index = (thread_id/TOTAL_NUM_NEURONS)*NUM_NETS;

                //Loop through each net
                int net = 0;
                for(; net < NUM_NETS; net++)
                {
                    min_dist_array[min_pos_Index + net] = INT_MAX;
                    //Loop through every neuron in that net
                    for(int neuron = 0; neuron < neurons_per_nets[net]; neuron++)
                    {
                        // if(thread_id == 0)
                            // printf("Chunck %d neuron %d distance %3.3f Input %3.3f %3.3f %3.3f Map %3.3f %3.3f %3.3f minDist %3.3f\n", chunck_index, neuron, distance_map[distance_map_Index + neuron], chunk_inp[chunck_index*NUM_FEATURES+0], chunk_inp[chunck_index*NUM_FEATURES+1], chunk_inp[chunck_index*NUM_FEATURES+2], map[distance_map_Index + neuron + 0], map[distance_map_Index + neuron + 1], map[distance_map_Index + neuron + 2], min_dist_array[min_pos_Index + net]);
                        if(distance_map[distance_map_Index + neuron] < min_dist_array[min_pos_Index + net])
                        {
                            //Capture min dist array neuron and it's Index
                            min_dist_array[min_pos_Index + net] = distance_map[distance_map_Index + neuron];
                            min_dist_pos[min_pos_Index+net] = neuron;
                        }
                    }
                    //Adjust distance map after every net
                    distance_map_Index += neurons_per_nets[net];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


            //Min Dist Pos Array index for each thread to see its winner in that net
            int min_dist_post_net_index = learning_rate_Index * NUM_FS * NUM_NETS + feature_space_blockID * NUM_NETS + which_net_index;

            //Capture Winner Index in that net
            int myWinner = min_dist_pos[min_dist_post_net_index];

            //Calculating the Neighbourhood distance for each neuron from its winner
            int neighbourhood_value = neighbourhood(myWinner, which_net, thread_position_in_map, 0);
            int Neurons_current_net = which_net * which_net;
            int update = 0;

            int neigh_rate_index = learning_rate_Index * NUM_FS * NUM_NETS + feature_space_blockID * NUM_NETS + which_net_index;

            if(neighbourhood_value <= (neigh_rate[neigh_rate_index] * (which_net)))
                update = 1;
            else
                update = 0;


            float cumulative_weight = 0;
            float neighbour_adjust = (1 - (float)(neighbourhood_value)/Neurons_current_net);

            //Find the corresponding Map Array Index
            for (int feature = 0; feature < no_of_features_featurespace_end-no_of_features_featurespace_start; feature++){
                float temp =  map[map_start_index + feature];
                float difference_map_input = map[map_start_index + feature] - chunk_inp[chunck_index * NUM_FEATURES + feature + no_of_features_featurespace_start];
                map[map_start_index + feature] = map[map_start_index + feature] - (neighbour_adjust * difference_map_input * update * learning_rate_list[learning_rate_Index]);
                //printf("\n map %f", map[map_start_index + feature]);

                cumulative_weight = cumulative_weight + fabs(map[map_start_index + feature] - temp);
            }
            cumulative_weight_change_per_neuron[thread_id] = cumulative_weight;
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
    }
}
