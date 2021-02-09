/*
This code is licensed and documented by Teuvonet Technologies.
Any use of this code, proprietary or personal, needs approval from the company.
*/
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
void check()
{
    printf("\nCheck");
}

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
        return fmax(abs(a_x-b_x) , abs(a_y-b_y));
    }
}

void loop_print(int *a, int n){
    int i;
    printf("\n%d %d",a[0],a[1]);
    for(i=0; i<n; i++){
        printf(" %d\t",a[i]);
    }
    printf("\n");
}

void phase2(
                        int TOTAL_NUM_NEURONS, // [3,4,5] = 50
                        int NUM_NETS, //[3,4,5] = 3 , [4,5] = 2
                        int CHUNK_SIZE, // datapoints to collect in each streaming operation
                        int NUM_FS, //Number of feature spaces
                        int NUM_FEATURES, //Number of features in given dataset
                        int total_num_features_all_featurespaces, //Cumulative sum of total features in all feature spaces
                        int no_Passes_Phase1, //Number of passes through the data
                        int *list_num_features_featurespaces, //List of features in each feature spaces
                        int *cumulative_no_features_featurespace, //Cumulative sum of feature til particulat feature space
                        int *neurons_per_nets,// number of neurons per each net .i.e [9,16,25]
                        float *map, // the centers of all neurons across all nets
                        int *min_dist_pos, //Minimum distance array position
                        float *min_dist_array,//Minimum distance array
                        float *chunk_inp, // input data
                        float *distance_map, // the distance of each neuron from the data points
                        float *neigh_rate,
                        float *learning_rate_list,
                        float *cumulative_weight_change_per_neuron,
                        int *net_sizes,
                        int thread_id,
                        int feature_space_blockID,
                        int learning_rate_Index
                                                 // [3,4,5]
                    )
{

    //const int thread_id = get_global_id(0) + get_global_id(1)*get_global_size(0);


    //int feature_space_blockID = get_global_id(0)/TOTAL_NUM_NEURONS; //Feature Space Number if Grid X direction
    int no_of_features_featurespace = list_num_features_featurespaces[feature_space_blockID]; //Get the Number of features that in each feature space
    //int learning_rate_Index = get_global_id(1); //Learning Rate Index for every thread

    int which_net_index = 0;
    int which_net = 0;
    int net_end_index = learning_rate_Index * NUM_FS *  TOTAL_NUM_NEURONS + feature_space_blockID * TOTAL_NUM_NEURONS;
    int net_start_index = learning_rate_Index * NUM_FS * TOTAL_NUM_NEURONS + feature_space_blockID * TOTAL_NUM_NEURONS;
    int thread_position_in_map = 0;
    int net;
    int map_start_index = learning_rate_Index * TOTAL_NUM_NEURONS * total_num_features_all_featurespaces; // skip one entire learning rate neurons
    map_start_index += TOTAL_NUM_NEURONS * cumulative_no_features_featurespace[feature_space_blockID]; // skip all blocks in same learning space before it
    map_start_index += no_of_features_featurespace * (thread_id % TOTAL_NUM_NEURONS); // skip all neurons before it in the same learning rate and block

    //printf("\n map start index %d ", map_start_index);
    //Finding Thread details - Which net it belongs to?
    for(net = 0; net < NUM_NETS; net++)
    {
        net_end_index = net_end_index + neurons_per_nets[net];

        if(thread_id >= net_start_index && thread_id < net_end_index)
        {
            which_net_index = net;
            which_net = net_sizes[net];
            thread_position_in_map = thread_id - net_start_index;
        }
        net_start_index = net_end_index;
    }

    //printf("\n start %d end %d thread %d feature_space_blockID %d, map_start_index %d", net_start_index,net_end_index, thread_id, feature_space_blockID, map_start_index);
    for(int pass = 0; pass < no_Passes_Phase1; pass++){

        int chunck_index = 0;
        //printf("\n distance map thread id %f", distance_map[thread_id]);
        for(; chunck_index<CHUNK_SIZE; chunck_index++)
        {

            float sum=0;

            for (int feature = 0; feature < no_of_features_featurespace; feature++){
                sum += fabs(chunk_inp[chunck_index * NUM_FEATURES + feature] - map[map_start_index + feature]);
                //printf("\n map %f", map[map_start_index + feature]);

            }

            distance_map[thread_id] = sum;
            //printf("\n distance_map[thread_id] %f", distance_map[thread_id]);

            //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

            if((NUM_FS*TOTAL_NUM_NEURONS)%TOTAL_NUM_NEURONS == 0)
            {
                //Distance Map index adjusting
                int distance_map_Index = thread_id;
                int min_pos_Index = (thread_id/TOTAL_NUM_NEURONS)*NUM_NETS;

                //Loop through each net
                int net = 0;
                for(; net < NUM_NETS; net++)
                {
                    min_dist_array[min_pos_Index + net] = 2147483647;
                    //Loop through every neuron in that net
                    for(int neuron = 0; neuron < neurons_per_nets[net]; neuron++)
                    {
                        // printf("Chunck %d learningrate %d thread_id %d neuron %d distance %f minDist %f\n", chunck_index, learning_rate_Index, thread_id, neuron, distance_map[distance_map_Index + neuron], min_dist_array[min_pos_startIndex + net]);
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
            //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


            //______________________________________________________________________________________________________________________________________________________
            //END OF STEP 2
            //----------------------------------------------------------------------------------------------------------------------------------------------------------

            //_____________________________________________________________________________________________________________________________________________________
            //STEP 4: Seeing the winner, calculating the neighbout value and update the Map
            //----------------------------------------------------------------------------------------------------------------------------------------------------


            //Min Dist Pos Array index for each thread to see its winner in that net
            int min_dist_post_net_index = learning_rate_Index * NUM_FS * NUM_NETS + feature_space_blockID * NUM_NETS + which_net_index;

            //Capture Winner Index in that net
            int myWinner = min_dist_pos[min_dist_post_net_index];


            //Calculating the Neighbourhood distance for each neuron from its winner
            int neighbourhood_value = neighbourhood(myWinner, which_net, thread_position_in_map, 0);
            int Neurons_current_net = which_net * which_net;
            int update = 0;

            // printf("%d %f\n",neighbourhood_value, neigh_rate[which_net+myWinner]*which_net);

            if(neighbourhood_value <= (neigh_rate[which_net+myWinner] * (which_net)))
                update = 1;
            else
                update = 0;

            float cumulative_weight = 0;

            //Find the corresponding Map Array Index
            for (int feature = 0; feature < no_of_features_featurespace; feature++){
                float temp =  map[map_start_index + feature];
                float neighbour_adjust = (1 - (float)(neighbourhood_value)/Neurons_current_net);
                //printf("\n neighbour_adjust %f", neighbour_adjust);

                float difference_map_input = map[map_start_index + feature] - chunk_inp[chunck_index * NUM_FEATURES + feature];
                map[map_start_index + feature] = (float)(map[map_start_index + feature] - (neighbour_adjust * difference_map_input * update * learning_rate_list[learning_rate_Index]));
                //printf("\n map %f", map[map_start_index + feature]);

                cumulative_weight =cumulative_weight + fabs(map[map_start_index + feature] - temp);
            }
            cumulative_weight_change_per_neuron[thread_id] = cumulative_weight;
            //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }

        // if(pass%5==0)
        learning_rate_list[learning_rate_Index] *= 0.999;

        if(thread_id==0){
            float s = 0;
            for(int i=0; i<TOTAL_NUM_NEURONS; i++)    //comapre the values of the code
                s+=cumulative_weight_change_per_neuron[i];
        }
    }
}
