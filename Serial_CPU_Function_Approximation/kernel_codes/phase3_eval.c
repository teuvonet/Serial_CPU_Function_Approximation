#include<stdio.h>
#include<math.h>
#include<stdlib.h>

void checker(){
    printf("\nCheck");
}

void loop_print(float *a, int n){
    int i;
    // printf("\n%d %d",a[0],a[1]);
    for(i=0; i<n; i++){
        printf("\t%2d %3.3f ",i, a[i]);
    }
}

void phase3_neuron_eval(
    int TOTAL_NUM_NEURONS, // [3,4,5] = 50
    int NUM_NETS, //[3,4,5] = 3 , [4,5] = 2
    int DATA_SIZE, // datapoints to collect in each streaming operation
    int NUM_FS, //Number of feature spaces
    int NUM_FEATURES, //Number of features in given dataset
    int total_num_features_all_featurespaces, //Cumulative sum of total features in all feature spaces
    int no_Passes_Phase3, //Number of passes through the data
    int *list_num_features_featurespaces, //List of features in each feature spaces
    int *cumulative_no_features_featurespace, //Cumulative sum of feature til particulat feature space
    int *neurons_per_nets,// number of neurons per each net .i.e [9,16,25]
    float *map, // the centers of all neurons across all nets
    int *active_centers,
    float *data_inp, // input data
    float *distance_map, // the distance of each neuron from the data points
    float *weights_array,
    float *f_array,
    float *radius_map,
    float *learning_rate_list,
    int *net_sizes, // [3,4,5]
    int euclidean,
    int thread_id,
    int feature_space_blockID,
    int learning_rate_Index
)
{
    //const int thread_id = get_global_id(0) + get_global_id(1)*get_global_size(0) ;

    const int f_Index = NUM_FS * DATA_SIZE * learning_rate_Index +  DATA_SIZE * (NUM_FS * TOTAL_NUM_NEURONS/TOTAL_NUM_NEURONS);

    //int feature_space_blockID = get_global_id(0)/TOTAL_NUM_NEURONS; //Feature Space Number if Grid X direction
    int no_of_features_featurespace = list_num_features_featurespaces[feature_space_blockID]; //Get the Number of features that in each feature space
    //int learning_rate_Index = get_global_id(1); //Learning Rate Index for every thread

    //Find the corresponding Map Array Index
    int map_start_index = learning_rate_Index * TOTAL_NUM_NEURONS * total_num_features_all_featurespaces; // skip one entire learning rate neurons
    map_start_index += TOTAL_NUM_NEURONS * cumulative_no_features_featurespace[feature_space_blockID]; // skip all blocks in same learning space before it
    map_start_index += no_of_features_featurespace * (thread_id % TOTAL_NUM_NEURONS); // skip all neurons before it in the same learning rate and block

    // if (thread_id == 0) loop_print(data_inp,DATA_SIZE*NUM_FEATURES);

    for(int data_index = 0; data_index<DATA_SIZE; data_index++)
    {   
        
        float sum=0;            
        for (int feature = 0; feature < no_of_features_featurespace; feature++){
            if (euclidean)
                sum += (data_inp[data_index * NUM_FEATURES + feature] - map[map_start_index + feature])*(data_inp[data_index * NUM_FEATURES + feature] - map[map_start_index + feature]);    
            else
                sum += fabs(data_inp[data_index * NUM_FEATURES + feature] - map[map_start_index + feature]);
        }

        distance_map[thread_id] = sum;
        // if (thread_id < 50 && data_index == 0) printf("\n Thread %2d Dist %2.2f Active %2d Radius %2.2f", thread_id, distance_map[thread_id], active_centers[thread_id], radius_map[thread_id]);

       //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        //______________________________________________________________________________________________________________________________________________________
        // Step 2: Find prediction
        //------------------------------------------------------------------------------------------------------------------------------------------------------

        if(((NUM_FS * TOTAL_NUM_NEURONS)%TOTAL_NUM_NEURONS) == 0)
        {
            //Distance Map index adjusting
            int distance_map_Index = thread_id;
            float sum = 0;

            //Loop through each net
            for(int net = 0; net < NUM_NETS; net++)
            {
                //Loop through every neuron in that net
                for(int neuron = 0; neuron < neurons_per_nets[net]; neuron++)
                {
                    if(active_centers[distance_map_Index+neuron] > 1)
                    {
                        // If manhattan just radius in denominator
                        float d = distance_map[distance_map_Index+neuron], r = radius_map[distance_map_Index+neuron];
                        float term;
                        if (euclidean){
                            term = exp(-d/((r*r)+0.00001));
                            if(d<=3*r){
                                // printf("Pass %2d LR %2d Data Index %2d NeuronIndex %3d Active %3d Input %3.3f %3.3f Distance %3.3f Wt %3.3f Term %3.3f Sum %3.3f Target %3.3f\n",pass,blockIdx.y,data_index,distance_map_Index+neuron,active_centers[distance_map_Index+neuron],data_inp[data_index*NUM_FEATURES], data_inp[data_index*NUM_FEATURES+1],distance_map[distance_map_Index+neuron],weights_array[distance_map_Index+neuron],exp(-d/(r+0.00001)),weights_array[distance_map_Index+neuron]*term,target[f_Index+data_index]);
                                sum += weights_array[distance_map_Index+neuron]*term;
                            }
                        }
                        else{
                            term = exp(-d/(r+0.00001));
                            if(d<=r){
                                // printf("Pass %2d LR %2d Data Index %2d NeuronIndex %3d Active %3d Input %3.3f %3.3f Distance %3.3f Wt %3.3f Term %3.3f Sum %3.3f Target %3.3f\n",pass,blockIdx.y,data_index,distance_map_Index+neuron,active_centers[distance_map_Index+neuron],data_inp[data_index*NUM_FEATURES], data_inp[data_index*NUM_FEATURES+1],distance_map[distance_map_Index+neuron],weights_array[distance_map_Index+neuron],exp(-d/(r+0.00001)),weights_array[distance_map_Index+neuron]*term,target[f_Index+data_index]);
                                sum += weights_array[distance_map_Index+neuron]*term;
                            }
                        }
                    }
                }
                //Adjust distance map after every net
                distance_map_Index += neurons_per_nets[net];
            }
            f_array[f_Index + data_index] = sum;
        }
        //barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}
