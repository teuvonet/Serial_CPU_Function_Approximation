void checker(){
    printf("\nCheck");
}

void loop_print(float *a, int n){
    int i;
    // printf("\n%d %d",a[0],a[1]);
    for(i=0; i<n; i++){
        printf("\n %2d %2.2f ",i, a[i]);
    }
}

__kernel void phase3_neuron(
    int TOTAL_NUM_NEURONS, // [3,4,5] = 50
    int NUM_NETS, //[3,4,5] = 3 , [4,5] = 2
    int DATA_SIZE, // datapoints to collect in each streaming operation
    int NUM_FS, //Number of feature spaces
    int NUM_FEATURES, //Number of features in given dataset
    int total_num_features_all_featurespaces, //Cumulative sum of total features in all feature spaces
    int no_Passes_Phase3, //Number of passes through the data
    __global int *list_num_features_featurespaces, //List of features in each feature spaces
    __global int *cumulative_no_features_featurespace, //Cumulative sum of feature til particulat feature space
    __global int *neurons_per_nets,// number of neurons per each net .i.e [9,16,25]
    __global float *map, // the centers of all neurons across all nets
    __global int *active_centers,
    __global float *data_inp, // input data
    __global float *distance_map, // the distance of each neuron from the data points
    __global float *target,
    __global float *weights_array,
    __global float *f_array,
    __global float *radius_map,
    __global float *error_train,
    __global float *error_test,
    __global float *error_pass,
    __global float *error_train_prev,
    __global float *error_test_prev,
    __global float *error_train_difference,
    __global float *error_test_difference,
    float convergence_threshold,
    float train_split,
    __global float *learning_rate_list,
    __global int *net_sizes, // [3,4,5]
    int euclidean // Whether to use euclidean distance or manhatten
)
{



    const int thread_id = get_global_id(0) + get_global_id(1)*get_global_size(0) ;

    const int f_Index = NUM_FS * DATA_SIZE * get_global_id(1) +  DATA_SIZE * (get_global_id(0)/TOTAL_NUM_NEURONS);
    const int e_Index = NUM_FS * get_global_id(1) + get_global_id(0)/TOTAL_NUM_NEURONS;

    int feature_space_blockID = get_global_id(0)/TOTAL_NUM_NEURONS; //Feature Space Number if Grid X direction
    int no_of_features_featurespace = list_num_features_featurespaces[feature_space_blockID]; //Get the Number of features that in each feature space
    int learning_rate_Index = get_global_id(1); //Learning Rate Index for every thread

    //Find the corresponding Map Array Index
    int map_start_index = learning_rate_Index * TOTAL_NUM_NEURONS * total_num_features_all_featurespaces; // skip one entire learning rate neurons
    map_start_index += TOTAL_NUM_NEURONS * cumulative_no_features_featurespace[feature_space_blockID]; // skip all blocks in same learning space before it
    map_start_index += no_of_features_featurespace * (thread_id % TOTAL_NUM_NEURONS); // skip all neurons before it in the same learning rate and block
    //printf("NUM_FEATURES");
    //printf("NUM_FEATURES %d    ", no_of_features_featurespace);
    // if(thread_id == 0) loop_print(map, 300);

    for(int pass = 0; pass < no_Passes_Phase3; pass++)
    {
        // Loop for each data point
        for(int data_index = 0; data_index<DATA_SIZE; data_index++)                        //for(int data_index = 0; data_index<DATA_SIZE; data_index++)
        {

            
            float sum=0;            
            for (int feature = 0; feature < no_of_features_featurespace; feature++){
                if (euclidean)
                    sum += (data_inp[data_index * NUM_FEATURES + feature] - map[map_start_index + feature])*(data_inp[data_index * NUM_FEATURES + feature] - map[map_start_index + feature]);
                else
                    sum += fabs(data_inp[data_index * NUM_FEATURES + feature] - map[map_start_index + feature]);
            }

            distance_map[thread_id] = sum;

            //printf("%f/", sum);

             barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

            //______________________________________________________________________________________________________________________________________________________
            // Step 2: Find prediction
            //------------------------------------------------------------------------------------------------------------------------------------------------------

            if(get_global_id(0)%TOTAL_NUM_NEURONS == 0)
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
                //printf("%d   ", data_index);
                //printf("%f/", f_array[f_Index + data_index]);

                float sq_err = (target[f_Index + data_index] - f_array[f_Index + data_index])*(target[f_Index + data_index] - f_array[f_Index + data_index]);
                //printf("%f/", sq_err);
                if(data_index < train_split*DATA_SIZE){
                    error_train[e_Index] +=  sq_err;
                }
                else{
                    error_test[e_Index] += sq_err;
                }
                error_pass[e_Index] += (target[f_Index + data_index] - f_array[f_Index + data_index]);
            }

             barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            
            if(active_centers[thread_id] > 1)
            {
                if(data_index < train_split*DATA_SIZE)
                {

                    float err= 0;


                    err = target[f_Index + data_index]-f_array[f_Index + data_index];
                    //printf("%f/", err);

                    float d = distance_map[thread_id], r = radius_map[thread_id];

                    if (euclidean){
                        if (d<=3*r){
                            // if (thread_id<50)
                                // printf("------Pass %2d Neuron Index %2d Active %2d Data Index %2d Target %3.3f Error %3.3f Wt %3.3f Pred %3.3f Term %3.3f DelW %3.3f\n",pass,thread_id,active_centers[thread_id],data_index,target[f_Index + data_index],err,weights_array[thread_id],f_array[f_Index + data_index],exp(-d/(r+0.00001)),learning_rate_list[learning_rate_Index]*err*exp(-d/(r+0.00001)));
                            float term = exp(-d/((r*r)+0.00001));

                            weights_array[thread_id] += learning_rate_list[learning_rate_Index]*err*exp(-d/((r*r)+0.00001));
                            }
                       }

                    else{
                        if (d<=r){
                                // if (thread_id<50)
                                // printf("------Pass %2d Neuron Index %2d Active %2d Data Index %2d Target %3.3f Error %3.3f Wt %3.3f Pred %3.3f Term %3.3f DelW %3.3f\n",pass,thread_id,active_centers[thread_id],data_index,target[f_Index + data_index],err,weights_array[thread_id],f_array[f_Index + data_index],exp(-d/(r+0.00001)),learning_rate_list[learning_rate_Index]*err*exp(-d/(r+0.00001)));
                            weights_array[thread_id] += learning_rate_list[learning_rate_Index]*err*exp(-d/(r+0.00001));
                        }
                    }
                    if(weights_array[thread_id] > 1)
                        weights_array[thread_id] = 1;
                        
                    if(weights_array[thread_id] < -1)
                        weights_array[thread_id] =-1;

                }
                //printf("%f/", target[f_Index + data_index]);
            }

             barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        }

        if(get_global_id(0)%TOTAL_NUM_NEURONS == 0 && get_global_id(0)/TOTAL_NUM_NEURONS == 0 && pass%2 == 0)
        {
            if(learning_rate_list[get_global_id(1)]>0.0001)
            {
                learning_rate_list[get_global_id(1)] = learning_rate_list[get_global_id(1)] * 0.99;
            }
            else
            {
                learning_rate_list[get_global_id(1)] = 0.0001;
            }                    
        }
         barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        if(pass>0 && pass%2 == 0 || pass == no_Passes_Phase3-1){
            error_train[e_Index] = sqrt(error_train[e_Index]/(train_split*DATA_SIZE));
            error_test[e_Index] = sqrt(error_test[e_Index]/((1.0-train_split)*DATA_SIZE));
            error_train_difference[e_Index] = fabs(error_train[e_Index] - error_train_prev[e_Index]);
            error_test_difference[e_Index] = fabs(error_test[e_Index] - error_test_prev[e_Index]);
         barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            float train_avg_dif=0, test_avg_diff = 0;
            if(thread_id == 0){
                for(int l=0; l<NUM_FS*get_global_size(1); l++){
                    train_avg_dif += error_train_difference[l];
                    test_avg_diff += error_test_difference[l];
                }
                train_avg_dif /= NUM_FS*get_global_size(1);
                test_avg_diff /= NUM_FS*get_global_size(1);
            }
            if((train_avg_dif<convergence_threshold || test_avg_diff<convergence_threshold) && thread_id == 0){
                //printf("Converged in %2d passes\n",pass+1);
                pass = no_Passes_Phase3;
                break;
            }
            error_train_prev[e_Index] = error_train[e_Index];
            error_test_prev[e_Index] = error_test[e_Index];
        }
        if(pass == no_Passes_Phase3-1 && thread_id == 0){
            printf("No Convergence so far\n");
        }
    }


}
