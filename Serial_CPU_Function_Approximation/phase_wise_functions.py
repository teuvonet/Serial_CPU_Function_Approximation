'''
This code is licensed and documented by Teuvonet Technologies.
Any use of this code, proprietary or personal, needs approval from the company.

'''

#Import Statements
import os
import sys
import copy
import numpy as np
import pandas as pd
import pyopencl as cl
import pyopencl.algorithm as algo
import pyopencl.array as pycl_array
from sklearn.metrics import r2_score
import ctypes

# np.random.seed(101)

def commenter(s):
    print("-"*100)
    print(s)


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# def stacking_model(new_weights,predictions,target,y,num_stack_passes = 2,train=True):
#     if train:
#         for _ in range(num_stack_passes):
#             y = np.zeros_like(y)
#             err = np.zeros_like(y)
#             for d in range(target.shape[2]):
#                 y[:,:,d] = np.sum(np.multiply(new_weights,predictions[:,:,:,d]), axis=0)
#                 # print("new_weights1")
#                 # print(new_weights)
#                 # print("predictions")
#                 # print(predictions)
#                 # print("y")
#                 # print(y)
#             err = target-y
#             for i in range(len(new_weights)):
#                 new_weights[i,:,:] += 0.5*np.mean(err*predictions[i,:,:,:],axis=2)
#                 new_weights[new_weights>1] = 1
#                 new_weights[new_weights<-1] = -1
#                 # print("new_weights2")
#                 # print(new_weights)
#             # print("new_weights")
#             # print(new_weights)
#         return new_weights,y
#     else:
#         for d in range(target.shape[2]):
#             y[:,:,d] = np.sum(new_weights*predictions[:,:,:,d],axis=0)
#             # print("new_weights")
#             # print(new_weights)
#         return y

def phase1(data_inp, 
           CHUNK_SIZE, 
           NUM_FEATURES, 
           NUM_FS, 
           NUM_NEURONS, 
           NUM_NETS, 
           learning_rate_list, 
           len_learning_rates, 
           list_num_features_featurespaces_start,
           list_num_features_featurespaces,
           total_num_features_all_featurespaces, 
           cumulative_no_features_featurespace, 
           NET_SIZES, 
           no_passes_phase1,
           NUM_DATASET_PASSES, 
           nr_multiplier=1):


    myFloatType = np.float32  # Global Float Type for Kernel
    myIntType = np.int32  # Global int Type for Kernel

    #Cumulative sum of number of neurons in net
    neurons_per_nets = np.array([i**2 for i in NET_SIZES],dtype=np.int32)
    #Base Map will have the neuron centers. Size = NoOfneuron * NoOfFeatures
    base_map_size = int(len_learning_rates*NUM_NEURONS*total_num_features_all_featurespaces)
    base_map = np.random.uniform(low=0, high=1, size=(base_map_size,)).astype(np.float32)

    #Distance map array stores the distance between neuron center and Input data of every feature
    distance_base_map = np.zeros((len_learning_rates, NUM_FS, NUM_NEURONS), dtype=np.float32)

    #Array to store minimum distance neuron for each net.
    min_dist_pos = np.array(np.full((len_learning_rates, NUM_FS, NUM_NETS), sys.maxsize), dtype=np.int32)
    #Array to store minimum distance position of a neuron in each net
    min_dist_array = np.array(np.full((len_learning_rates, NUM_FS, NUM_NETS), sys.maxsize), dtype=np.float32)

    #NeighBourRate Array
    neigh_rate = np.array(np.ones(NUM_NETS * NUM_FS * len_learning_rates)*nr_multiplier, dtype = np.float32)
    new_neigh_rate = copy.deepcopy(neigh_rate)
    new_learning_rate_list = copy.deepcopy(learning_rate_list)
    # cumulative weight change
    cumulative_weight_change_per_neuron = np.zeros((len_learning_rates, NUM_FS, NUM_NEURONS), dtype=np.float32)
    NET_SIZES = np.array(NET_SIZES, dtype=np.int32)

    # print("NET_SIZES", NET_SIZES)
    #CALLING THE KERNEL FUNCTION. phase2.cu - File Name

    
    #Calling phase2 function in the given kernel file

    
    # commenter("Finding New Map Positions")

    for l in range(NUM_DATASET_PASSES):
        print("Dataset Pass ", l + 1)
        sum_cum_wt_change = 0
        NUM_LEARNING_RATES = len(learning_rate_list)
        for learning_rate_Index in range(0, NUM_LEARNING_RATES):
            print("learning_rate_Index", learning_rate_Index)
            thread_id = 0 + NUM_FS * NUM_NEURONS * learning_rate_Index
            for j in range(0, NUM_FS):
                for k in range(0, NUM_NEURONS):
                    #

                    #print("k ,thread_id", k, thread_id)
                # print("Dataset Pass ", l+1)
                    for i in range(0, len(data_inp), CHUNK_SIZE):
                        chunk_inp = data_inp.iloc[i:i + CHUNK_SIZE, :].values.astype(np.float32).ravel()
                        chunk_inp = np.array(chunk_inp, dtype=myFloatType)
                        # print(np.count_nonzero(chunk_inp>1))

                        # Define the sizes of the blocks and grids
                        #grid_X, grid_Y = NUM_FS * NUM_NEURONS, len_learning_rates
                        # workUnits_X, workUnits_Y, workUnits_Z = NUM_NEURONS, 1, 1
                        #print("thread_id", thread_id)
                        feature_space_blockID = j
                        # print("Chunk ",i+1)
                        # Calling kernel function with Parameters. Lookout for InOut or In
                        fun = ctypes.CDLL("/home/asim/Function Approximation Serial CPU/kernel_codes/phase1.so")
                        fun.phase1.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                        np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
                                                        np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
                                                        np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
                                                        np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
                                                        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                        np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
                                                        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                        np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
                                                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
                        returnVale = fun.phase1(
                                       np.int32(NUM_NEURONS),
                                       np.int32(NUM_NETS),
                                       np.int32(data_inp.iloc[i:i + CHUNK_SIZE, :].shape[0]),
                                       np.int32(NUM_FS),
                                       np.int32(NUM_FEATURES),
                                       np.int32(total_num_features_all_featurespaces),
                                       np.int32(no_passes_phase1),
                                       list_num_features_featurespaces_start,
                                       list_num_features_featurespaces,
                                       cumulative_no_features_featurespace,
                                       neurons_per_nets,
                                       base_map,
                                       min_dist_pos,
                                       min_dist_array,
                                       chunk_inp,
                                       distance_base_map,
                                       new_neigh_rate,
                                       new_learning_rate_list,
                                       cumulative_weight_change_per_neuron,
                                       NET_SIZES,
                                       thread_id,
                                       feature_space_blockID,
                                       learning_rate_Index)
                        #print("returnVale", returnVale)
                        #print("map_data", base_map)
                    thread_id = thread_id + 1
                    #print("iterartion over")

        cumulative_weight_change_per_neuron_new = cumulative_weight_change_per_neuron.reshape((len_learning_rates, -1))
        sum_cum_wt_change += np.sum(cumulative_weight_change_per_neuron_new, axis=1)
        if (i % 100 == 0):
            new_neigh_rate = np.array([lr * 0.95 for lr in new_neigh_rate], dtype=np.float32)
            new_learning_rate_list = np.array([lr * 0.99 for lr in new_learning_rate_list], dtype=np.float32)
                            # print(new_neigh_rate, new_learning_rate_list)
        print("First phase")
        # print(max(base_map), min(base_map))
        # print(np.round(sum_cum_wt_change, decimals=3))

    # print("map_data", base_map)
    # print(max(base_map), min(base_map))
    # print(np.round(sum_cum_wt_change, decimals=3))

    # Distance map array stores the distance between neuron center and Input data of every feature
    distance_base_map = np.zeros((len_learning_rates, NUM_FS, NUM_NEURONS), dtype=np.float32)
    # Array to store minimum distance neuron for each net.
    min_dist_pos = np.array(np.full((len_learning_rates, NUM_FS, NUM_NETS), sys.maxsize), dtype=np.int32)
    # Array to store minimum distance position of a neuron in each net
    min_dist_array = np.array(np.full((len_learning_rates, NUM_FS, NUM_NETS), sys.maxsize), dtype=np.float32)
    # active centers contains number of datapoints belong to each neuron
    active_centers = np.array(np.zeros((len_learning_rates, NUM_FS, NUM_NEURONS)), dtype=np.int32)
    data_inp1 = data_inp.values.flatten('C').astype(np.float32)


    #Define the sizes of the blocks and grids

    commenter("Finding Active Centers")
    commenter("Finding Active Centers")
    for learning_rate_Index in range(0, NUM_LEARNING_RATES):
        print("learning_rate_Index", learning_rate_Index)
        thread_id = 0 + NUM_FS * NUM_NEURONS * learning_rate_Index
        for j in range(0, NUM_FS):
            for k in range(0, NUM_NEURONS):
                #print("k ,thread_id", k, thread_id)
                feature_space_blockID = j
                active_center_cal = ctypes.CDLL("/home/asim/Function Approximation Serial CPU/kernel_codes/phase1_activecenters.so")
                active_center_cal.phase1_active_centers.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                            ctypes.c_int, ctypes.c_int, ctypes.c_int]

                returnVale = active_center_cal.phase1_active_centers(np.int32(NUM_NEURONS),
                                  np.int32(NUM_NETS),
                                  np.int32(NUM_FEATURES),
                                  np.int32(data_inp.shape[0]),
                                  np.int32(NUM_FS),
                                  np.int32(total_num_features_all_featurespaces),
                                  list_num_features_featurespaces,
                                  cumulative_no_features_featurespace,
                                  neurons_per_nets,
                                  base_map,
                                  active_centers,
                                  min_dist_pos,
                                  min_dist_array,
                                  data_inp1,
                                  distance_base_map,
                                  NET_SIZES,
                                  thread_id,
                                  feature_space_blockID,
                                  learning_rate_Index)

    # print(base_map)
    print("Phase 1 part 2 is done")
    # commenter("Finding Distances Moved")


    #Define the sizes of the blocks and grids

    overall_distances = np.array(np.zeros((NUM_FEATURES * len_learning_rates)), dtype=np.float32)
    for learning_rate_Index in range(0, NUM_LEARNING_RATES):
        print("learning_rate_Index", learning_rate_Index)
        thread_id = 0 + NUM_NEURONS * learning_rate_Index

        for k in range(0, NUM_NEURONS):
            # print("k ,thread_id", k, thread_id)
            #print("thread_id", thread_id)

            dist_cal = ctypes.CDLL("/home/asim/Function Approximation Serial CPU/kernel_codes/phase1_distance_computation.so")
            dist_cal.phase1_distance_computation.argtypes = [
                                                            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                            ctypes.c_int,
                                                            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                            ctypes.c_int, ctypes.c_int,ctypes.c_int,

                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),

                                                            ctypes.c_int, ctypes.c_int]
            returnVale = dist_cal.phase1_distance_computation(
                                        base_map,
                                        np.int32(NUM_FEATURES),
                                        overall_distances,
                                        np.int32(NUM_NEURONS),
                                        np.int32(NUM_FS),
                                        np.int32(NUM_NETS),
                                        neurons_per_nets,
                                        list_num_features_featurespaces,
                                        active_centers,
                                        thread_id,
                                        learning_rate_Index

                                          )
            thread_id = thread_id + 1

    # print("overall distances", overall_distances)
    return overall_distances
print("end of phase 1")
# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------

def phase2(data_inp, 
           CHUNK_SIZE, 
           NUM_FEATURES, 
           NUM_FS, 
           NUM_NEURONS, 
           NUM_NETS, 
           learning_rate_list, 
           len_learning_rates, 
           list_num_features_featurespaces,
           total_num_features_all_featurespaces, 
           cumulative_no_features_featurespace, 
           NET_SIZES, 
           no_passes_phase2, 
           nr_multiplier=1):



    mem_flags = cl.mem_flags
    myFloatType = np.float32  # Global Float Type for Kernel
    myIntType = np.int32  # Global int Type for Kernel

    #Cumulative sum of number of neurons in net
    neurons_per_nets = np.array([i**2 for i in NET_SIZES],dtype=np.int32)
    #Base Map will have the neuron centers. Size = NoOfneuron * NoOfFeatures
    base_map_size = int(len_learning_rates * NUM_NEURONS * total_num_features_all_featurespaces)
    base_map = np.random.uniform(low=0, high=1, size=(base_map_size,)).astype(np.float32)
    #Distance map array stores the distance between neuron center and Input data of every feature
    distance_base_map = np.zeros((len_learning_rates, NUM_FS, NUM_NEURONS), dtype=np.float32)
    #Array to store minimum distance neuron for each net.
    min_dist_pos = np.array(np.full((len_learning_rates, NUM_FS, NUM_NETS), sys.maxsize), dtype=np.int32)
    #Array to store minimum distance position of a neuron in each net
    min_dist_array = np.array(np.full((len_learning_rates, NUM_FS, NUM_NETS), sys.maxsize), dtype=np.float32)
    #NeighBourRate Array
    neigh_rate = np.array(np.ones(NUM_NETS * NUM_FS * len_learning_rates)*nr_multiplier, dtype = np.float32)
    # cumulative weight change
    cumulative_weight_change_per_neuron = np.zeros((len_learning_rates, NUM_FS, NUM_NEURONS ), dtype=np.float32)
                                                            # write buffer
    #active centers contains number of datapoints belong to each neuron
    active_centers = np.array(np.zeros((len_learning_rates,NUM_FS,NUM_NEURONS)), dtype = np.int32)

    #Radius map array stores the distance between neuron center and Input data of every feature
    radius_base_map = np.zeros((len_learning_rates, NUM_FS, NUM_NEURONS), dtype=np.float32)

    NET_SIZES = np.array(NET_SIZES, dtype=np.int32)
    data_inp_1 = data_inp
    data_inp = data_inp.values.flatten('C').astype(np.float32)

    # print("base_map")
    # print(base_map)
    # print("data_inp")
    # print(data_inp)





    NUM_LEARNING_RATES = len(learning_rate_list)

    for learning_rate_Index in range(0, NUM_LEARNING_RATES):
        sum_cum_wt_change = 0
        print("learning_rate_Index", learning_rate_Index)
        thread_id = 0 + NUM_FS * NUM_NEURONS * learning_rate_Index
        for j in range(0, NUM_FS):
            for k in range(0, NUM_NEURONS):
                #print("k ,thread_id", k, thread_id)


                #-9print("thread_id", thread_id)
                feature_space_blockID = j
                # print("Chunk ",i+1)
                # Calling kernel function with Parameters. Lookout for InOut or In
                fun = ctypes.CDLL("/home/asim/Function Approximation Serial CPU/kernel_codes/phase2.so")
                fun.phase2.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),

                                       np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(dtype=myIntType, flags="C_CONTIGUOUS"),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int]

                #Calling kernel function with Parameters. Lookout for InOut or In
                returnVale = fun.phase2(
                               np.int32(NUM_NEURONS),
                               np.int32(NUM_NETS),
                               np.int32(data_inp_1.shape[0]),
                               np.int32(NUM_FS),
                               np.int32(NUM_FEATURES),
                               np.int32(total_num_features_all_featurespaces),
                               np.int32(no_passes_phase2),
                               list_num_features_featurespaces,
                               cumulative_no_features_featurespace,
                               neurons_per_nets,
                               base_map,
                               min_dist_pos,
                               min_dist_array,
                               data_inp,
                               distance_base_map,
                               neigh_rate,
                               learning_rate_list,
                               cumulative_weight_change_per_neuron,
                               NET_SIZES,
                               thread_id,
                               feature_space_blockID,
                               learning_rate_Index
                               )
                thread_id = thread_id + 1

    # print(cumulative_weight_change_per_neuron)

    cumulative_weight_change_per_neuron_new = cumulative_weight_change_per_neuron.reshape((len_learning_rates, -1))
    sum_cum_wt_change += np.sum(cumulative_weight_change_per_neuron_new, axis=1)
    print("Phase2 part 1")
    # print("base_map")
    # print(base_map)
    # print("distance_base_map")
    # print(distance_base_map)
    # print(cumulative_weight_change_per_neuron_new)
    #Distance map array stores the distance between neuron center and Input data of every feature
    distance_base_map = np.zeros((len_learning_rates, NUM_FS, NUM_NEURONS), dtype=np.float32)
    #Array to store minimum distance neuron for each net.
    min_dist_pos = np.array(np.full((len_learning_rates,NUM_FS,NUM_NETS), sys.maxsize), dtype=np.int32)
    #Array to store minimum distance position of a neuron in each net
    min_dist_array = np.array(np.full((len_learning_rates, NUM_FS, NUM_NETS), sys.maxsize), dtype=np.float32)
    active_centers = np.array(np.zeros((len_learning_rates, NUM_FS, NUM_NEURONS)), dtype=np.int32)
    radius_base_map = np.zeros((len_learning_rates, NUM_FS, NUM_NEURONS), dtype=np.float32)
    base_map_size = int(len_learning_rates * NUM_NEURONS * total_num_features_all_featurespaces)
    base_map = np.random.uniform(low=0, high=1, size=(base_map_size,)).astype(np.float32)
    # print("radius_base_map")
    # print(radius_base_map)
    #CALLING THE KERNEL FUNCTION. phase2.cu - File Name

    # Calling phase2 function in the given kernel file
    # mod = SourceModule(kernel_phase2_cr)
    # kernel_phase2_cr = mod.get_function("phase2_calculateradius")

    # Define the sizes of the blocks and grids
    grid_X, grid_Y = NUM_FS * NUM_NEURONS, len_learning_rates
    # workUnits_X, workUnits_Y, workUnits_Z = NUM_NEURONS, 1, 1
    for learning_rate_Index in range(0, NUM_LEARNING_RATES):
        print("learning_rate_Index", learning_rate_Index)
        thread_id = 0 + NUM_FS * NUM_NEURONS * learning_rate_Index
        for j in range(0, NUM_FS):
            for k in range(0, NUM_NEURONS):
                #print("k ,thread_id", k, thread_id)
                #print("thread_id", thread_id)
                feature_space_blockID = j
                radius_cal = ctypes.CDLL("/home/asim/Function Approximation Serial CPU/kernel_codes/phase2_calculateradius.so")
                print("fail is here")
                radius_cal.phase2_calculateradius.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                            ctypes.c_int, ctypes.c_int,
                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                            ctypes.c_int, ctypes.c_int, ctypes.c_int]
                print("fail is after  array")
    #Calling kernel function with Parameters. Lookout for InOut or In
                returnVale = radius_cal.phase2_calculateradius(
                                   np.int32(NUM_NEURONS),
                                   np.int32(NUM_NETS),
                                   np.int32(data_inp_1.shape[0]),
                                   np.int32(NUM_FEATURES),
                                   np.int32(NUM_FS),
                                   np.int32(total_num_features_all_featurespaces),
                                   neurons_per_nets,
                                   list_num_features_featurespaces,
                                   cumulative_no_features_featurespace,
                                   base_map,
                                   min_dist_pos,
                                   min_dist_array,
                                   distance_base_map,
                                   radius_base_map,
                                   active_centers,
                                   data_inp,
                                   thread_id,
                                   feature_space_blockID,
                                   learning_rate_Index
                                     )
                thread_id = thread_id + 1
            print("Partion")
        print("Learning rate")
    print("outer Learning rate")
    print("radius_base_map")
    print(active_centers)
    print("radius_base_map")
    print(radius_base_map)
    print("base_map")
    print(base_map)
    print("distance_base_map")
    print(distance_base_map)
    return radius_base_map.ravel(), base_map.ravel(), active_centers, distance_base_map.ravel()


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------

def phase3(main_pred_list, target_list, difference_list, data_inp,
           test_data, 
           test_targets,
           DATA_SIZE, 
           targets, 
           scaler, 
           NUM_FEATURES, 
           NUM_FS, 
           NUM_NEURONS, 
           NUM_NETS,
           prev_lrs,
           train_split, 
           learning_rate_list, 
           len_learning_rates, 
           list_num_features_featurespaces,
           total_num_features_all_featurespaces, 
           cumulative_no_features_featurespace, 
           NET_SIZES,
           base_map, 
           radius_map, 
           active_centers,
           num_boosting_trials, 
           no_passes_phase3, 
           convergence_threshold,
           lambda_phase3s,
           stack_passes=50):


    myFloatType = np.float32  # Global Float Type for Kernel
    myIntType = np.int32  # Global int Type for Kernel

    #Cumulative sum of number of neurons in net
    neurons_per_nets = np.array([i**2 for i in NET_SIZES],dtype=np.int32)



    data_inp_1 = data_inp

    data_inp = data_inp.values.flatten('C').astype(np.float32)



    NET_SIZES = np.array(NET_SIZES, dtype=np.int32)


    test_data_1 = test_data
    test_data = test_data.values.flatten('C').astype(np.float32)








    main_preds = np.zeros((len_learning_rates,NUM_FS,DATA_SIZE), dtype=np.float32)
    test_preds = np.zeros((len_learning_rates,NUM_FS,test_data_1.shape[0]), dtype=np.float32)
    avg_target = np.mean(targets)
    # main_preds = main_preds + avg_target
    # print("main prediction")
    # print(main_preds)
    main_preds = main_preds + avg_target
    test_preds = test_preds + avg_target
    main_learning_rates = copy.deepcopy(learning_rate_list)


    for lambda_phase3 in lambda_phase3s:
        all_fs_targets = np.reshape([[targets-avg_target]*NUM_FS]*len_learning_rates, (len_learning_rates,NUM_FS,-1))
        all_fs_test_targets = np.reshape([[test_targets - avg_target] * NUM_FS] * len_learning_rates,
                                    (len_learning_rates, NUM_FS, -1))

        cur_targets = np.reshape([[targets-avg_target]*NUM_FS]*len_learning_rates, (len_learning_rates,NUM_FS,-1))
        cur_targets = cur_targets.astype(np.float32)



        learning_rate_list = copy.deepcopy(main_learning_rates)
        weights,main_predictions,test_predictions = [],[],[]

        for bt in range(num_boosting_trials):
            commenter("Booting Trial {}".format(bt+1))
            print(lambda_phase3)
            # print(np.around(cur_targets[0,0,:5],decimals=3))
            # weights array
            glorot_init = 0
            weights_array = np.random.uniform(-glorot_init,glorot_init,size=(len_learning_rates,NUM_FS,NUM_NEURONS)).astype(np.float32)    #(-glorot_init,glorot_init,size=(len_learning_rates,NUM_FS,NUM_NEURONS)).astype(np.float32)

            # print("weights_array1")
            # print(weights_array)

            # get predictions from all neurons
            f_array = np.zeros((len_learning_rates, NUM_FS, DATA_SIZE), dtype=np.float32)

            # error_train
            err_train = np.zeros((len_learning_rates,NUM_FS), dtype=np.float32)

            # error test
            err_test = np.zeros((len_learning_rates,NUM_FS), dtype=np.float32)

            # error train prev
            err_train_prev = np.zeros((len_learning_rates,NUM_FS), dtype=np.float32)

            # error test prev
            err_test_prev = np.zeros((len_learning_rates, NUM_FS), dtype=np.float32)

            # error train err_test_prev
            err_train_difference = np.zeros((len_learning_rates, NUM_FS), dtype=np.float32)

            # error test difference
            err_test_difference = np.zeros((len_learning_rates, NUM_FS), dtype=np.float32)


            # error pass
            err_pass = np.zeros((len_learning_rates, NUM_FS), dtype=np.float32)

            # prediction train and test
            distance_base_map = np.zeros((len_learning_rates, NUM_FS, NUM_NEURONS), dtype=np.float32)

            # print("f_array")
            # print(f_array)
            #NUM_LEARNING_RATES = len(learning_rate_list)
            #Calling kernel function with Parameters. Lookout for InOut or In
            for learning_rate_Index in range(0, NUM_LEARNING_RATES):
                print("learning_rate_Index", learning_rate_Index)
                thread_id = 0 + NUM_FS * NUM_NEURONS * learning_rate_Index
                for j in range(0, NUM_FS):
                    for k in range(0, NUM_NEURONS):
                        # print("k ,thread_id", k, thread_id)
                        # print("thread_id", thread_id)
                        feature_space_blockID = j
                        phase3 = ctypes.CDLL("/home/asim/Function Approximation Serial CPU/kernel_codes/phase3.so")
                        print("fail is here")
                        phase3.phase3_neuron.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                                      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                                      np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      ctypes.c_float, ctypes.c_float,
                                                                      np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      ctypes.c_int, ctypes.c_int, ctypes.c_int,ctypes.c_int, ctypes.c_int ]
                        print("fail is after  array")
                        returnVale = phase3.phase3_neuron(
                        np.int32(NUM_NEURONS),
                        np.int32(NUM_NETS),
                        np.int32(data_inp_1.shape[0]),
                        np.int32(NUM_FS),
                        np.int32(NUM_FEATURES),
                        np.int32(total_num_features_all_featurespaces),
                        np.int32(no_passes_phase3),
                        list_num_features_featurespaces,
                        cumulative_no_features_featurespace,
                        neurons_per_nets,
                        base_map,
                        active_centers,
                        data_inp,
                        distance_base_map,
                        cur_targets,
                        weights_array,
                        f_array,
                        radius_map,
                        err_train,
                        err_test,
                        err_pass,
                        err_train_prev,
                        err_test_prev,
                        err_train_difference,
                        err_test_difference,
                        np.float32(convergence_threshold),
                        np.float32(train_split),
                        learning_rate_list,
                        NET_SIZES,
                        bnp.int32(1),
                        thread_id,
                        feature_space_blockID,
                        learning_rate_Index,
                        len_learning_rates)

                        thread_id = thread_id + 1

            # print(f_array)
            main_preds += f_array*lambda_phase3
            # print("main_preds")
            # print(main_preds)

            main_predictions.append(f_array)
            # print("cur_targets_old")
            # print(cur_targets)
            # print(np.around(f_array[0,0,:5],decimals=3))





            # diff = main_preds - (all_fs_targets + avg_target)
            # difference_list = []
            # main_pred_list = []
            # target_list = []
            # # print("diff")
            # # print(diff)
            # for i in range(len(learning_rate_list)):
            #     for j in range(NUM_FS):
            #         difference_list.append(diff[i][j])
            #         main_pred_list.append(main_preds[i][j])
            #         target_list.append(all_fs_targets[i][j] + avg_target)

            # difference_list = list(map(list, zip(*difference_list)))
            # target_list = list(map(list, zip(*target_list)))
            # main_pred_list = list(map(list, zip(*main_pred_list)))

            # np.savetxt('difference.csv', difference_list, delimiter=',', fmt='%.9f')
            # np.savetxt('main predictions.csv', main_pred_list, delimiter=',', fmt='%.9f')
            # np.savetxt('targets.csv', target_list, delimiter=',', fmt='%.9f')
            cur_targets -= f_array * lambda_phase3


            # print("test_data_1.shape[0]").
            # print(test_data_1.shape[0])
            # print("test_data_1")
            # print(test_data_1)

            # print("cur_targets_new")
            # print(cur_targets)
            f_array = np.zeros((len_learning_rates, NUM_FS, test_data_1.shape[0]), dtype=np.float32)
            distance_base_map = np.zeros((len_learning_rates, NUM_FS, NUM_NEURONS), dtype=np.float32)


            for learning_rate_Index in range(0, NUM_LEARNING_RATES):
                print("learning_rate_Index", learning_rate_Index)
                thread_id = 0 + NUM_FS * NUM_NEURONS * learning_rate_Index
                for j in range(0, NUM_FS):
                    for k in range(0, NUM_NEURONS):
                        # print("k ,thread_id", k, thread_id)
                        # print("thread_id", thread_id)
                        feature_space_blockID = j
                        phase3_eval = ctypes.CDLL("/home/asim/Function Approximation Serial CPU/kernel_codes/phase3_eval.so")
                        print("fail is here")
                        phase3_eval.phase3_neuron_eval.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                                      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                                      np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
                                                                      np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
                                                                      ctypes.c_int, ctypes.c_int, ctypes.c_int,ctypes.c_int]

            # Define the sizes of the blocks and grids

                        phase3_eval.phase3_neuron_eval(
                            np.int32(NUM_NEURONS),
                            np.int32(NUM_NETS),
                            np.int32(test_data_1.shape[0]),
                            np.int32(NUM_FS),
                            np.int32(NUM_FEATURES),
                            np.int32(total_num_features_all_featurespaces),
                            np.int32(no_passes_phase3),
                            list_num_features_featurespaces,
                            cumulative_no_features_featurespace,
                            neurons_per_nets,
                            base_map,
                            active_centers,
                            test_data,
                            distance_base_map,
                            weights_array,
                            f_array,
                            radius_map,
                            learning_rate_list,
                            NET_SIZES,
                            np.int32(1) )


                        thread_id = thread_id + 1

            weights.append(weights_array)

            test_preds += f_array*lambda_phase3

            test_predictions.append(f_array)





        trainRMSE = np.zeros((len_learning_rates, NUM_FS), dtype=np.float32)
        CVRMSE = np.zeros((len_learning_rates, NUM_FS), dtype=np.float32)
        train_err_list = np.zeros((len_learning_rates, NUM_FS , int(0.7 * DATA_SIZE)), dtype=np.float32)

        CV_err_list = np.zeros((len_learning_rates, NUM_FS, DATA_SIZE - int(0.7 * DATA_SIZE)), dtype=np.float32)

        tar = all_fs_targets + avg_target
        test_tar = all_fs_test_targets + avg_target
        # print("tar")
        # print(all_fs_targets + avg_target)
        sum1 = 0.0
        sum2 = 0.0
        for i in range(len_learning_rates):
            for j in range(NUM_FS):
                for k in range(DATA_SIZE):
                    if k < int(0.7 * DATA_SIZE):
                        sum1 = abs(tar[i][j][k] - main_preds[i][j][k]) + sum1
                        train_err_list[i][j][k] = abs(tar[i][j][k] - main_preds[i][j][k])
                    else:
                        sum2 = abs(tar[i][j][k] - main_preds[i][j][k]) + sum2
                        CV_err_list[i][j][k - int(0.7 * DATA_SIZE)] = abs(tar[i][j][k] - main_preds[i][j][k])
                trainRMSE[i][j] = sum1
                CVRMSE[i][j] = sum2
                sum1 = 0.0
                sum2 = 0.0
        train_rmse_err = np.zeros((len_learning_rates, NUM_FS), dtype=np.float32)
        CV_rmse_err = np.zeros((len_learning_rates, NUM_FS), dtype=np.float32)

        for i in range(len_learning_rates):
            for j in range(NUM_FS):
                train_rmse_err[i][j] = np.sqrt(np.mean(np.square(train_err_list[i][j])))
        for i in range(len_learning_rates):
            for j in range(NUM_FS):
                CV_rmse_err[i][j] = np.sqrt(np.mean(np.square(CV_err_list[i][j])))
        # print("train_err_list")
        # print(train_err_list)
        # print("CV_err_list")
        # print(CV_err_list)
        # print("train_rmse_err")
        # print(train_rmse_err)
        # print("CV_rmse_err")
        # print(CV_rmse_err)
        lr_train = 0.0
        fs_train = 0
        lr_train_index = 0
        fs_train_index = 0
        max = train_rmse_err[0][0]
        min = train_rmse_err[0][0]
        for i in range(len_learning_rates):
            for j in range(NUM_FS):
                if train_rmse_err[i][j] > max:
                    max = train_rmse_err[i][j]

                if train_rmse_err[i][j] < min:
                    min = train_rmse_err[i][j]
                    lr_train = learning_rate_list[i]
                    lr_train_index = i
                    fs_train = list_num_features_featurespaces[j]
                    fs_train_index = j

        train_rmse = min
        # print("train RMSE")
        # print(min)
        # print(list_num_features_featurespaces)
        # print("lr_train")
        # print(lr_train)
        # print("lr_train_index")
        # print(lr_train_index)
        # print("fs_train")
        # print(fs_train)
        # print("fs_train_index")
        # print(fs_train_index)

        max1 = CV_rmse_err[0][0]
        min1 = CV_rmse_err[0][0]
        for i in range(len_learning_rates):
            for j in range(NUM_FS):
                if CV_rmse_err[i][j] > max1:
                    max1 = CV_rmse_err[i][j]
                if CV_rmse_err[i][j] < min1:
                    min1 = CV_rmse_err[i][j]
        # print(min)



        test_err_list = np.zeros((len_learning_rates, NUM_FS, test_targets.shape[0]), dtype=np.float32)
        sum1 = 0.0


        for i in range(len_learning_rates):
            for j in range(NUM_FS):
                for k in range(test_targets.shape[0]):

                        sum1 = abs(tar[i][j][k] - test_preds[i][j][k]) + sum1
                        test_err_list[i][j][k] = abs(test_tar[i][j][k] - test_preds[i][j][k])


                sum1 = 0.0
        test_rmse_err = np.zeros((len_learning_rates, NUM_FS), dtype=np.float32)

        for i in range(len_learning_rates):
            for j in range(NUM_FS):
                test_rmse_err[i][j] = np.sqrt(np.mean(np.square(train_err_list[i][j])))

        max = test_rmse_err[0][0]
        min = test_rmse_err[0][0]
        lr_test = 0.0
        lr_test_index = 0
        fs_test = 0.0
        fs_test_index = 0
        for i in range(len_learning_rates):
            for j in range(NUM_FS):
                if test_rmse_err[i][j] > max:
                    max = test_rmse_err[i][j]

                if test_rmse_err[i][j] < min:
                    min = test_rmse_err[i][j]
                    lr_test = learning_rate_list[i]
                    lr_test_index = i
                    fs_test = list_num_features_featurespaces[j]
                    fs_test_index = j

        test_rmse = min
        # print("test RMSE")
        # print(min)
        # print("lr_test")
        # print(lr_test)
        # print("lr_test_index")
        # print(lr_test_index)
        # print("fs_test")
        # print(fs_test)
        # print("fs_test_index")
        # print(fs_test_index)



        # print(main_preds[lr_train_index][fs_train_index])
        # print(test_preds[lr_test_index][fs_test_index])
        # np.savetxt('Best_predictions.csv', main_preds[lr_train_index][fs_train_index], delimiter=',', fmt='%.9f')

        best_main_prediction = main_preds[lr_train_index][fs_train_index]
        best_test_prediction = test_preds[lr_test_index][fs_test_index]
        best_feature_train =  fs_train
        best_feature_test = fs_test
        best_learning_rate_train = lr_train
        best_learning_rate_test = lr_test
        #
        #
        # print(best_main_prediction)
        # print(best_test_prediction)
        # print(best_feature_train)
        # print(best_feature_test)
        # print(best_learning_rate_train)
        # print(best_learning_rate_test)
        # print(train_rmse)
        # print(test_rmse)



        # print("best_stack_eval_rmse")
        # print(best_stack_eval_rmse)


    return best_main_prediction, best_test_prediction, best_feature_train, best_feature_test, best_learning_rate_train, best_learning_rate_test, train_rmse, test_rmse
