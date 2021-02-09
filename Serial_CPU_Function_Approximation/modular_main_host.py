# Import Statements
import os
import ast
import time
import json
import itertools
import numpy as np
import pandas as pd
from distance_encode_new import distance_encode
from collections import Counter
from phase_wise_functions import phase1, phase2, phase3, commenter


class MainApp:
    '''
    
    Class to run the main app. The class is created by taking 4 arguments
    
    Arguments:
    phase1_params - a dictionary that has the parameters used to control phase1
    phase2_params - a dictionary that has the parameters used to control phase2
    phase3_params - a dictionary that has the parameters used to control phase3
    files_to_run - a text file that has the names of datasets to run for

    '''

    def __init__(self, phase1_params, phase2_params, phase3_params, files_to_run):
        self.phase1_params = phase1_params
        self.phase2_params = phase2_params
        self.phase3_params = phase3_params
        self.fp = open(files_to_run, "r")

    def commenter(self, s):
        print("-" * 100)
        print(s)

    def call_phase1(self):
        '''
        The function is a higher level call for Phase1 execution. The function preapres the list of variables that we calculate for 
        each chunk of dataset. These variables include, NET_SIZES, LEARNING_RATES, NUM_SHUFFLES, NUM_NEURONS etc.

        Arguments:
        train_data - the training dataframe thats loaded
        '''

        self.commenter("Phase1")
        features = self.Input_TrainData.columns

        ''' ==================================================================================================
        Parameters Declaration
        -------------------------------------------------------------------------------------------------- '''
        # Number of features in dataset
        NUM_FEATURES = self.Input_TrainData.shape[1]
        NUM_FS = int(np.ceil(NUM_FEATURES / self.phase1_params['partition_size']))  # Number of Feature Spaces
        NUM_SHUFFLES = NUM_FS
        NUM_NEURONS = int(np.sum([i * i for i in self.phase1_params['net_sizes']]))  # Number of Neurons
        NUM_NETS = len(self.phase1_params['net_sizes'])  # Number of nets calculate here
        len_learning_rates = len(self.phase1_params['lr'])  # Number of Learning rates
        list_num_features_featurespaces_start, list_num_features_featurespaces = [], []
        for i in range(0, NUM_FEATURES, self.phase1_params['partition_size']):
            list_num_features_featurespaces_start.append(i)
            if i + self.phase1_params['partition_size'] < NUM_FEATURES:
                list_num_features_featurespaces.append(i + self.phase1_params['partition_size'])
            else:
                list_num_features_featurespaces.append(NUM_FEATURES)

        cumulative_no_features_featurespace = []
        cum = 0
        cumulative_no_features_featurespace.append(0)
        for v in range(NUM_FS):
            cum += list_num_features_featurespaces[v]
            cumulative_no_features_featurespace.append(cum)
        list_num_features_featurespaces = np.array(list_num_features_featurespaces, dtype=np.int32)
        list_num_features_featurespaces_start = np.array(list_num_features_featurespaces_start, dtype=np.int32)
        total_num_features_all_featurespaces = sum(list_num_features_featurespaces)

        cumulative_no_features_featurespace = np.array(cumulative_no_features_featurespace, dtype=np.int32)

        learning_rate_list = np.array(self.phase1_params['lr'], dtype=np.float32)

        '''===================================================================================================='''

        overall_dstances_all = []
        feat_order = list(range(0, self.Input_TrainData.shape[1], 1))
        start_time = time.time()

        for it in range(NUM_SHUFFLES):
            # Shuffling the data for repeated iterations
            np.random.shuffle(feat_order)

            train_data = self.Input_TrainData.iloc[:, feat_order]
            #print(train_data)
            feat_order_all_blocks = np.array(
                [np.argsort(feat_order) + NUM_FEATURES * lr for lr in range(len_learning_rates)],
                dtype=np.int32).ravel()
            # Calling Phase1 for every trial
            temp = phase1(train_data, min(self.phase1_params['chunk_size'], self.Input_TrainData.shape[0]),
                          NUM_FEATURES, NUM_FS, NUM_NEURONS,
                          NUM_NETS, learning_rate_list, len_learning_rates, list_num_features_featurespaces_start,
                          list_num_features_featurespaces,
                          total_num_features_all_featurespaces, cumulative_no_features_featurespace,
                          self.phase1_params['net_sizes'],
                          self.phase1_params['no_chunk_passes'], self.phase1_params['no_data_passes'],
                          self.phase1_params['neigh_rate'])
            overall_dstances_all.append(temp[feat_order_all_blocks])

        '''Finding feature ranking'''
        final_overall_distances = np.mean(np.array(overall_dstances_all, dtype=np.float32), axis=0)
        # print(final_overall_distances)
        new_distances = np.zeros((NUM_FEATURES), dtype=np.float32)
        for k in range(len_learning_rates):
            new_distances += final_overall_distances[k * NUM_FEATURES:(k + 1) * NUM_FEATURES]
        f_ranking = np.argsort(new_distances)[::-1]
        self.ranks = f_ranking.tolist()

        commenter("Finding Ranks")
        self.ranked_features = features[self.ranks].tolist()
        print(self.ranked_features[:50])

        end_time = time.time()
        print("Phase1 total time {}".format(end_time - start_time))

        self.phase1_time = end_time - start_time

    def call_phase2(self):

        self.commenter("Phase2")
        ranks_columns = self.Input_columns[self.ranks]
        # print("len(self.ranks)")
        print(self.Input_columns[self.ranks])

        new_data = self.Input_TrainData[ranks_columns].iloc[:, :self.phase2_params['top_ranks']]       #self.phase2_params['top_ranks']
        # Parameters Declaration
        NUM_FEATURES = new_data.shape[1]  # Number of features in dataset
        CHUNK_SIZE = new_data.shape[0]  # size of stream taken
        NUM_FS = self.phase2_params['top_ranks']       # Number of Feature Spaces
        NUM_NEURONS = int(np.sum([i * i for i in self.phase2_params['net_sizes']]))  # Number of Neurons
        NUM_NETS = len(self.phase2_params['net_sizes'])  # Number of nets calculate here
        len_learning_rates = len(self.phase2_params['lr'])  # Number of Learning rates
        list_num_features_featurespaces = np.array(list(range(1, self.phase2_params['top_ranks'] + 1, 1)),
                                                   dtype=np.int32)
        cumulative_no_features_featurespace = []
        cum = 0
        cumulative_no_features_featurespace.append(0)
        for v in range(NUM_FS):
            cum += list_num_features_featurespaces[v]
            cumulative_no_features_featurespace.append(cum)
        total_num_features_all_featurespaces = sum(list_num_features_featurespaces)
        cumulative_no_features_featurespace = np.array(cumulative_no_features_featurespace, dtype=np.int32)
        learning_rate_list = np.array(self.phase2_params['lr'], dtype=np.float32)
        # print(NUM_FEATURES)
        # print(CHUNK_SIZE)
        # print(NUM_FS)
        # print(NUM_NETS)
        # print(len_learning_rates)
        # print(list_num_features_featurespaces)
        # print(total_num_features_all_featurespaces)
        # print("cumulative_no_features_featurespace")
        # print(cumulative_no_features_featurespace)
        # print(learning_rate_list)

        # Measuring time
        start_time = time.time()

        radius_map, base_map, active_centers, distance_map = phase2(new_data, CHUNK_SIZE, NUM_FEATURES, NUM_FS,
                                                                    NUM_NEURONS,
                                                                    NUM_NETS, learning_rate_list, len_learning_rates,
                                                                    list_num_features_featurespaces,
                                                                    total_num_features_all_featurespaces,
                                                                    cumulative_no_features_featurespace,
                                                                    self.phase2_params['net_sizes'],
                                                                    self.phase2_params['no_passes'],
                                                                    self.phase1_params['neigh_rate'])

        o_active_centers = active_centers
        active_centers = active_centers.ravel()

        # print("radius map")
        # print(radius_map)
        # print("base_map")
        # print(base_map)

        # commenter("Finding raduis map for each neuron")
        # print("radius map centers")
        # print(radius_map)
        # print("active centers")
        # print(active_centers)

        # Choosing FS based on number of actiavted neurons in each
        fs_selected = []
        for j in range(len_learning_rates):
            l = []
            for i in range(j * self.phase2_params['top_ranks'] * NUM_NEURONS,
                           (j + 1) * self.phase2_params['top_ranks'] * NUM_NEURONS, NUM_NEURONS):
                l.append(np.sum(active_centers[i:i + NUM_NEURONS] != 0))
            fs_selected.append(l)
        fs_sum = np.sum(np.array(fs_selected), axis=0)
        # print("fs_sum")
        # print(fs_sum)

        self.selected = np.argsort(fs_sum)[::-1][:self.phase2_params['select_fs_cnt']]

        # print("active_centers p[2]")
        # print(active_centers)
        # print("self.selected")
        # print(self.selected)

        # selected = list(range(top_ranks))
        # for l in range(len_learning_rates):
        #
        #     print(o_active_centers[l, self.selected[0], :])



        new_base_map, new_radius_map, new_active_centers, new_distance_map = [], [], [], []
        sum_fs_chosen = 0
        for fs in self.selected:
            sum_fs_chosen += fs + 1
            for lr in range(len_learning_rates):
                map_index = lr * total_num_features_all_featurespaces * NUM_NEURONS + NUM_NEURONS * \
                            cumulative_no_features_featurespace[fs]
                neuron_index = lr * NUM_FS * NUM_NEURONS + NUM_NEURONS * fs
                new_base_map.extend(
                    base_map[map_index:map_index + NUM_NEURONS * list_num_features_featurespaces[fs]].tolist())
                new_radius_map.extend(radius_map[neuron_index:neuron_index + NUM_NEURONS].tolist())
                new_active_centers.extend(active_centers[neuron_index:neuron_index + NUM_NEURONS].tolist())
                new_distance_map.extend(distance_map[neuron_index:neuron_index + NUM_NEURONS].tolist())
        # print("active_centers n[2]")
        # print(new_active_centers)
        self.new_base_map = np.array(new_base_map, dtype=np.float32).reshape(
            (sum_fs_chosen, NUM_NEURONS * len_learning_rates))
        self.new_radius_map = np.array(new_radius_map, dtype=np.float32).reshape(
            (len(self.selected), NUM_NEURONS * len_learning_rates))
        self.new_active_centers = np.array(new_active_centers, dtype=np.int32).reshape(
            (len(self.selected), NUM_NEURONS * len_learning_rates))
        self.new_distance_map = np.array(new_distance_map, dtype=np.float32).reshape(
            (len(self.selected), NUM_NEURONS * len_learning_rates))
        
        end_time = time.time()
        print("Phase2 total time {}".format(end_time - start_time))

        self.phase2_time = end_time - start_time

    def call_phase3(self):

        self.commenter("Phase3")
        prev_lrs = len(self.phase2_params['lr'])
        ranks_columns = self.Input_columns[self.ranks]
        new_train_data = self.Input_TrainData[ranks_columns].iloc[:, :self.phase2_params['top_ranks']]
        new_test_data = self.Input_TestData[ranks_columns].iloc[:, :self.phase2_params['top_ranks']]

        # Parameters Declaration
        NUM_FEATURES = new_train_data.shape[1]  # Number of features in dataset
        CHUNK_SIZE = new_train_data.shape[0]  # size of stream taken
        NUM_FS = len(self.selected)  # Number of Feature Spaces
        NET_SIZES = list(itertools.chain(*[self.phase3_params['net_sizes']] * prev_lrs))
        NUM_NEURONS = int(np.sum([i * i for i in self.phase3_params['net_sizes']]))  # Number of Neurons
        NUM_NETS = len(self.phase3_params['net_sizes'])  # Number of nets calculate here
        len_learning_rates = len(self.phase3_params['lr'])  # Number of Learning rates
        list_num_features_featurespaces = [i + 1 for i in self.selected]
        cumulative_no_features_featurespace = []
        cum = 0
        cumulative_no_features_featurespace.append(0)
        for v in range(NUM_FS):
            cum += list_num_features_featurespaces[v]
            cumulative_no_features_featurespace.append(cum)
        list_num_features_featurespaces = np.array(list_num_features_featurespaces, dtype=np.int32)
        total_num_features_all_featurespaces = sum(list_num_features_featurespaces)
        cumulative_no_features_featurespace = np.array(cumulative_no_features_featurespace, dtype=np.int32)
        # print(list_num_features_featurespaces, cumulative_no_features_featurespace, total_num_features_all_featurespaces)
        learning_rate_list = np.array(self.phase3_params['lr'], dtype=np.float32)
        base_map = np.array([self.new_base_map] * len_learning_rates)
        radius_map = np.array([self.new_radius_map] * len_learning_rates)
        active_centers = np.array([self.new_active_centers] * len_learning_rates)
        main_pred_list = []
        target_list = []
        difference_list = []

        # print(base_map)

        # Measuring time
        start_time = time.time()
        self.best_main_prediction, self.best_test_prediction, self.best_feature_train, self.best_feature_test, self.best_learning_rate_train, self.best_learning_rate_test, self.train_rmse, self.test_rmse = phase3(main_pred_list, target_list, difference_list, new_train_data, new_test_data, self.Input_Test_Target,
                                                                CHUNK_SIZE, self.Input_Target, self.train_norm_scale,
                                                                NUM_FEATURES, NUM_FS, NUM_NEURONS, NUM_NETS, prev_lrs,
                                                                self.phase3_params['train_split'],
                                                                learning_rate_list, len_learning_rates,
                                                                list_num_features_featurespaces,
                                                                total_num_features_all_featurespaces,
                                                                cumulative_no_features_featurespace, NET_SIZES,
                                                                base_map, radius_map, active_centers,
                                                                self.phase3_params['boost_trials'],
                                                                self.phase3_params['no_passes'],
                                                                self.phase3_params['convergence_threshold'],
                                                                self.phase3_params['lambda'],
                                                                self.phase3_params['stack_passes'])
        end_time = time.time()
        print("Phase3 total time {}".format(end_time - start_time))

        self.phase3_time = end_time - start_time

    def load_data_and_targets(self, path_to_train_file, path_to_test_file):
        # Reading the files
        TrainData = pd.read_csv(path_to_train_file)
        TestData = pd.read_csv(path_to_test_file)

        from sklearn.utils import shuffle
        TrainData = shuffle(TrainData)
        TestData = shuffle(TestData)

        # Train and Test
        Input_TrainData = TrainData.iloc[:, :-1]
        # print(Input_TrainData)
        Input_TestData = TestData.iloc[:, :-1]
        Input_Target = TrainData.iloc[:, -1].values.reshape(-1, 1)
        Input_Test_Target = TestData.iloc[:, -1].values.reshape(-1, 1)
        # print(Input_Target)
        Target = TrainData.columns[-1]

        # Combining Test and Target data for consistent distributed encoding
        Combine_Train_Test_DistributedEncoding = pd.concat([Input_TrainData, Input_TestData], keys=[0, 1], sort=False)

        # Distributed Encoding for the Train Data
        Combine_Train_Test_DistributedEncoding = distance_encode(Combine_Train_Test_DistributedEncoding)  # copying Back the Data

        # Divideback into Train and test
        Input_TrainData = Combine_Train_Test_DistributedEncoding.xs(0)
        Input_TestData = Combine_Train_Test_DistributedEncoding.xs(1)
        self.Input_columns = Input_TrainData.columns  # Read the column names


        # Normalizing both Train and Test data
        from sklearn.preprocessing import MinMaxScaler
        self.train_norm_scale = MinMaxScaler()
        Input_TrainData = self.train_norm_scale.fit_transform(Input_TrainData)
        self.Input_TrainData = pd.DataFrame(Input_TrainData, columns=self.Input_columns)
        Input_TestData = self.train_norm_scale.transform(Input_TestData)
        self.Input_TestData = pd.DataFrame(Input_TestData, columns=self.Input_columns)

        # Normalizing Targets
        self.Input_Target = self.train_norm_scale.fit_transform(Input_Target)
        self.Input_Test_Target = self.train_norm_scale.transform(Input_Test_Target)

        print( self.Input_Target)

    def get_top_k(self, data, k):
        d = data.iloc[:, 9:k + 9]
        counter = Counter(d.values.ravel().tolist())
        threshold = np.floor(np.mean(list(counter.values())))
        topped = [key for key, v in counter.items()][:k]
        threshold = np.mean([v for v in counter.values()][:k]) / 10
        return topped, threshold

    def main_results(self, sets=[-1, -2, -3]):
        t = [list(range(k * 10, k * 10 + 10)) for k in sets]
        res_columns = ['Dataset_Name', 'Net Size', 'LR', 'Neigh_Rate', 'Time', 'Total Num Features', 'Top k Features',
                       'Overlap Measure']
        res = []
        for dataset_name in self.fp:
            dataset_name = dataset_name.rstrip('\n')
            path = "Results/" + dataset_name + "/FEATURE_RANKING_RESULTS.csv"
            df = pd.read_csv(path)
            for ind in t:
                data = df.iloc[ind, :]
                topped, threshold = self.get_top_k(data, 10)
                res.append((dataset_name, ast.literal_eval(data.values[0, 3])[0], data.values[0, 5], data.values[0, 7], data.values[0, 8],
                            data.values[0, 6], topped, threshold))
        
        pd.DataFrame(res, columns=res_columns).to_csv("Results/Main_Results.csv", sep=',', index=None)

    def table_results(self):
        df = pd.read_csv("Results/Main_Results.csv")
        all_mat = {}
        for i in range(0, len(df), 3):
            comb = itertools.combinations(list(range(i, i + 3)), 2)
            mat = np.ones((3, 3), dtype=np.float32)
            for k in comb:
                l1, l2 = set(ast.literal_eval(df.iloc[k[0], -2])), set(ast.literal_eval(df.iloc[k[1], -2]))
                mat[k[0] % 3, k[1] % 3] = len(l1.intersection(l2)) / len(l1)
                mat[k[1] % 3, k[0] % 3] = len(l1.intersection(l2)) / len(l2)
            all_mat[str(df.values[i, 0])+str(i)] = {}
            all_mat[str(df.values[i, 0])+str(i)]['lr'] = df.values[i, 2]
            all_mat[str(df.values[i, 0])+str(i)]['nr'] = df.values[i, 3]
            all_mat[str(df.values[i, 0])+str(i)]['time'] = df.values[i, 4]
            all_mat[str(df.values[i, 0])+str(i)]['mat'] = mat.tolist()
        json.dump(all_mat, open("Main_Results/all_matrices.json", "w"))

    def read_final_results(self):
        all_mat = json.load(open("Main_Results/all_matrices.json", "r"))
        for dataset_name in all_mat:
            print(dataset_name)
            print(np.array(all_mat[dataset_name]['mat']))

    def contain_results(self):
        res_columns = ['Dataset_Name', 'Set_no', 'Phase1-Time', 'Phase2-Time', 'Phase3-Time', 'Best Train RMSE', 'Best Val RMSE', 'Best Stack Train RMSE',
                       'Best Stack Val RMSE', 'Best Test RMSE', 'Best Stack Test RMSE']
        all_res = []
        for dataset_name in self.fp:
            dataset_name = dataset_name.rstrip('\n')
            path = "Results/" + dataset_name + "/MAIN_RESULTS.csv"
            df = pd.read_csv(path)
            if len(df) >= 10 and len(df) % 10 == 0:
                best_result, best_rmse = None, 1000000
                for k in df.columns[[-7, -6, -4, -3, -2, -1]]:
                    df[k] = df[k].apply(lambda x: ast.literal_eval(x)[0])

                for i in range(0, len(df), 10):
                    set_no = (i / 10) + 1
                    avg_res = df.iloc[i:i + 10, [-24,-18,-9,-7, -6, -4, -3, -2, -1]].mean().values
                    if avg_res[-2] < best_rmse:
                        best_result = [dataset_name] + [set_no] + avg_res.tolist()
                        best_rmse = avg_res[-2]
                    if avg_res[-1] < best_rmse:
                        best_result = [dataset_name] + [set_no] + avg_res.tolist()
                        best_rmse = avg_res[-1]
                all_res.append(tuple(best_result))
        pd.DataFrame(all_res, columns=res_columns).to_csv("Main_Results/all_combined.csv", index=None, sep=',')





    # def collect_results(self, seed, dataset_name, path_to_file, main_res=False):
    #     if not main_res:
    #         FEATURE_RANKING_COLUMNS = ['Seed', 'Phase1_CHUNK_SIZE', 'Phase1_NUM_DATASET_PASSES', 'Phase1_NET_SIZES',
    #                                    'Phase1_SIZE_OF_PARTITIONS',
    #                                    'Phase1_learning_rates', 'Total Features', 'Neigh_Rate', 'Time'] + ['F' + str(i + 1) for
    #                                                                                                i in range(
    #                 min(len(self.ranks), 50))]
    #
    #         if not os.path.isfile("Results/" + dataset_name + "/FEATURE_RANKING_RESULTS.csv"):
    #             pd.DataFrame([], columns=FEATURE_RANKING_COLUMNS).to_csv(
    #                 "Results/" + path_to_file.split("/")[1] + "/FEATURE_RANKING_RESULTS.csv", sep=',', index=None)
    #
    #             FEATURE_RANKING_RESULTS = np.array(
    #                 [seed, self.phase1_params['chunk_size'], self.phase1_params['no_data_passes'],
    #                 self.phase1_params['net_sizes'], \
    #                 self.phase1_params['partition_size'], self.phase1_params['lr'], len(self.Input_columns),
    #                 self.phase1_params['neigh_rate'], self.phase1_time] + self.ranked_features[:50]).reshape(-1, 1).T
    #             FEATURE_RANKING_RESULTS = pd.DataFrame(FEATURE_RANKING_RESULTS, columns=FEATURE_RANKING_COLUMNS)
    #             FEATURE_RANKING_RESULTS.to_csv("Results/" + path_to_file.split("/")[1] + "/FEATURE_RANKING_RESULTS.csv",
    #                                         sep=',', mode='a', header=None, index=None)
    #
    #     else:
    #         MAIN_RESULTS_COLUMNS = ['Seed', 'Phase1_CHUNK_SIZE', 'Phase1_NUM_DATASET_PASSES', 'Phase1_NET_SIZES',
    #                                 'Phase1_SIZE_OF_PARTITIONS',
    #                                 'Phase1_learning_rates', 'Neigh_Rate', 'Phase 1 Time', 'Feature_Ranking', 'Top_ranks',
    #                                 'Phase2_No_Passes', 'Phase2_NET_SIZES',
    #                                 'Phase2_learning_rates', 'Phase 2 Time', 'Selected_Feature_Spaces', 'Phase3_No_Passes',
    #                                 'Phase3_NET_SIZES',
    #                                 'Phase3_learning_rates', 'Train_Split', 'Phase3_No_Boosting_Trials',
    #                                 'Phase3_Convergence_Threshold',
    #                                 'Phase3_Lambda', 'Phase 3 Time', 'Best LR,FS Combination', 'Best_Train_RMSE',
    #                                 'Best_Validation_RMSE', 'Best Stacked LR,FS Combination',
    #                                 'Stacked_Best_Train_RMSE', 'Stacked_Best_Validation_RMSE', 'Test RMSE',
    #                                 'Stacked Test RMSE']
    #         if not os.path.isfile("Results/" + dataset_name + "/MAIN_RESULTS.csv"):
    #             pd.DataFrame([], columns=MAIN_RESULTS_COLUMNS).to_csv("Results/" + dataset_name + "/MAIN_RESULTS.csv",
    #                                                                   sep=',', index=None)
    #
    #         MAIN_RESULTS = np.array([seed, self.phase1_params['chunk_size'], self.phase1_params['no_data_passes'],
    #                                  self.phase1_params['net_sizes'],
    #                                  self.phase1_params['partition_size'], self.phase1_params['lr'],
    #                                  self.phase1_params['neigh_rate'], self.phase1_time, self.ranks[:50], self.phase2_params['top_ranks'],
    #                                  self.phase2_params['no_passes'],
    #                                  self.phase2_params['net_sizes'], self.phase2_params['lr'], self.phase2_time, self.selected,
    #                                  self.phase3_params['no_passes'], self.phase3_params['net_sizes'],
    #                                  self.phase3_params['lr'], self.phase3_params['train_split'],
    #                                  self.phase3_params['boost_trials'],
    #                                  self.phase3_params['convergence_threshold'], self.phase3_params['lambda'], self.phase3_time,
    #                                  self.best_fl_indices, self.best_train_rmse, self.best_val_rmse,
    #                                  self.best_stack_fl_indices, self.best_stack_train_rmse, self.best_stack_val_rmse,
    #                                  self.best_eval_rmse, self.best_stack_eval_rmse]).reshape(-1, 1).T
    #         MAIN_RESULTS = pd.DataFrame(MAIN_RESULTS, columns=MAIN_RESULTS_COLUMNS)
    #         MAIN_RESULTS.to_csv("Results/" + dataset_name + "/MAIN_RESULTS.csv", sep=',', mode='a', header=None,
    #                             index=None)

    def dataset_loop(self):
        for dataset_name in self.fp:
            dataset_name = dataset_name.rstrip('\n')

            if not os.path.isdir("Results"):
                os.mkdir("Results")
            if not os.path.isdir("Results/" + dataset_name):
                os.mkdir("Results/" + dataset_name)
            if not os.path.isdir("Main_Results"):
                os.mkdir("Main_Results")
        
            seed = 0
            print("datasets",dataset_name)
            dataset_name = "ECG"
            all_files = os.listdir(os.path.join("Datasets", dataset_name, "Train"))
            all_files = [i for i in all_files if dataset_name in i]
            for files in all_files:
                dir = os.path.join("Datasets", dataset_name)
                path_to_file = os.path.join(dir, "Train", files)
                path_to_test_file = os.path.join(dir, "Test", files.replace("Train", "Test"))
                print(path_to_file)
                print(path_to_test_file)
        
                ### Ladaing the data with out without normalization
                self.load_data_and_targets(path_to_train_file=path_to_file, path_to_test_file=path_to_test_file)
        
                ### Calling Phase 1
                self.phase1_params['chunk_size'] = min(self.phase1_params['chunk_size'], self.Input_TrainData.shape[0])
                self.call_phase1()
                # self.collect_results(seed, dataset_name, path_to_file)
        
                ### Calling Phase 2
                self.phase2_params['top_ranks'] = min(50,len(self.ranks))
                self.call_phase2()
                print(self.selected)
        
                ### Calling Phase 3
                self.call_phase3()
                # self.collect_results(seed, dataset_name, path_to_file, True)
                seed += 50
