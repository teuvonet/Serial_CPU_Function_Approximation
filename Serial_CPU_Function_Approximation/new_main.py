from modular_main_host import MainApp
import time

if __name__ == "__main__":
    start = time.time()
    phase1_params = {
        'net_sizes': [5],
        'partition_size': 100,
        'chunk_size': 100,
        'lr': [0.5, 0.1],
        'no_chunk_passes': 1,
        'no_data_passes': 1,
        'neigh_rate': 0.8,
    }

    phase2_params = {
        'net_sizes': [5,6],
        'lr': [0.05,0.01],
        'no_passes': 1,
        'top_ranks': 1, # max number of features to consider for phase2
        'select_fs_cnt': 25, # max number of feature spaces to use for phase 3
    }

    phase3_params = {
        'net_sizes': phase2_params['net_sizes'],
        'lr': [0.001,0.003],
        'train_split': 0.7, # split between train and validation set
        'boost_trials': 2, # number of boosting models to build
        'no_passes': 1, # number of passes through the dataset for each boosting model
        'stack_passes': 2, # number of stacking passes to train the linear regression weights on top of boosting model results
        'lambda': [0.3], # boosting model lambda
        'convergence_threshold': 0.0001
    }

    files_to_run = 'collect_small.txt'

    app = MainApp(phase1_params,phase2_params,phase3_params,files_to_run)

    app.dataset_loop()
    app.contain_results()
    print("Total Time taken is %d" % (time.time() - start))
    # app.main_results(sets=[-1,-2,-3,-4,-5,-6])
    # app.table_results()
    # app.read_final_results()
