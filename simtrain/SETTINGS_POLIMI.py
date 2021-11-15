import numpy as np
from os.path import join


# change this line if there are enough resources to run with full userset:
N_SUBSAMPLE_USERS = 5000
NUMEXPR_MAX_THREADS = '32' # enter your number of cores here

# general settings related to dataset (will be overwritten when a dataset is loaded)
stg = {
    'NI': 10,  # num items
    'NU': 100,  # num users
    'T': 100,  # duration of simulation (in units of days)
    'NS': 100,  # num simulations
    'INF_TIME': 1000,  # how is infinity time defined (unit of days)
    'lambda_impression': 18.6,  # avg. num impressions per visit
}

INFTY = int(1e6)  # definition of infinity
EPS = 1e-9  # definition of infinitesimally small

# hyperparameters, intended to fit with cross-validation
hyp = {
    'seed': 0,  # random seed
    'cores': 48,  # 64 # number of cores processing

    # inhomogeneous Poisson process sampling
    'max_lam': 0.25, #2.0, #9.0,  # guess of the max value of intensity function (useful for rejection sampling). if error is raised then this value needs to be increased
    'block_size': 0.5,  # in units of days
    'max_lam_factor': 2.0, #2.0,  # multiply empirical num. user visits by this factor to derive a guess of max lambda

    # proportion of impressions that are recommender driven (by convention, the first index refers to recommenders)
    'rec_prop': np.array([1.0]), #np.array([0.8, 0.2]),
    'user_select_count': 1000, # how many potential actions does user have? (i.e. num items they would stream iff shown)

    # hyperparameters for models and training models
    'rec_model_hyp': {'dropout_rate': 0.15,  # 0.15
                      'n_hidden_layers': 2, # 1
                      'n_nodes': [1000,500],  # was [500]
                      'hidden_activation': 'relu',
                      'max_epoch': 10,
                      'min_delta': 0.01,  # for test, previously used 0.001,
                      'batch_size':2048, # was 256 'batch_size': 64,
                      'train_ips': False,
                      'time_slots_per_day': 1.0,  # number of time buckets per day
                      'popularity_alpha':0.0 # exponential weighting for removing popularity
                      },

    'user_model_hyp': {'dropout_rate': 0.15,  # 0.15
                       'n_hidden_layers': 1, # 1
                       'n_nodes': [500], # was [200]
                       'hidden_activation': 'relu',
                       'max_epoch': 10,
                       'min_delta': 0.001,
                       'batch_size': 2048, # was 64
                       'train_ips': False},  # whether or not to use ips during training
    
    'hyp_study': { 'init_args':{'n_components':10,
                      'init':'random',
                      'random_state':0,
                               },
                   'temperature':1.0,
                  'temperature_sweep': [1, 0.5, 0.25, 0.125, 0.0625],
                  'constant_rate':True
    },

    'visit_model_hyp': {
        # hyperparameters for calculating regularizer of Poisson process:
        'visit_loss_method': 'event_sample', # other option 'event_score', 'event_sample', #'full', #'point_sample' # 'event_sample',
        # 'event_sample', # how to estimate the regularizer in the Poisson visit model [full | event_sample | point_sample]
        'lambda_grid_factor': 0.01, #0.005,
        'n_samples_per_user': 100,
        # [only used with full method] what proportion of the time range to sample the regularizer at
        'user_point_sample_factor': 1.0, # 0.1
        # [only used with point_sample method] what proportion of users to sample the regularizer at?
        'time_sample_size': 100,  # [only used with point_sample method] how many time events to sample?
        'parallel_process_features': True, 
        
        # for simulation, use a mixture of pure self-exciting visitation with total intensity to reduce variance of prediction:
        'self_exciting_mixture': 0.8, 

        # DEPRECATED: how to sample next visit time:
        'visit_sample_grid_size': 0.5,  # units of 50% of a day ~= 12 hours
        
        # log step-wise granular stats on the objective:
        'fit_log_granular': False, 
        'log_history_path': 'dat/log/model_history_log.csv',

        'window_size': 5,  # number of events to look back on for Hawkes intensity
        'batch_size': 128, # 64,
        'max_epoch': 100, #10, #100, #20, #10, #100
        'patience': 0, #100,
        'min_delta': 1e-4, #0.001,
        'train_val_split': 1.0, 
        'time_model': {'n_hidden_layers': 1,
                       'n_nodes': [100], #[300,100], #[300,100], #[100],
                       'hidden_activation': 'relu',
                       'dropout_rate': 0.5 #0.15
                      },
        'state_model': {'n_hidden_layers': 1,
                        'n_nodes': [20], #[500],  # 100
                        'hidden_activation': 'relu',
                        'dropout_rate': 0.5,
                        'dropout_positives': True},
    },

    # hyperparameters for precompute optimization
    'precompute': {
        'cadence_days': 1.0,  # time interval in days between each precomputation sweep
        'timeout_days': 3.0, #1.0, #10.0, #3.0,  # time interval in days between last precompute and switch to live compute upon visit
        'cutoff_normalized_rank': 0.2, #0.4, #0.8,  # what proportion of users make it to each day's precompute?
        'lookahead_factor': 1.0, # how many days (candence_days*lookahead_factor) to plan ahead?
        'model_sample_freq': 5, # number of samples to take from visit model before taking the first event as visit time
        'simulated_annealing_factor': 0.0, # in matroid precompute, how large simulated annealing factor?
        'reference_time': 0.5, # what part of the day to reference in loss when looking at intensity only?

        'loss_approximation': {
            'look_ahead_days': 3,  # number of days to look ahead when sampling users, allows ordering of non-visiting users
            'safety_factor': 3,  # by what factor multiplied by precompute time per user do we assume for deadline?
            'n_samples': 5,  # how many Monte Carlo samples per user to approximate loss?
            # upper bound on intensity function (N.B. have different max_lam's for approximating loss and simulating
            # users for practical purposes):
            'max_lam': 9.0,
            'sample_method': 'upper_range', # or marginal; method for converting visit probabilities to scores
            # penalty in days for scheduling precompute and having user visit first, higher values result in more attention paid
            # in getting ranking correct vs. deciding whether or not to precompute:
            'miss_penalty': 1.0,
            'worst_case_pc':-60
        }
    }

}

rootpaths = {
    'models': 'saved_models_polimi',
    'plots': 'fig_polimi',
    'input': 'input'
}

filepaths = {
    #'impressions_data': 'impr_featurized.csv',  # training data of real impressions
    'impressions_data_test': 'ContentWise-%s-subitems.csv.gz',  # training data of real impressions
    'user_model': join(rootpaths['models'], 'user_model.h5'),
    'user_model_ips': join(rootpaths['models'], 'user_model_ips.h5'),
    'user_model_dropout': join(rootpaths['models'], 'user_model_dropout.h5'),
    'user_model_test': join(rootpaths['models'], 'user_model_test-%s.h5'),
    'user_model_binary': join(rootpaths['models'], 'user_model_binary_test%i.h5'),
    'user_model_binary_ips': join(rootpaths['models'], 'user_model_binary_ips_test%i.h5'),
    'rec_model': join(rootpaths['models'], 'rec_model.h5'),
    'rec_model_t': join(rootpaths['models'], 'rec_model_t.h5'),
    'rec_model_t_test': join(rootpaths['models'], 'rec_model_t_test-%s_cell%i_rec%i.h5'),
    'visit_model': join(rootpaths['models'], 'visit1_%s.h5'),  # path stem indicating how to find multiple files
    'visit_model_test0': join(rootpaths['models'], 'visit1_%s_test0.h5'),  # path stem indicating how to find multiple files
    'visit_model_test-train': join(rootpaths['models'], 'visit1_%s_test0.h5'),  # path stem indicating how to find multiple files
}

simulation_components = {
    'ab_test_id': 'train',
    'rec_model_cell_nbr': [1],  # data used to train rec_models with different rec_id (0 indicates production data)
    'user_model_cell_nbr': 1,
    'visit_model_cell_nbr': 1
}

validation_settings = {
    'visit_window_sizes': [1, 3, 7],  # size of windows in days to test visit models
}
