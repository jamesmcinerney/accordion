import os

import numpy as np
import pandas as pd
from scipy import sparse as sp
from typing import Callable, List, Tuple, Dict
from os.path import join

from . import utils, process_dat
from . import SETTINGS_POLIMI as SETTINGS

os.environ['NUMEXPR_MAX_THREADS'] = str(SETTINGS.hyp['cores'])
pd.options.mode.chained_assignment = None  # allow assigning to copy of DataFrame slice without warnings


def simulate_batch(initial_user_state: np.ndarray, models: tuple, hyp: dict, stg: dict,
                   tevmin: float, tevmax: float, 
                   tevmin_train: float, tevmax_train: float, tevmean_train: float,
                   verbose: int = 0, user_max_lam: np.ndarray = None, user_n_imp: np.ndarray = None,
                   fix_n_imp: bool = False) -> pd.DataFrame:
    """
        Runs simulation given models of members (consumption and visits) and recommenders (impressions).

        Idea is to sample "visit potential points" for each user based on an upper bound of intensity
        with homogeneous Poisson process, then use rejection sampling to decide the actual visit time points.

        @param initial_user_state: (n_users, n_items) csr_matrix start state for each member
        @param models: (UserModel, [RecModel], VisitModel) tuple
        @param hyp: hyperparameter dict
        @param stg: settings dict
        @param tevmin: float min time of visits
        @param tevmax: float max time of visits
        @param tevmin_train: float min time of visits in training data
        @param tevmax_train: float max time of visits in training data
        @param tevmean_train: float mean time of visits in training data
        @param verbose: bool verbose mode
        @param user_max_lam: np.array maximum intensity per member
        @param user_n_imp: np.array mean number of impressions per member
        @param fix_n_imp: if true then take `user_n_imp` fixed impressions per memeber
        @return: simulated dataset of impressions
    """

    # need local import otherwise multiprocessing will not work (silent error):
    from . import sim_models
    import tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tensorflow.keras.backend.set_floatx('float64')
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    user_model, rec_models, visit_model = models  # unpack models

    # initialize user_states: list of np.array representing complete history for each user
    user_states = initial_user_state.toarray().copy()  # tracks current user state for all users
    #user_states = initial_user_state.copy()  # tracks current user state for all users

    # define some constants:
    if user_max_lam is None:
        user_max_lam = hyp['max_lam'] * np.ones(stg['NU'])  # estimate upper bound for all users
    n_potential_visits = [np.random.poisson(lam=m * stg['T']) for m in user_max_lam]  # num events per user

    # create np.array of NU-by-N visit deltas for each user
    # (indicate non-visits with nan):
    max_n_visits = max(n_potential_visits)
    user_potential_visits = np.nan + np.zeros((stg['NU'], max_n_visits))  # stg['INF_TIME'] +
    
    for u, n_points in enumerate(n_potential_visits):
        user_potential_visits[u, :n_points] = np.sort(np.random.uniform(low=tevmin, high=tevmax, size=n_points))
    # user_potential_visits is now the array indicating global visit time for set of potential visits per user
    
    print('tevmin, tevmax', tevmin, tevmax)
    
    arb_rec_model = rec_models[0]
    time_rec = isinstance(arb_rec_model, sim_models.RecModelTime)
    if time_rec:
        # create time features for each potential visit per member:
        T = np.zeros((max_n_visits, stg['NU'], arb_rec_model._time_bins.shape[0] - 1))
        for u in range(stg['NU']):
            T[:, u, :] = arb_rec_model.process_time_feature(user_potential_visits[u, :], create_bins=False).toarray()

    # randomly draw num of impressions for each user and potential visit
    # (todo: refactor this part into separate function)
    if user_n_imp is None:
        imp_shape = (stg['NU'], max_n_visits)
        if fix_n_imp:
            user_nimp = stg['lambda_impression']*np.ones(imp_shape)
        else:
            user_nimp = np.random.poisson(stg['lambda_impression'],
                                      size=imp_shape)  # number of impressions at each potential visit
    else:
        if fix_n_imp:
            user_nimp = np.repeat(user_n_imp[:,np.newaxis], max_n_visits, axis=1)
        else:
            user_nimp = np.array([np.random.poisson(user_n_imp) for _ in range(max_n_visits)]).T
    user_nimp = np.minimum(int(stg['NI']/2), np.maximum(1, user_nimp)) # ensure n_impressions per visit 1 <= n_imp <= num items/2.
    user_nimp = user_nimp.astype('int64')

    iteration = 0
    # init simulation, update on every iteration of the main loop:
    simdict = {'user_id': np.zeros(0, dtype='int64'),
               'time': np.zeros(0),
               'action': np.zeros(0, dtype='int64'),
               'potential_action': np.zeros(0, dtype='int64'),
               'state': [],
               'reward': np.zeros(0)}
    simulation = pd.DataFrame(simdict)

    print('running batch simulation...')
    while iteration < max_n_visits:
        # process all users in each iteration (all users who have not had final visit).
        # decide which users will be potentially visiting in this time block
        # in order to prepare features for visit_model prediction:
        # evaluate intensity function for all potential visitors, then decide which ones 
        # are actually visiting with rejection sampling:
        pvs, = np.where(~np.isnan(user_potential_visits[:, iteration]))  # potential visitors
        # prepare feature vector X (one per potentially active user)
        if hyp['hyp_study']['constant_rate']:
            X = np.zeros(1)
        else:
            X = sim_models.process_features_static_points(simulation, user_potential_visits[pvs, iteration], pvs, (
                stg['NI'], hyp['visit_model_hyp']['window_size'], tevmean_train, tevmin_train, tevmax_train),
                                                          init_user_states=initial_user_state)

        # update prediction cache for user and rec models
        rec_states = np.hstack((user_states, T[iteration, :, :])) if time_rec else user_states
        [rec_model.populate_cache(rec_states) for rec_model in rec_models]
        user_model.populate_cache(user_states)
        if verbose:
            print('updated user/rec policy per user.')
            print('processed stochastic process features.')

        if X.shape[0] > 0:
            if hyp['hyp_study']['constant_rate']:
                pvs_lam = np.zeros_like(pvs)
                accept_pr = np.ones_like(pvs)
            else:
                mix = hyp['visit_model_hyp']['self_exciting_mixture']
                pvs_lam_norm = (1-mix)*visit_model._model.predict(X.toarray()) + mix*visit_model._lambda_e_model.predict(X.toarray()) # predicted lambda for each active user
                pvs_lam = pvs_lam_norm * visit_model.time_scale  # model predicts over [-1,+1] time, need to spread over
                # original range
                accept_pr = pvs_lam.ravel() / user_max_lam[pvs]
                
            print_every = 20
            if (iteration % print_every) == 0:
                print('iteration %i, num. of active users' % iteration, pvs_lam.ravel().shape)
                print('pvs_lam', pvs_lam.ravel())
                print('max_lam', user_max_lam[pvs])
            assert np.all(accept_pr <= 1.0), pvs_lam.ravel().max()
            is_visit = np.array(np.random.binomial(1, accept_pr), dtype=np.bool_)
            uz_visit = pvs[is_visit]  # decide which users are actually visiting

            if verbose: print('sampled visiting users.')
            # iterate over visiting users to draw recommendations:
            for u in uz_visit:
                user_state = user_states[u, :].copy()  # pull out current user state
                rec_state = rec_states[u, :]
                n_imp = user_nimp[u, iteration]
                rec_prop = SETTINGS.hyp['rec_prop']  # draw rec_id based on global probability
                rec_id = np.random.choice(np.arange(len(rec_models), dtype='int16'), p=rec_prop)
                # draw recommendation for user, depending on which rec_id to use:
                az = rec_models[rec_id].select(rec_state, n_imp, hyp, stg)
                # select count depends on whether using Bernoulli or Categorical model:
                select_count = az if isinstance(user_model, sim_models.UserModelBinary) else hyp['user_select_count']
                # draw consumed title for user (i.e. title they would have consumed if they had been recommended it)
                cz = user_model.select(user_state, select_count, hyp, stg)
                assert n_imp > 0, n_imp

                simdict, user_state = calculate_reward_by_overlap(user_state, u, n_imp, az, cz, user_potential_visits,
                                                                  iteration, rec_id, stg)
                # update user state
                user_states[u, :] = user_state  # update current user state for user u
                try:
                    new_sim = pd.DataFrame(simdict)
                except ValueError:
                    print([len(x) for x in [simdict['user_id'], simdict['state'], simdict['action'], simdict['potential_action'], simdict['time']]])
                    raise ValueError()
                    
                simulation = simulation.append(new_sim, ignore_index=True, sort=False)
        iteration += 1

    return simulation


def calculate_reward_by_overlap(user_state: np.ndarray, u: int, n_imp: int, az: np.ndarray, cz: np.ndarray,
                                user_potential_visits: np.ndarray, iteration: int, rec_id: int, stg: Dict) \
        -> Tuple[Dict, np.ndarray]:
    """
        Calculates reward by inspecting actions taken by member and recommender and looking for any overlap.

        @param user_state: np.array(n_members, n_items) representing state of each member
        @param u: int, member id
        @param n_imp: int, number of impressions
        @param az: np.array of actions taken by recommender
        @param cz: np.array of (potential) actions taken by member
        @param user_potential_visits: np.array(n_users, n_iterations) of time of visit for each member
        @param iteration: int, iteration id
        @param rec_id: int, recommender id
        @param stg: dict of settings
        @return: dictionary representing new rows for the simulation, updated user state
    """

    A = np.bincount(az, minlength=stg['NI'])
    C = np.bincount(cz, minlength=stg['NI'])
    R = A * C
    streamed, = np.where(R > 0)
    # build list of streamed and non-streamed actions
    az_success = list(streamed)
    cz_success = list(streamed)
    az_fail = list(set(az) - set(az_success))
    cz_fail = list(set(cz) - set(cz_success))
    len_rz = min(1, len(streamed))
    simdict = {'user_id': np.array([u] * n_imp, dtype='int64'),  # np.repeat(u,n_imp,dtype='int64'),
               'time': [user_potential_visits[u, iteration]] * n_imp,  # np.repeat(global_t,n_imp),
               'action': az_success + az_fail,
               'potential_action': [-1] * n_imp,  # cz_success+cz_fail,
               'state': [utils.bow(user_state)] * n_imp,
               'reward': [1] * len_rz + [0] * (n_imp - len_rz),
               'rec_id': rec_id}
    user_state[np.random.permutation(streamed)[:1]] = 1
    return simdict, user_state


def load_models(user_model_path: str, rec_model_paths: str, visit_model_path: str, hyp: Dict, stg: Dict,
                rec_type: str = 'static', user_binary: bool = False) -> Tuple:
    # load models from disk
    from . import sim_models

    # load user model:
    if user_binary:
        user_model = sim_models.UserModelBinary(stg['NI'], stg['NI'], hyp['user_model_hyp'])
    else:
        user_model = sim_models.UserModel(stg['NI'], stg['NI'], hyp['user_model_hyp'])
    user_model.load(user_model_path)

    # load recommender models (multiple models because we consider multiple recommenders in any single simulation):
    ceil_t = np.ceil(stg['T']).astype('int')
    rec_models = []
    for rec_model_path in rec_model_paths:
        if type(rec_model_path) == str:
            if rec_type == 'static':
                rec_model = sim_models.RecModel(stg['NI'], stg['NI'], hyp['rec_model_hyp'])            
                rec_model.load(rec_model_path)
            elif rec_type == 'time':
                rec_model = sim_models.RecModelTime(stg['NI'], stg['NI'], hyp['rec_model_hyp'], ceil_t)
                rec_model.load(rec_model_path)
            elif rec_type == 'nmf':    
                print('nmf model with hyp =',SETTINGS.hyp['hyp_study'])
                rec_model = sim_models.NMFRec(stg['NI'], SETTINGS.hyp['hyp_study'])
                if len(rec_model_path)>0:
                    nmf_rec_model_dat = pd.read_csv(rec_model_path) # path is to history of data not model here
                    rec_model.fit(nmf_rec_model_dat, stg['NI'])
            else:
                raise Exception('rec model type unrecognized: ' + rec_type)
        else:
            rec_model = rec_model_path

        rec_models.append(rec_model)

    # load user visit model:
    visit_model = sim_models.VisitModel(stg['NI'], stg['NU'], hyp['visit_model_hyp'])
    visit_model.load(visit_model_path, None)

    return (user_model, rec_models, visit_model)


def batch_sim(uids: np.ndarray, kwargs: Dict) -> pd.DataFrame:
    print('batch sim on uids', uids.min(), 'to', uids.max(), '(inclusive)')
    stg_local = kwargs['stg'].copy()
    uid_min, uid_max = uids.min(), uids.max()+1
    stg_local['NU'] = uid_max - uid_min
    stg_local['T'] = kwargs['tevmax']
    init_states = kwargs['S_init'][uid_min:uid_max, :]
    model_paths = kwargs['model_paths']
    if 'hyp' in kwargs:
        local_hyp = kwargs['hyp']
    else:
        local_hyp = SETTINGS.hyp
    if kwargs['user_max_lam'] is not None:
        user_max_lam = kwargs['user_max_lam'][uid_min:uid_max]
    else:
        user_max_lam = None
    user_path, rec_path, visit_path = model_paths['user_model'], model_paths['rec_model'], model_paths['visit_model']
    models = load_models(user_path, rec_path, visit_path, local_hyp, stg_local, rec_type='nmf', user_binary=True)
    s = simulate_batch(init_states, models, local_hyp, stg_local, kwargs['tevmin'],
                       kwargs['tevmax'], kwargs['tevmin_train'], kwargs['tevmax_train'], kwargs['tevmean_train'], 
                       user_max_lam=user_max_lam, user_n_imp=kwargs['empirical_mean_imp_per_visit'][uid_min:uid_max],
                       fix_n_imp=kwargs['fix_n_imp'] if 'fix_n_imp' in kwargs else False)
    # adjust uid to avoid clash with other child processes:
    s.user_id = s.user_id + uid_min
    return s


def prepare_sim(simulation: pd.DataFrame, stg: Dict, base_path: str) -> Dict:
    # find initial states for all users:
    # please note: groupby preserves order when sort=False https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
    first_imps = simulation.sort_values(['time']).groupby('user_id',sort=False).first().sort_values(['user_id'])
    user_ids = np.sort(first_imps.index)
    # initial_user_state is np.array NU-x-NI int64
    S_init = utils.inv_bow_all(first_imps.state.values, stg['NI'], dense=False).tocsr()
    
    # update num. users to reflect only users with streams in observation:
    stg_local = stg.copy()
    n_users = S_init.shape[0]
    np.random.seed(0)
    uids = np.random.choice(np.arange(stg['NU'],dtype='int32'),size=n_users,replace=False)
    tevmean, tevmin, tevmax = process_dat.calc_tev(simulation)
    
    # calculate average number of visits per user (defined as user_id-by-time group of impressions):
    sg = simulation.groupby(['user_id','time'])
    empirical_n_visits = sg.count().groupby('user_id')['action'].count().values
    
    # calculate average number of impressions per visit per user
    imp_series = sg['action'].count().groupby('user_id').mean()
    assert np.all(imp_series.index == user_ids) # make sure user ids match up

    return {'S_init': S_init,
           'stg': stg,
           'tevmax': tevmax, 
           'tevmin': tevmin,
           'tevmean': tevmean, 
           'user_max_lam': None,
           'empirical_mean_imp_per_visit': imp_series.values,
           'base_path': base_path
           }


def prepare_sim_contentwise(simulation: pd.DataFrame, stg: Dict, hyp: Dict, user_lam: np.ndarray, base_path: str) -> Dict:
    # find initial states for all users:
    # please note: groupby preserves order when sort=False https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html
    first_imps = simulation.sort_values(['time']).groupby('user_id',sort=False).first().sort_values(['user_id'])
    user_ids = np.sort(first_imps.index)
    # initial_user_state is np.array NU-x-NI int64
    S_init = utils.inv_bow_all(first_imps.state.values, stg['NI'], dense=False).tocsr()
    
    np.random.seed(0)
    tevmean, tevmin, tevmax = process_dat.calc_tev(simulation)
    
    # calculate average number of visits per user (defined as user_id-by-time group of impressions):
    sg = simulation.groupby(['user_id','round_time'])
    empirical_n_visits = sg.first().groupby('user_id').action.count().values
    
    # calculate average number of impressions per visit per user
    imp_series = sg['action'].count().groupby('user_id').mean()
    assert np.all(imp_series.index == user_ids) # make sure user ids match up

    if not(hyp['constant_rate']): user_lam = None
        
    return {'S_init': S_init,
           'stg': stg,
           'tevmax': tevmax, 
           'tevmin': tevmin,
           'tevmean': tevmean, 
           'user_max_lam': user_lam, 
           'empirical_mean_imp_per_visit': imp_series.values,
           'base_path': base_path
           }

def parsim(rec_model_config: List, **kwargs):
    user_set = np.arange(kwargs['S_init'].shape[0], dtype='int64')
    test_id = SETTINGS.simulation_components['ab_test_id']
    # add model_paths to kwargs and pass through to batch_sim:
    kwargs['model_paths'] = {'user_model': join(kwargs['base_path'], SETTINGS.filepaths['user_model_test']) % test_id,
                             'rec_model': [join(kwargs['base_path'], SETTINGS.filepaths['rec_model_t_test']) % (test_id, cell_id, rec_id) for
                                           test_id, cell_id, rec_id in rec_model_config],
                             'visit_model': join(kwargs['base_path'], SETTINGS.filepaths['visit_model_test-%s' % test_id] + '.big')
                             }
    return utils.parallelize_fnc(batch_sim, user_set, kwargs, int(SETTINGS.hyp['cores']/2))
