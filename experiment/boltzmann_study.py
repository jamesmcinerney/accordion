import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy
from simtrain import run_sim_ab as run_sim
from simtrain import SETTINGS_POLIMI as SETTINGS
from simtrain import utils
import paths
from os.path import join


# plot stats for temp sweep:
def plot_temp_sweep(st, suffix_plot):
    results_temp = pd.DataFrame(columns=['inverse temperature', 'mean total reward per user', 'mean n_visits per user', 'sem reward', 'sem visits'])
    for T in SETTINGS.hyp['hyp_study']['temperature_sweep']:
        print('INV TEMP',T)
        event_count_per_user, rewards_per_user = simple_summary(st[T], verbose=False)
        results_temp = results_temp.append({'inverse temperature':T, 
                             'mean total reward per user':rewards_per_user.mean(), 
                             'mean n_visits per user':event_count_per_user.mean(), 
                             'eb reward':rewards_per_user.sem(), 
                             'eb visits':event_count_per_user.sem()},
                           ignore_index=True)

    line_keys = [('mean total reward per user','eb reward','-'),
                ('mean n_visits per user', 'eb visits', '--')]

    for line_mean, line_sem, ls in line_keys:
        plt.errorbar(results_temp['inverse temperature'], results_temp[line_mean], yerr=results_temp[line_sem], label=line_mean, ls=ls)
    plt.legend()
    plt.xlabel('inverse temperature')
    plt.savefig('dat/' + SETTINGS.rootpaths['plots'] + 'inv_temp_sweep_%s.pdf' % suffix_plot, dpi=300) 


# calculate basic stats from training data extrapolated to test data:
def simple_summary(dat, verbose=True):
    # calculate avg num visits per user per day in train / validation:
    uids = dat.user_id.unique()
    if verbose: print('targeting uids',uids)
    
    sessions = dat.sort_values(['user_id','reward'],ascending=[True,False]) \
                            .groupby(['user_id','round_time'],sort=False,as_index=False).first()
    sessions_per_user = sessions.groupby('user_id')
    rewards_per_user = sessions_per_user.reward.sum()
    event_count_per_user = sessions_per_user.action.count()
    
    if verbose:
        print('avg_reward per session:',sessions.reward.mean(),'\n')
        print('n_sessions per user:\n','median = ',event_count_per_user.median(),'\n',event_count_per_user.describe(),'\n')
        print('total reward per user:\n','median = ',rewards_per_user.median(),'\n',rewards_per_user.describe(),'\n')

    return event_count_per_user, rewards_per_user
    

def batch_sim_kwargs(rec_model_config, **kwargs):
    test_id = SETTINGS.simulation_components['ab_test_id']
    # add model_paths to kwargs and pass through to batch_sim:
    kwargs['model_paths'] = {'user_model': join(paths.dat, 'opt_user_model.h5'),
                             'rec_model': [join(paths.dat, 'opt_rec_model.h5') for
                                           test_id, cell_id, rec_id in rec_model_config],
                             'visit_model': join(paths.visit_model, SETTINGS.filepaths['visit_model_test-%s' % test_id] + '.big')
                             }
    return kwargs


def user_activity(dat):
    # calculate homogeneous rate of visits for all users:
    user_lam = dat.sort_values('user_id') \
                        .groupby(['user_id','round_time'],sort=False) \
                        .first() \
                        .groupby('user_id',sort=False) \
                        .action \
                        .count() \
                        .values / (dat.time.max() - dat.time.min())
    return user_lam

    
def simulate_temperature(user_set, user_lam, train_dat, test_dat, train_stg, test_stg, rec_model_config, time_sweep, rate_style, rnd_seed=0):
    # tweak settings for retraining model during simulation:
    local_hyp = SETTINGS.hyp['hyp_study']
    local_hyp['constant_rate'] = (rate_style == 'pp')
    sim_settings = run_sim.prepare_sim_contentwise(test_dat, test_stg, local_hyp, user_lam, paths.dat)
    sim_settings['tevmin_train'] = train_dat.time.min()
    sim_settings['tevmax_train'] = train_dat.time.max()
    sim_settings['tevmean_train'] = train_dat.time.mean()
    batch_sim_kwargs_0 = batch_sim_kwargs(rec_model_config, **sim_settings)
    print('batch_sim_kwargs_0',batch_sim_kwargs_0)
    np.random.seed(rnd_seed)
    s_temp = {} # maps temperature -> simulation

    for temperature in SETTINGS.hyp['hyp_study']['temperature_sweep']:
        print("\n\n\n\n*********************************************")
        print("TEMP =",temperature)
        batch_sim_kwargs_0['hyp'] = SETTINGS.hyp.copy()
        batch_sim_kwargs_0['hyp']['hyp_study']['temperature'] = temperature
        batch_sim_kwargs_0['model_paths']['rec_model'][0] = ''
        s_all = None

        for ti in range(len(time_sweep)-1):
            print('\nsimulating time range =',time_sweep[ti:ti+2],'...')
            batch_sim_kwargs_0['stg']['T'] = time_sweep[ti+1] - time_sweep[ti]        
            # increment time by running clock in kwargs
            batch_sim_kwargs_0['tevmin'] = time_sweep[ti]
            batch_sim_kwargs_0['tevmax'] = time_sweep[ti+1]
            s_next = run_sim.batch_sim(user_set, batch_sim_kwargs_0)
            if s_all is None: 
                s_all = s_next
            else:
                s_all = s_all.append(s_next, ignore_index=True)
            s_all['round_time'] = np.floor(s_all.time.values*48)/48
            s_all.to_csv(paths.nmf_dat_stem % ti)
            # update train dat path and init user states:
            batch_sim_kwargs_0['model_paths']['rec_model'][0] = paths.nmf_dat_stem % ti
            s_next_pos = s_next[s_next.reward>0]
            batch_sim_kwargs_0['S_init'][s_next_pos.user_id,s_next_pos.action] = 1
        s_temp[temperature] = s_all
    return s_temp