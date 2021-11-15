import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from . import SETTINGS_POLIMI as SETTINGS, utils, constants
from sklearn import metrics
import pandas as pd
from typing import Callable, List, Tuple, Dict
from os.path import join

DAYS_PER_HOUR = 1 / 24.



def plot_rnd_user_visits(seed: int, show_settings: Dict, simulation: pd.DataFrame, visit_model, stg: Dict, hyp: Dict,
                         base_path: str, uid: int = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
        Plot the visits and Poisson process intensities (and print summary statistics) for a random member.
    """
    from . import sim_models
    plt.figure(figsize=(9,4)) # golden ratio
    # pick a random streaming user to plot:
    user_id = uid if uid is not None else simulation[simulation.reward > 0].user_id.sample(random_state=seed).values[0]
    max_t = stg['T']
    ts = np.arange(0, max_t, DAYS_PER_HOUR / 2)  # plot increments of half hour
    lambda_models = []
    (show_base, show_state, show_hawkes, show_total) = show_settings
    if show_base: lambda_models.append((visit_model._lambda_0_model, 'global intensity', '-'))
    if show_state: lambda_models.append((visit_model._lambda_u_model, 'state intensity', '-'))
    if show_hawkes: lambda_models.append((visit_model._lambda_e_model, 'self-exciting intensity', '-'))
    if show_total: lambda_models.append((visit_model._model, 'total intensity', '--'))
    simulation_u = simulation[simulation.user_id == user_id]
    # get reward times (simulation_original, T, U, fixed_params)
    reward_times = simulation_u[(simulation_u.reward > 0) & (simulation_u.time <= max_t)].sort_values('time')[
        ['time', 'action']].values
    visit_times = simulation_u[simulation_u.time <= max_t].time.values
    Tevmean, Tevmin, Tevmax = visit_model._T_mean, visit_model._T_min, visit_model._T_max
    Xexplore = sim_models.process_features_static_points(simulation_u, ts, user_id * np.ones(len(ts), dtype='int64'), (
        stg['NI'], hyp['visit_model_hyp']['window_size'], Tevmean, Tevmin, Tevmax)).toarray()
    plt.scatter(visit_times, np.zeros_like(visit_times), marker='x', label='visit', color='gray')
    plt.scatter(reward_times[:, 0], np.zeros_like(reward_times[:, 0]), marker='x', label='positive interaction', color='blue')
    for m, l, s in lambda_models:
        lambdas = m.predict(Xexplore) * visit_model.time_scale
        plt.plot(ts, lambdas, label=l, ls=s)    
    plt.xlabel('time (days)')
    plt.ylabel('intensity')
    
    # fix order of legend:
    handles, labels = plt.gca().get_legend_handles_labels()
    # sort both labels and handles by labels
    legend_ord = [4, 5, 3, 0, 1, 2]
    labels = [labels[i] for i in legend_ord]
    handles = [handles[i] for i in legend_ord]
    plt.gca().legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    if base_path is not None:
        fig_path = join(base_path, SETTINGS.rootpaths['plots'], 'example_intensity.pdf')
        plt.savefig(fig_path, dpi=300)
        print('saved fig to',fig_path)
    user_rewards = simulation[simulation.user_id == user_id].reward.values
    print(stats.describe(user_rewards))
    print('total imp for user', simulation_u.shape[0])
    print('total visits for user', simulation_u.groupby('time').ngroups)
    print('total streams for user', user_rewards.sum())
    print('ave streams per day for user', user_rewards.sum() / (stg['T']))
    print('ave imp per day for user', len(user_rewards) / (stg['T']))
    print('ave imp per visit for user', simulation_u.groupby('time').time.count().mean())
    sim_time = simulation_u.sort_values('time')
    print('num. streamed titles at t=0', utils.inv_bow(sim_time.iloc[0, :].state, stg['NI']).sum())
    print('num. streamed titles at final t', utils.inv_bow(sim_time.iloc[-1, :].state, stg['NI']).sum())
    return Xexplore, ts


def summarize_listwise(rec_model, user_model, simulation: pd.DataFrame, topk: int, stg: Dict):
    """
    Calculates precision@K and recall@K for a dataset given recommender and users models.

    @param rec_model: tf.keras.Model recommender model
    @param user_model: tf.keras.Model member model
    @param simulation: pd.DataFrame test dataset
    @param topk: at-K parameter
    @param stg: settings dict
    @return: None
    """

    def summarize_user_fit(pred, sim, explore=1, topk=100):
        # make scatter of highest scoring items according to rec and usr models:
        # find out which items are common to top of both models
        # 'sim' is pandas.DataFrame
        rec_pred, usr_pred, user_id_ = pred
        top_common_items = (rec_pred * usr_pred).argsort()[-topk:]  # approximate method
        if explore:
            plt.scatter(rec_pred[top_common_items], usr_pred[top_common_items], marker='x')
            plt.title('rec model score vs. user model score for top ')

        user_consumed_items = np.unique(sim[(sim.user_id == user_id_) & (sim.reward > 0)].action.values)
        intersection = set(top_common_items).intersection(set(user_consumed_items))
        recall = len(intersection) / len(user_consumed_items) if len(user_consumed_items) > 0 else np.nan
        prec = len(intersection) / float(topk)
        if explore:
            print('top_common_items', np.sort(top_common_items))
            print('consumed items', np.sort(user_consumed_items))
            print('overlap', intersection)
            print('recall', recall)
            print('precision@%i' % topk, prec)
        return recall, prec

    rnd_rows = simulation.sample(1000, random_state=0)
    states = utils.inv_bow_all(rnd_rows.state.values, stg['NI'], dense=True)
    rec_states, _ = rec_model.process_features(rnd_rows, stg['NI'])
    userids = rnd_rows.user_id.values
    listwise_stats = np.array(list(map(lambda xs: summarize_user_fit(xs, simulation, explore=0, topk=topk),
                                       zip(list(rec_model._model.predict(rec_states)),
                                           list(user_model._model.predict(states)),
                                           userids))))
    print('recall@%i,prec@%i:\n' % (topk, topk), listwise_stats)
    print('ave. recall@%i, precision@%i per user (on training data):\n' % (topk, topk),
          np.nanmean(listwise_stats, axis=0))


def summarize_sim(S: pd.DataFrame, hist_settings: Dict = None) -> Dict:
    """
    Summary statistics of a dataset S, helps compare simulations at the aggregate level.
    @param S: DataFrame
    @param show_hist: boolean, if true then plot summary as histogram
    @param hist_settings: dict of histogram settings (if not(show_history) then has no effect)
    @return: dictionary of summary
    """
    if hist_settings is None:
        hist_settings = {'n_bins': 200,
                         'log_scale': False,
                         'x_max': None
                         }
    S1 = S.copy()
    S1['01reward'] = np.minimum(1, S.reward.values)
    sg = S.groupby(['user_id', 'time'])
    n_visits = sg.count().groupby(['user_id'])['action'].count().values
    R01 = np.minimum(np.ones_like(S.reward), S.reward)
    n_users = S.user_id.nunique()
    S1['time_int'] = S1.time.map(int)

    results = {}
    results['num. users with > 0 impressions'] = S.user_id.nunique()
    results['num. users with > 0 rewards'] = S[S.reward > 0].user_id.nunique()
    results['ave. user total visits\t'] = n_visits.mean()
    results['median user total visits\t'] = np.median(n_visits)
    results['ave. user total impressions\t'] = sg.count().groupby('user_id')['action'].sum().mean()
    results['ave. user unique impressions\t'] = S.groupby('user_id')['action'].nunique().mean()
    results['ave. impressions per visit\t'] = S.shape[0] / S.groupby(['user_id', 'time']).ngroups
    results['ave. user total reward (0/1 reward)\t'] = R01.sum() / n_users
    results['ave. user total reward\t\t'] = S.reward.sum() / n_users
    results['ave. reward (0/1 reward)\t'] = R01.mean()
    results['ave. reward\t\t\t'] = S.reward.mean()
    results['median user reward total\t'] = S.groupby('user_id').reward.sum().median()
    results['median user reward total\t'] = S.groupby('user_id').reward.sum().median()
    results['median user 0/1 reward total\t'] = S1.groupby('user_id').reward.sum().median()
    results['days with a qualified play'] = (S1.groupby(['user_id', 'time_int']).reward.sum() > 0).sum() / (
        S1.user_id.nunique())
    results['reward (0/1) describe\t'] = S.reward.describe()
    # print()
    # print('reward describe\t\t',R01.describe())
    xs = np.log(n_visits) if hist_settings['log_scale'] else n_visits
    plt.hist(xs, bins=hist_settings['n_bins'])
    if hist_settings['x_max'] is not None: plt.xlim(0, hist_settings['x_max'])
    # print(json.dumps(results))
    return results


def explore_title_streams(sim: pd.DataFrame, reference_sim: pd.DataFrame = None, K: int = 100):
    """
    Plot histogram and print statistics showing how many times each title was streamed.
    @param sim: DataFrame
    @param reference_sim: DataFrame of alternative dataset with which to find overlap with `sim`
    @param K: cutoff parameter for overlap stat
    @return: None
    """
    # bins equal to num titles
    NI1 = sim.action.nunique()
    bins = np.arange(0, NI1 + 1, 10)
    plt.xlim(0, NI1)
    plt.ylim(0, 300)
    plt.hist(sim.groupby('action').reward.sum(), bins=bins)
    plt.xlabel('title')
    plt.ylabel('frequency')
    print('top 10 actions', sim.groupby('action').reward.sum().sort_values(ascending=False)[:10])
    if reference_sim is not None:
        topk_A = sim.groupby('action').reward.sum().sort_values(ascending=False)[:K]
        topk_B = reference_sim.groupby('action').reward.sum().sort_values(ascending=False)[:K]
        overlap = len(set(topk_A) - set(topk_B)) / float(K)
        print('percent overlap with reference simulation @ top %i = ' % K, 100 * overlap)


def plot_overlap(simA: pd.DataFrame, simB: pd.DataFrame, Ks: List, imp: bool = False):
    """
    Plot the overlap of two datasets w.r.t. actions and positive reward actions.
    @param simA: DataFrame of dataset A
    @param simB: DataFrame of dataset B
    @param Ks:  [int] cutoff parameters
    @param imp: bool indicating whether or not to ignore reward of actions
    @return: None
    """
    if not (imp):
        print('num unique success actions in simA = ', simA[simA['reward'] > 0].action.nunique())
        print('num unique success actions in simB = ', simB[simB['reward'] > 0].action.nunique())
    else:
        print('num unique actions in simA = ', simA.action.nunique())
        print('num unique actions in simB = ', simB.action.nunique())

    get_top = lambda s: s.groupby('action').reward.sum().sort_values(ascending=False) if not (imp) else \
        s.groupby('action').reward.count().sort_values(ascending=False)
    topk_A = get_top(simA).index
    topk_B = get_top(simB).index
    overlap = np.zeros(len(Ks))
    for i, K in enumerate(Ks):
        overlap[i] = len(set(topk_A[:K]).intersection(set(topk_B[:K])))  # / float(K)
    plt.plot(Ks, overlap)
    plt.xlabel('top K')
    plt.ylabel('number of overlapping unique titles')
    # show identity line for comparison
    plt.plot([0, max(Ks)], [0, max(Ks)], ls='--', color='g')
