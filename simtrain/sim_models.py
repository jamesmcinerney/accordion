import itertools

import scipy
from scipy import sparse
import numpy as np
from . import SETTINGS_POLIMI as SETTINGS, utils, constants
import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import backend as kb
from typing import Tuple, Dict, Callable

tf.keras.backend.set_floatx('float64')
from tensorflow.keras.callbacks import EarlyStopping, Callback
import os
from os.path import join
import pandas as pd
from sklearn.decomposition import NMF


os.environ['NUMEXPR_MAX_THREADS'] = SETTINGS.NUMEXPR_MAX_THREADS
pd.options.mode.chained_assignment = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('loaded sim_models')

def normalize_layer() -> k.layers.Layer:
    return k.layers.Lambda(lambda x: x / (SETTINGS.EPS + kb.sum(x * x, axis=-1, keepdims=True)))


def divide_by_layer(V: float) -> k.layers.Layer:
    return k.layers.Lambda(lambda x: x / V)


def densify_layer(L: int) -> k.layers.Layer:
    # L is length of dense vector
    return k.layers.Lambda(lambda x: tf.ensure_shape(tf.sparse.to_dense(x), (None, L)))


class SimulatorModel(object):
    """
        Base class for trainable simulation models.
    """

    def __init__(self, ndims_in: int, ndims_out: int, model_hyp: Dict):
        state_model = self._build_state_model(ndims_in, model_hyp)
        self._model = self._build_output_model(state_model, ndims_out)
        self._pred_cache = {}  # caches input->output mapping
        self._hyp = model_hyp

    def _build_state_model(self, ndims_in: int, model_hyp: Dict) -> k.models.Model:
        """
            Build a model that takes state as input and outputs an embedding.
            @param ndims_in: number of dimensions of input
            @param model_hyp: model hyperparameters
            @return: keras model
        """
        dropout_rate = model_hyp['dropout_rate']

        model = k.models.Sequential()
        model.add(k.Input(shape=(ndims_in,)))
        if dropout_rate > 0: model.add(k.layers.Dropout(dropout_rate))
        model.add(normalize_layer())  # TODO: extend to sparse vesion
        # model.add(k.layers.Lambda(lambda x: x/kb.sum(x*x)))

        for layer_id in range(model_hyp['n_hidden_layers']):
            model.add(k.layers.Dense(model_hyp['n_nodes'][layer_id], input_dim=ndims_in, \
                                     activation=model_hyp['hidden_activation']))
            if dropout_rate > 0: model.add(k.layers.Dropout(dropout_rate))

        return model

    def _build_output_model(self, state_model: k.models.Model, ndims_out: int) -> k.models.Model:
        """
            Build a model on top of state model to predict output.
            @param state_model:
            @param ndims_out:
            @return: keras model
        """
        state_model.add(k.layers.Dense(ndims_out, activation='softmax'))
        state_model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
        return state_model

    def select(self, user_state: np.ndarray, n: int, hyp: Dict, stg: Dict, verbose: int = 0) -> np.ndarray:
        """
            Draw unique categorical items from model predictive distribution.
            @param user_state: array representing bag of words of items consumed
            @param n: number of samples
            @param hyp: hyperparameters
            @param stg: general settings
            @param verbose: print internals around caching
            @return: sample of item IDs
        """
        key = utils.bow(user_state)
        if key in self._pred_cache:
            p = self._pred_cache[key]
            if verbose: print('cache hit in simulator model')
        else:
            if verbose: print('cache miss in simulator model')
            #p = self._model.predict(np.array([user_state])).ravel()  # inefficient
            p = self.predict(np.array([user_state])).ravel()  # inefficient
            # store in cache
            self._pred_cache[key] = p
        #val = np.random.choice(len(p), n, replace=False, p=p)
        if n < len(p[p>0]):
            val = np.random.choice(len(p), n, replace=False, p=p)
        else:
            # sampling with replacement
            val = np.random.choice(len(p), n, replace=True, p=p)

        return val

    def populate_cache(self, user_states: sparse.csr_matrix):
        """
            Do batch prediction then store results in a dict to cache.
            @param user_states: sparse matrix of consumed items
            @return: None
        """
        ps = self.predict(user_states)
        for i in range(user_states.shape[0]):
            key_str = utils.bow(user_states[i, :])
            self._pred_cache[key_str] = ps[i]

    def fit(self, state_history: sparse.csr_matrix, reward_history: np.ndarray, hyp: Dict):
        """
            Fit the model from a history of impressions, each with state and given reward as outcome.
            @param state_history: history of states, one for each impression
            @param reward_history:  history of rewards, one for each impression
            @param hyp: dictionary of hyperparameters
            @return: None
        """
        raise NotImplementedError

    def decode_state(self, state: np.ndarray, NI: int) -> sparse.csr_matrix:
        """
            Converts state to a sparse matrix.
            @param state: array of items consumed
            @param NI: number of items
            @return: sparse matrix
        """
        return utils.inv_bow_all(list(state), NI, dense=0)

    def predict(self, x: sparse.csr_matrix) -> np.ndarray:
        """
            Predicts output of model given input.
            @param x: input
            @return: output prediction of model
        """
        return self._model.predict(x)

    def nws(self, sim: pd.DataFrame, NI: int) -> np.ndarray:
        """
            Calculate normalized inverse propensity scores from given data.
            @param sim: dataset from which to derive features
            @param NI: number of items
            @return: array of normalized inverse propensity scores
        """
        #
        X = self.process_features(sim, NI)
        P = self.predict(X)
        prop = P[np.arange(sim.shape[0], dtype='int64'), sim.action.values]
        ws = 1 / prop
        nws = ws / ws.sum()
        return nws

    def save(self, filepath: str):
        self._model.save(filepath)

    def load(self, filepath: str):
        """
            Supports prediction but not training from saved model.
            @param filepath:
            @return: None
        """
        self._model = k.models.load_model(filepath, custom_objects={"kb": kb})


class SimulatorModelTime(SimulatorModel):
    """
        Extends `SimulatorModel` class to include time as a feature for taking into account temporal popularity.
    """

    def __init__(self, ndims_in: int, ndims_out: int, model_hyp: Dict, num_days: int):
        # define inputs
        time_dim = int(np.ceil(model_hyp['time_slots_per_day'] * num_days))
        all_input = k.layers.Input(shape=(ndims_in + time_dim,))
        state_input = all_input[:, :ndims_in]
        time_input = all_input[:, ndims_in:]

        # get embedding of state:
        state_model = self._build_state_model(ndims_in, model_hyp)
        state_embedding = state_model(state_input)

        # add time feature in final "hidden" layer
        h = k.layers.Concatenate(axis=-1)([state_embedding, time_input])

        # map to output layer
        output = k.layers.Dense(ndims_out, activation='softmax')(h)

        # put it together:
        self._model = k.models.Model(all_input, output)
        self._model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
        self._pred_cache = {}  # caches input->output mapping
        self._hyp = model_hyp
        self._time_dim = time_dim


class RecModel(SimulatorModel):
    """
        Trainable recommender model that takes member state as input and output probability of impression.
    """

    # todo: consider doing version that explicitly accounts for depleted items

    def process_features(self, simulation: pd.DataFrame, NI: int) \
            -> (sparse.csr_matrix, sparse.csr_matrix):
        """
            Returns features for training a model: matrices of states and actions.
            @param simulation: dataset from which to derive features
            @param NI: number of items
            @return: matrices of states and actions. `state_history` is a list of NU binary arrays, each of size (NV_u, NI)
            where NV_u is num visits user u made. `action history` is a list of NU int arrays, each of size NV_u,
            indicating recommendation made.
        """
        print('\t decoding state:')
        S = self.decode_state(simulation['state'].values, NI)
        print('\t one-hot encoding actions:')
        A = utils.get_dummies_int(simulation['action'], NI, dense=0)
        S_ = S.tocsr()
        A_ = A.tocsr()

        return S_, A_

    def fit(self, simulation: pd.DataFrame, NI: int, verbose: int = 0, 
               validation: pd.DataFrame = None, W_: np.ndarray = None):
        """
        Fit the model.
        @param simulation: unfeaturized dataset
        @param NI: number of items
        @param verbose: print internals
        @return: None
        """
        
        # process features:
        if verbose:
            print('fitting recommender model:')
            print('processing features:')
        
        S1, A = self.process_features(simulation, NI)
        
        if validation is not None:
            Sv, Av = self.process_features(validation, NI)
            val_dat_tuple = (Sv.todense(), Av.todense())
            stop_loss = 'val_loss'
        else:
            val_dat_tuple = None
            stop_loss = 'loss'

        if verbose:
            print('S1', S1)
            print('A', A)

        # next task is to train where target is consumed item
        if verbose: print('fitting model:')

        batch_size = self._hyp['batch_size']
        steps_per_epoch = int(np.ceil(S1.shape[0] / batch_size))  # number of batches in an epoch
        
        self._model.fit_generator(
            utils.generate_dense_arrays([S1, A], ['int64', 'int64'], batch_size, steps_per_epoch, W_=W_),
            steps_per_epoch=steps_per_epoch, verbose=1, epochs=self._hyp['max_epoch'],
            callbacks=[EarlyStopping(monitor=stop_loss, min_delta=self._hyp['min_delta'], patience=0, mode='min')],
            validation_data=val_dat_tuple)

    def predict(self, x):
        return self._model.predict(x)


class RecModelTime(SimulatorModelTime, RecModel):
    """
        Extends both the time-based simulator base model and the recommender model.
    """

    def process_features(self, simulation: pd.DataFrame, NI: int, sparse: bool = True) \
            -> (sparse.csr_matrix, sparse.csr_matrix):
        # process state, action arrays:
        S, A = super().process_features(simulation, NI)

        # process time as feature:
        print('processing time features (dim %i) for rec model...' % self._time_dim)
        T = self.process_time_feature(simulation.time)

        # create overall feature matrix:
        X = scipy.sparse.hstack((S, T)).tocsr()

        return X, A

    def process_time_feature(self, time: np.ndarray, create_bins: bool = True) -> sparse.csr_matrix:
        """
            Process an array of time stamps into sparse matrix binned times.
            @param time: array of timestamps
            @param create_bins: if false, rely on self._time_bins binning to do featurization
            @return: matrix of binned times
        """
        if create_bins: self._time_bins = np.arange(1 + self._time_dim)
        binned = pd.cut(time, self._time_bins, include_lowest=True)
        return scipy.sparse.csr_matrix(pd.get_dummies(binned, sparse=True).to_numpy())

    def save(self, filepath: str):
        super().save(filepath)
        np.save(join(filepath, 'time_bins'), self._time_bins)

    def load(self, filepath: str):
        # supports prediction but not training from saved model:
        super().load(filepath)
        self._time_bins = np.load(join(filepath, 'time_bins.npy'))


class UserModel(SimulatorModel):
    """
        Model of user streams as a function of user state.
    """

    def process_features(self, simulation: pd.DataFrame, NI: int) -> (sparse.csr_matrix, sparse.csr_matrix, np.ndarray):
        """
        Derives features for training a user model.
        @param simulation: unfeaturized dataset
        @param NI: number of items
        @return: states, actions, and rewards
        """
        succ = simulation[simulation['reward'] > 0]
        S = self.decode_state(succ['state'].values, NI)
        A = utils.get_dummies_int(succ['action'], NI, dense=0)
        S_ = S.tocsr()
        A_ = A.tocsr()
        return S_, A_, succ['reward']

    def fit(self, simulation: pd.DataFrame, rec_pred: k.Model, NI: int, ips: int = 1, verbose: int = 0,
               W_: np.ndarray = None):
        # state_history: list of NU binary arrays, each of size (NV_u, NI) where NV_u is num visits user u made
        # reward history: list of NU float arrays, each of size NV_u
        # uses estimated rec model pr as IPS for user model training

        print('fitting user model:')

        S1, A, R = self.process_features(simulation, NI)
        N, D = S1.shape

        if verbose:
            print('S1', S1)
            print('A', A)

        if ips:
            # next task is to train where target is consumed item
            assert rec_pred.shape[0] == S1.shape[0], 'need equal num of propensity score weights and contexts'
            ws = np.array(
                [rec_pred[i, j] for i, j in enumerate(A.argmax(axis=1))])  # pull out weights of logged actions
            ws = ws.ravel()
            nws = N * ws / ws.sum()
        print('train ips =', ips)
        if not (ips): nws = None
        if verbose:
            print('ws', ws)
            print('nws', nws)
            print('rec_pred', rec_pred)
            print('shapes of A, rec_pred', A.shape, rec_pred.shape)
            
        if W_ is None:
            pass

        batch_size = self._hyp['batch_size']
        steps_per_epoch = int(np.ceil(
            S1.shape[0] / batch_size))  # number of batches in an epoch (last one likely to be a smaller batch of data)

        self._model.fit_generator(
            utils.generate_dense_arrays([S1, A], ['int64', 'int64'], batch_size, steps_per_epoch, W_=W_), \
            steps_per_epoch=steps_per_epoch,
            verbose=1, epochs=self._hyp['max_epoch'], \
            callbacks=[EarlyStopping(monitor='loss', min_delta=self._hyp['min_delta'], patience=0, mode='min')])


class UserModelPoint(SimulatorModel):
    def process_features(self, simulation, NI):
        # returns features for training a model: np.arrays of states and actions where reward>0
        S = self.decode_state(simulation['state'].values, NI)
        A = utils.get_dummies_int(simulation['action'], NI, dense=0)
        R = utils.get_dummies_int(np.minimum(1, simulation['reward']), 2, dense=0)
        S_ = S.tocsr()
        A_ = A.tocsr()
        R_ = R.tocsr()
        return scipy.sparse.hstack((S_, A_)), R_, simulation['reward']

    def fit(self, simulation: pd.DataFrame, rec_model: RecModel, NI: int, hyp: Dict, ips: int = 1, verbose: int = 0):
        """
        Fit the user model.
        @param simulation: unfeaturized dataset
        @param rec_model: a trained recommender model (used for IPS weights)
        @param NI: number of items
        @param hyp: hyperparameters
        @param ips: reweight during training using IPS
        @param verbose: print internals
        @return: None
        """
        print('fitting user model:')

        S1, A, R = self.process_features(simulation, NI)

        if verbose:
            print('S1', S1)
            print('A', A)

        # next task is to train where target is consumed item
        batch_size = hyp['simulator_model_batch_size']
        steps_per_epoch = int(np.ceil(
            S1.shape[0] / batch_size))  # number of batches in an epoch (last one likely to be a smaller batch of data)

        self._model.fit_generator(
            utils.generate_dense_arrays([S1, A], ['int64', 'int64'], batch_size, steps_per_epoch, verbose=0), \
            steps_per_epoch=steps_per_epoch,
            verbose=1, epochs=hyp['max_epoch_user_model'], \
            callbacks=[EarlyStopping(monitor='loss', min_delta=0.001, patience=0, mode='min')])


class UserModelBinary(UserModel):
    """
         A user model in which each item has a separate Bernoulli probability of being streamed (as opposed to categorical).
    """

    def _masked_loss_function(self, y_true: tf.Tensor, y_pred: tf.Tensor, mask_value: int = 0) -> tf.Tensor:
        """
        Masked loss function which calculates the binary cross entropy between predicted and true values
        under censoring of some elements in the tensor.
        @param y_true: +1 or -1 ground truth
        @param y_pred: 0 or 1 prediction
        @param mask_value: value that indicates censoring
        @return: binary cross entropy
        """
        mask = kb.cast(kb.not_equal(y_true, mask_value), kb.floatx())
        return kb.binary_crossentropy((0.5 + 0.5 * y_true) * mask, y_pred * mask)

    def _build_output_model(self, state_model: k.Model, ndims_out: int) -> k.Model:
        state_model.add(k.layers.Dense(ndims_out, activation='sigmoid'))
        state_model.compile(loss=self._masked_loss_function, optimizer='adagrad', metrics=['accuracy'])
        return state_model

    def process_features(self, simulation: pd.DataFrame, NI: int) -> (sparse.csr_matrix, sparse.csr_matrix, np.ndarray):
        # returns features for training a binary user model
        print('\t decoding state:')
        S = self.decode_state(simulation['state'].values, NI)
        print('\t one-hot encoding actions:')
        R = np.minimum(1, simulation['reward'].values)
        A = utils.get_dummies_int(simulation['action'], NI, dat=(2 * R - 1),
                                  dense=0)  # true negatives encoded as -1, positives as +1, missing as 0
        S_ = S.tocsr()
        A_ = A.tocsr()
        print('\t en/decoding done.')
        return S_, A_, R

    def select(self, user_state: np.ndarray, xs: np.ndarray, hyp: Dict, stg: Dict, verbose: int = 0) -> np.ndarray:
        """
            Draw binary items from model predictive distribution
            @param user_state: array representing bag of words of items consumed
            @param xs: item IDs for which to take a sample (the rest are censored)
            @param hyp: hyperparameters
            @param stg: general settings
            @param verbose: print internals around caching
            @return: sample of item IDs
        """
        key = utils.bow(user_state)
        if key in self._pred_cache:
            p = self._pred_cache[key]
            if verbose: print('cache hit in simulator model')
        else:
            if verbose: print('cache miss in simulator model')
            p = self._model.predict(np.array([user_state])).ravel()  # inefficient
            # store in cache
            self._pred_cache[key] = p
        p1 = np.zeros_like(p)
        p1[xs] = p[xs]
        val, = np.where(np.random.binomial(1, p1) > 0)
        return val

    def load(self, filepath: str):
        # supports prediction but not training from saved model:
        self._model = k.models.load_model(filepath, custom_objects={"kb": kb}, compile=False)


class NMFRec(RecModel):
    def __init__(self, NI: int, model_hyp: Dict):
        self._hyp = model_hyp
        self._NI = NI
        self._model = None
        self._pred_cache = {}  # caches input->output mapping
        
    def process_features_train(self, dat, NI):
        n_rows = dat.user_id.max()+1
        return sparse.csr_matrix((np.ones(dat.shape[0]),
                              (dat.user_id,
                              dat.action)),
                                shape=(n_rows,NI))
    
    def process_features(self, dat, NI):
        return utils.inv_bow_all(list(dat.state.values), NI)

    def fit(self, simulation: pd.DataFrame, NI: int, verbose: int = 0, 
               validation: pd.DataFrame = None, W_: np.ndarray = None):
        """
        Fit the model.
        @param simulation: unfeaturized dataset
        @param NI: number of items
        @param verbose: print internals
        @return: None
        """
        
        # process features:
        if verbose:
            print('fitting recommender model:')
            print('processing features:')
        
        X = self.process_features_train(simulation, NI)
        
        self._model = NMF(**self._hyp['init_args'])
        self._model.fit(X)
        
    def predict(self, x):
        m,n = x.shape
        if self._model is not None:
            # get factor loadings for given users (may have repeats in different rows):
            w = self._model.transform(x)
            # use factor loadings to predict score for items:
            s = w @ self._model.components_
            # use temperature to shape scores:
            T = self._hyp['temperature']
            p = s**T
            p_denom = p.sum(axis=1)
            zs, = np.where(np.abs(p_denom) < 1e-9)
            # for rows with zero output, use uniform dist:
            p[zs,:] = 1e-3
            p_denom[zs] = 1e-3*n
            p /= p_denom[:,np.newaxis]
        else:
            p = np.ones((m,n))/n
            
        return p

        
        
def process_features_static(simulation_original: pd.DataFrame,
                            fixed_params: Tuple[int, int, float, float, float, float], **kwargs) \
        -> sparse.csr_matrix:
    """
        Returns featurization for visit model given a dataset.
        @param simulation_original: dataset
        @param fixed_params: tuple of numbers for featurization
        @param kwargs: dictionary of settings
        @return: sparse matrix of features
    """
    # pull out timestamps and users:
    time_shift = fixed_params[5]
    T = simulation_original['time'].values
    if time_shift < 0:
        T += np.random.uniform(time_shift, 0, size=T.shape[0])
    U = simulation_original['user_id'].values

    return process_features_static_points(simulation_original, T, U, fixed_params[:5], **kwargs)


def process_features_static_binary(sim: pd.DataFrame, fixed_params: Tuple[int, int, float, float, float, float],
                                   **kwargs) \
        -> sparse.csr_matrix:
    """
        Returns featurization and target class for visit model given a dataset.
        Approach taken is to ensure that we have all the positive examples with a uniform random shift 
        before the visit combined with an approximately equal number of negative examples from 
        sampling the timeline uniformly.
        @param sim: dataset
        @param fixed_params: tuple of numbers for featurization
        @param kwargs: dictionary of settings
        @return: sparse matrix of features, binary target class Y
    """
    time_shift = fixed_params[5]

    # pull out timestamps and users of positive examples (visits):
    N_pos = sim.shape[0]
    NU = sim.user_id.nunique()
    T_pos = sim['time'].values
    T_pos += np.random.uniform(time_shift, 0,
                               size=N_pos)  # add random time shift, limited ensure examples are still positive
    U_pos = sim['user_id'].values

    # now add an equal number of random time sampled examples (some of which will be negative):
    N_neg = 1000000
    rnd_ts = np.random.uniform(sim.time.min(), sim.time.max(), size=N_neg)
    rnd_us = np.random.choice(sim.user_id.unique(), replace=True, size=N_neg)
    # annotate time points that overlap with a visit within the delta_time margin as positive:
    query_points = pd.DataFrame({'time': rnd_ts, 'user_id': rnd_us, 'query_point': True})
    sim_query = sim.copy()
    sim_query['query_point'] = False
    sim_query = sim_query.append(query_points, ignore_index=True, sort=False)
    # for each user, go through and annotate the rows as either query points of actual visits
    query_points = []  # this will hold [(time, user_id, is_visit)] array
    # following loop has O(U * T^2) time performance where U = number of users, T = number of events per user 
    # (this could be improved).
    for u, g in sim_query.groupby('user_id'):
        visit_times = g[~g['query_point']].time
        query_times = g[g['query_point']].time
        # provide a dummy visit time if there are no visits for this user:
        if visit_times.shape[0] == 0: visit_times = np.inf * np.ones(1)
        n = query_times.shape[0]
        if n > 0:
            time_diff = visit_times[np.newaxis, :] - query_times[:, np.newaxis]
            time_diff[time_diff < 0] = np.inf  # mark negative time differences as invalid
            query_points += list(zip(query_times, n * [u], (time_diff.argmin(axis=1) < -time_shift).astype('int16')))
    Q = np.array(query_points)
    # combine positive and negative time points:
    T = Q[:, 0]  # np.concatenate((T_pos, Q[:,0]))
    U = Q[:, 1]  # np.concatenate((U_pos, Q[:,1]))
    Y = Q[:, 2:3]  # np.concatenate((np.ones(N_pos), Q[:,2]))[:,np.newaxis]
    Y = Y[T >= 0, :]
    X = process_features_static_points(sim, T[T >= 0], U[T >= 0], fixed_params[:5], **kwargs)

    return sparse.hstack((X, Y)).tocsr()


def process_features_static_wrapper(simulation: pd.DataFrame, T: np.ndarray, U: np.ndarray,
                                    fixed_params: Tuple[int, int, float, float, float], **kwargs) \
        -> sparse.csr_matrix:
    """
            Returns featurization for visit model given a dataset.
            @param simulation: dataset
            @param fixed_params: tuple of numbers for featurization
            @param kwargs: dictionary of settings
            @return: sparse matrix of features
    """
    # ps = list(fixed_params)
    # N = ps[0]
    # fixed_params_subset = tuple(ps[1:])
    # U = np.random.choice(simulation.user_id.unique(), size=N)
    # T = np.random.uniform(0, simulation.time.max(), size=N)

    return process_features_static_points(simulation, T, U, fixed_params, **kwargs)


def process_features_static_points(simulation_original: pd.DataFrame, T: np.ndarray, U: np.ndarray,
                                   fixed_params: Tuple[int, int, float, float, float],
                                   init_user_states: np.ndarray = None, verbose: int = 0) -> sparse.csr_matrix:
    """
        Returns featurization for visit model given array of query times T (there may or may not be an event at time T_i).
        Features take the form:
        # [normalized time, state_history_vector, self-excitory history features]
        # where self-excitory history features are window_size*[delta t, item_embedding]
        # (looking at most recent #window_size events)
        Static method is pickle-able.
        @param simulation_original: dataset
        @param T: query times
        @param U: query users
        @param fixed_params: tuple of numbers for featurization
        @param init_user_states: initial state for each user
        @param verbose: print internals
        @return: sparse matrix
    """

    NI, window_size, Tevmean, Tevmin, Tevmax = fixed_params
    
    #  will be annotating the data frame, so a copy is made:
    simulation = simulation_original.copy()

    if len(T) == 0:
        # no query times so just return empty sparse matrix of correct size:
        return sparse.csr_matrix((0, 1 + NI + (NI + 1) * window_size), dtype=float)

    # step 1. annotate table with normalized time of each event
    def norm_time(t):
        if len(t) > 0: return 2 * (((t - Tevmean) / (Tevmax - Tevmin)))
        return np.zeros(0)

    simulation['time_normed'] = norm_time(simulation['time'].values)

    # step 2. convert sample points to normalized time also
    Tnorm = norm_time(T)
    #assert np.all(np.abs(Tnorm) <= 1.2), list(
    #    zip(list(T), list(Tnorm)))  # allow 20% deviation outside training range for extrapolation but no more

    # step 3. ensure simulation table is sorted first by user id then by time
    simulation.sort_values(by=['user_id', 'time_normed'], inplace=True)

    # step 4. at the sample time points Tnorm, look up the user's state and recent history of consumed items
    exhist = []  # self-excitation history, sparse representation [(dat,row,col)]
    stat = []  # state of user at sample time point, sparse representation [(dat,row,col)]
    # todo: decide what history should be when there is no user-specific history
    init_state = {}  # maps user id to initial state until first stream
    for u in np.unique(U):
        if init_user_states is None:
            # pull out initial state from simulation dataset
            first_state_str = simulation[simulation['user_id'] == u].sort_values('time').state.iloc[0]
            first_state = utils.inv_bow_sp(first_state_str, NI)
            init_state[u] = first_state
        else:
            # use init_user_state provided
            if isinstance(init_user_states, np.ndarray) or isinstance(init_user_states, sparse.csr_matrix):
                init_state[u] = init_user_states[u, :].tocoo().col
            elif isinstance(init_user_states, dict):
                init_state[u] = init_user_states[u]
            else:
                raise Exception('unhandled type',type(init_user_states))
                
    #print('init_state ...',init_state,U)
                
    # preprocess history of impressions to efficiently pull out recent history for any proposal time:
    sorted_sim = simulation[simulation['reward'] > 0].sort_values(by=['time_normed'], ascending=False)
    user_recents = sorted_sim.groupby('user_id', sort=False)
    for itr, t, u in zip(itertools.count(), Tnorm, U):
        # TODO: this line repeats a lot of computation, replace with more efficient version:
        user_streams = user_recents.get_group(u) if u in user_recents.groups else pd.DataFrame({'time_normed': []})
        succ = user_streams[user_streams['time_normed'] < t]
        pad_hist_size = max(window_size - succ.shape[0], 0)  # how much to pad history to make it window_size
        if succ.shape[0] > 0:
            last_stream = succ.iloc[0, :]
            # update last state based on stream and decode to bow representation:
            last_state_str = last_stream['state']
            last_state = utils.inv_bow_sp(last_state_str, NI)
            #last_state.append(last_stream['action']) # NOTE: NOT UPDATING STATE!!!
            NLS = len(last_state)  # num non-zero entries for this row
            stat += list(zip(np.ones(NLS), np.repeat(itr, NLS), last_state))  # state at time t as sparse representation
            h = succ.iloc[:window_size, :]
            assert np.all(t - h['time_normed']), 'error in processing historical events for hawkes features'
            if verbose:
                print('h action', h['action'].values)
                print('h encoding', utils.get_dummies_int(h['action'], NI))
            time_diff = t - h['time_normed']
            # pad, if necessary:
            time_diff = np.concatenate((time_diff, SETTINGS.stg['INF_TIME'] * np.ones(pad_hist_size)))
            window_offsets = np.arange(0, (NI + 1) * window_size, NI + 1)  # starting pos for each block
            exhist += list(zip(time_diff, np.repeat(itr, window_size), window_offsets))
            LA = len(h['action'])
            if LA > 0:  # otherwise just leave state representation to zero vector
                exhist += list(zip(np.ones(LA), np.repeat(itr, LA), 1 + window_offsets[:LA] + h['action'].values))
        else:
            # need to look ahead to first impression (which is in the future) to pull out state:
            # (it will be the right starting state because it will be before any stream events happen)
            LIS = len(init_state[u])
            stat += list(zip(np.ones(LIS), np.repeat(itr, LIS), init_state[u]))  # state at time t

    # final features
    S = utils.make_csr(stat, (len(T), NI))
    EH_ncols = (NI + 1) * window_size
    EH = utils.make_csr(exhist, (len(T), EH_ncols))
    if verbose:
        print('tnorm', Tnorm, Tnorm.shape)
        print('state', S, S.shape)
        print('effective history', EH, EH.shape)
    X = sparse.hstack([Tnorm[:, np.newaxis], S, EH]).tocsr()
    return X


class SelfExcitingLayer(k.layers.Layer):
    """
        Layer that implements Hawkes process intensity as a function of recency and count of historical events.
    """

    # todo: input is sparse, can take advantage in this class
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SelfExcitingLayer, self).__init__(**kwargs)

    def build(self, input_shape: Tuple) -> k.layers.Layer:
        """
            Intensity is a_i * exp{ b_i( T - t) }.
            Input is: [t, sparse_one_hot_item_encoding], i.e., NI+1 dimensional sparse vec.
            @param input_shape:
            @return: layer
        """
        self.a = self.add_weight(name='a',
                                 shape=(input_shape[1] - 1, self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='b',
                                 shape=(input_shape[1] - 1, self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        super(SelfExcitingLayer, self).build(input_shape)

    def call(self, x):
        t = x[:, :1]  # time diff.
        v = x[:, 1:]  # title encoding
        return kb.dot(v, kb.exp(self.a)) * kb.exp(- kb.dot(v, kb.exp(self.b)) * t)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # "returns a dictionary containing the configuration of the layer"
        return {'output_dim': 1}  # TODO: avoid hard coding output dimension
    
    
class SelfExcitingLayerOffset(k.layers.Layer):
    """
        Layer that implements Hawkes process intensity with base rate as a function of recency and count of historical events.
    """

    # todo: input is sparse, can take advantage in this class
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SelfExcitingLayerOffset, self).__init__(**kwargs)

    def build(self, input_shape: Tuple) -> k.layers.Layer:
        """
            Intensity is c + a_i * exp{ b_i( T - t) }.
            Input is: [t, sparse_one_hot_item_encoding], i.e., NI+1 dimensional sparse vec.
            @param input_shape:
            @return: layer
        """
        self.a = self.add_weight(name='a',
                                 shape=(input_shape[1] - 1, self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='b',
                                 shape=(input_shape[1] - 1, self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        self.c = self.add_weight(name='c',
                                 shape=(1, self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        super(SelfExcitingLayerOffset, self).build(input_shape)

    def call(self, x):
        t = x[:, :1]  # time diff.
        v = x[:, 1:]  # title encoding
        return kb.exp(self.c) + kb.dot(v, kb.exp(self.a)) * kb.exp(- kb.dot(v, kb.exp(self.b)) * t)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # "returns a dictionary containing the configuration of the layer"
        return {'output_dim': 1}  # TODO: avoid hard coding output dimension



class SelfExcitingLayerVector(k.layers.Layer):
    """
        Layer that implements Hawkes process intensity as a function of recency and count of historical events.
    """

    # todo: input is sparse, can take advantage in this class
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SelfExcitingLayer, self).__init__(**kwargs)

    def build(self, input_shape: Tuple) -> k.layers.Layer:
        """
            Intensity is a_i * exp{ b_i( T - t) }.
            Input is: [t, sparse_one_hot_item_encoding], i.e., NI+1 dimensional sparse vec.
            @param input_shape:
            @return: layer
        """
        self.a = self.add_weight(name='a',
                                 shape=(input_shape[1] - 1, self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='b',
                                 shape=(input_shape[1] - 1, self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        super(SelfExcitingLayer, self).build(input_shape)

    def call(self, x):
        t = x[:, :1]  # time diff.
        v = x[:, 1:]  # title encoding
        return kb.dot(v, kb.exp(self.a)) * kb.exp(- kb.dot(v, kb.exp(self.b)) * t)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # "returns a dictionary containing the configuration of the layer"
        return {'output_dim': 1}  # TODO: avoid hard coding output dimension

class VisitModel(SimulatorModel):
    """
        Model of inhomogeneous Poisson process intensity as a function of time and previous realized events.
    """

    def __init__(self, NI: int, NU: int, model_hyp: Dict):
        """
            There are three parts to visit model [all intensities must be non-negative]:
                lambda_0(t): global base rate, depends on time
                lambda_u(s): user base rate, depends on user state
                lambda_e(t,H): self-excitory rate, depends on time t and user history up to t
            @param NI: number of items
            @param NU: number of users
            @param model_hyp: value for `window_size` key describes how many events in personal history to look back
        """
        window_size = model_hyp['window_size']

        # define inputs and positions of is_event, time, user activity, state, history:
        time_start_indx = 0
        state_start_indx = time_start_indx + 1
        state_end_indx = NI + state_start_indx
        history_size = NI + 1  # dimensionality of a single history block
        history_start_indx = state_end_indx
        history_end_indx = history_start_indx + window_size * history_size
        inputs_all = k.layers.Input(shape=(history_end_indx,))
        print('history_end_indx', history_end_indx)

        # global base rate intensity:
        time_hyp = model_hyp['time_model']
        time_input = inputs_all[:, time_start_indx:state_start_indx]
        prev_layer = time_input
        # note: no dropout on input since it only has 1 dimension
        for layer_id in range(time_hyp['n_hidden_layers']):
            prev_layer = k.layers.Dense(time_hyp['n_nodes'][layer_id], \
                                        activation=time_hyp['hidden_activation'])(prev_layer)
            if time_hyp['dropout_rate'] > 0:
                prev_layer = k.layers.Dropout(time_hyp['dropout_rate'])(prev_layer)

        lambda_0 = k.layers.Dense(1, activation="exponential")(prev_layer)
        self._lambda_0_model = k.models.Model(inputs_all, lambda_0)

        # user history intensity:
        state_input = inputs_all[:, state_start_indx:state_end_indx]
        state_hyp = model_hyp['state_model']
        prev_layer = self._get_init_state_layer(state_input, inputs_all)
        if state_hyp['dropout_rate'] > 0:
            prev_layer = k.layers.Dropout(state_hyp['dropout_rate'])(prev_layer)
        for layer_id in range(state_hyp['n_hidden_layers']):
            prev_layer = k.layers.Dense(state_hyp['n_nodes'][layer_id], \
                                        activation=state_hyp['hidden_activation'])(prev_layer)
        lambda_u = k.layers.Dense(1, activation="exponential")(prev_layer)
        self._lambda_u_model = k.models.Model(inputs_all, lambda_u)

        # self-excitory intensity: (a_i * exp{b_i(T - t_e)})
        # todo: consider adding self-inhibitory component that decreases intensity
        history_input = inputs_all[:, history_start_indx:history_end_indx]

        def divide_history_blocks():
            for i in range(0, history_end_indx - history_size, history_size):
                yield history_input[:, i:i + history_size]

        se_model = k.models.Sequential()
        se_model.add(SelfExcitingLayer(1))
        print([b for b in divide_history_blocks()])
        lambda_hs = [se_model(b) for b in divide_history_blocks()]
        lambda_e = k.layers.Add()(lambda_hs)
        self._lambda_e_model = k.models.Model(inputs_all, lambda_e)

        # combine intensities into one
        lambda_total = k.layers.Add()([lambda_0, lambda_u, lambda_e])
        #lambda_total = k.layers.Add()([lambda_0, lambda_e])

        model = k.models.Model(inputs_all, lambda_total)
        # delay compiling until we know N (it is needed to calculate the stochastic loss)
        self._model = model

        self._window_size = window_size
        self._history_size = history_size
        self._NI = NI

        # IMPRESSION MODEL
        # ----------------
        # Poisson regression model to estimate lambda parameter that determines number of impressions per visit
        # SIMPLE FOR NOW: one lambda for all users
        self._lambda_impression = 1.0

        self._pred_cache = {}  # dict cache
        
        self._history_log = [] # for storing logs during training
        

    def _get_init_state_layer(self, state_input, inputs_all):
        norm_state = state_input  # normalize_layer()(state_input)
        return norm_state

    def get_time_params(self):
        return self._T_mean, self._T_min, self._T_max

    def norm_time(t):
        raise Exception('need to run fit() first')

    def inv_norm_time(t):
        raise Exception('need to run fit() first')

    def norm_time(self, t):
        return 2 * (t - self._T_mean) / (self._T_max - self._T_min)

    def inv_norm_time(self, t):
        return t * 0.5 * (self._T_max - self._T_min) + self._T_mean

    def time_scale(self) -> float:
        # time_scale: by how much have we streched/shrunk the x-axis of intensity function with normalization?
        Tnorm_range = self._Tnorm_max - self._Tnorm
        return Tnorm_range / (self._T_max - self._T_min)
    
    def fit(self, simulation: pd.DataFrame, n_impressions_by_visit: np.ndarray, hyp: Dict, stg: Dict, verbose: int = 0,
            parallel: bool = True):
        """
        Prepare data and fit the model using tensorflow/keras.
        @param simulation: unfeaturized dataset
        @param n_impressions_by_visit: number of impressions for each visit
        @param hyp: hyperparameters
        @param stg: general settings
        @param verbose: print internals if > 0
        @param parallel: use parallel processing to featurize dataset
        @return: None
        """

        # model is expecting input for each event:
        # (time, user_state, history)

        print('fitting visit model:')

        # pull out features for the event times and users we observe
        NU = stg['NU']  # number of users
        NI = stg['NI']  # number of items

        # remember various aspects of the dataset, including the time of first and last event:
        # need to first define normalized and unnormalized time:
        Tev = simulation['time']
        self._T_min = Tev.min()
        self._T_max = Tev.max()
        self._T_mean = Tev.mean()
        Tnorm = self.norm_time(Tev.values)
        self._Tnorm_min = Tnorm.min()
        self._Tnorm_max = Tnorm.max()
        self._Tnorm_mean = Tnorm.mean()
        Tnorm_range = self._Tnorm_max - self._Tnorm_min
        self._simulation = simulation
        self.time_scale = Tnorm_range / (self._T_max - self._T_min)

        X, Y = self.process_features(simulation, Tev, Tnorm_range, NI, parallel)
        N = X.shape[0]

        if verbose:
            print('X', list(X))
           
        def count_error(y_true, y_pred):
            N_est = NU * y_pred * Tnorm_range
            return kb.abs(N - kb.mean(N_est)) / N

        # create validation data (if not using event_sample method then regularizer leaks info from validation):
        batch_size = hyp['batch_size']
        train_val_split = hyp['train_val_split']

        size_rnd_points = int(N / 2)
        steps_per_epoch_rnd = int(np.ceil(size_rnd_points / batch_size))
        X_rnd = self._create_regularizer_features(simulation, \
                                                  np.random.uniform(self._T_min, self._T_max, size=size_rnd_points), \
                                                  simulation['user_id'].sample(n=size_rnd_points), \
                                                  hyp, stg)
        validation_rnd_points = utils.generate_dense_arrays([X_rnd, sparse.csr_matrix(np.ones((size_rnd_points, 1)))], ['float64', 'float64'], batch_size, steps_per_epoch_rnd)

        X_reg_ = self._create_regularizer_features(simulation, \
                                                  simulation['time'].values, \
                                                  simulation['user_id'].values, \
                                                  hyp, stg)
        X_reg = X_reg_.todense() #None if (hyp['visit_loss_method']=='event_sample') else X_reg_.todense()
        metrics = ['accuracy', count_error]
        self._model.compile(loss=self.loss(*loss_args), \
                            optimizer='rmsprop', metrics=metrics)  # adagrad


        def generator_ws(Xs, Ts, batch_size: int, steps_per_epoch: int) -> Tuple:
            N, K1 = Xs[0].shape
            assert np.all([X.shape[0] == N for X in Xs]), ','.join([str(X.shape[0]) for X in Xs])
            assert len(Xs) == len(Ts)
            while True:
                ns = np.arange(N, dtype='int64')
                shuffle_ns = np.random.permutation(ns)
                sum_W = 0 # track total reward for epoch
                for b in range(steps_per_epoch):
                    # get batch of random indices
                    shuffle_ns_batch = shuffle_ns[b * batch_size:(b + 1) * batch_size]
                    Xs_dense = [X[shuffle_ns_batch, :].toarray().astype(T) for (X, T) in zip(Xs, Ts)]  # 'int64'
                    yield tuple(Xs_dense)
        
        steps_per_epoch = int(np.ceil(N / batch_size))
        print('X.shape,type(X)', X.shape, type(X))
        log_history = self._model.fit(
            generator_ws([X, Y], ['float64', 'float64'], batch_size, steps_per_epoch), \
            steps_per_epoch=steps_per_epoch,
            verbose=1, epochs=hyp['max_epoch'],
            validation_data=validation_rnd_points, \
            validation_steps=steps_per_epoch_rnd,
            callbacks=[EarlyStopping(monitor='val_count_error', min_delta=hyp['min_delta'], patience=hyp['patience'], mode='min')]) 

        # fit impression count model:
        # simple for now: one lambda for all users
        self._lambda_impression = n_impressions_by_visit.mean()
        return log_history


    def process_features(self, sim: pd.DataFrame, tev: np.ndarray, tnorm_range: float, NI: int, parallel: bool) \
            -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        fixed_params = (NI, self._window_size, tev.mean(), tev.min(), tev.max(), 0)
        if parallel:
            X = utils.parallelize_fnc_groups(process_features_static, sim, \
                                             fixed_params, \
                                             'user_id', int(int(os.environ['NUMEXPR_MAX_THREADS']) / 2), \
                                             concat_mode='sparse')
        else:
            X = process_features_static(sim, fixed_params)

        N, D = X.shape
        n_events_per_user = sim.groupby('user_id').action.count()
        seq_user_id = pd.DataFrame({'user_id':np.sort(n_events_per_user.index), 'seq_id':np.arange(n_events_per_user.shape[0],dtype='int32')})
        event_scale = n_events_per_user[
                          sim.user_id.values].values / tnorm_range  # equivalent to homogeneous Poisson rate for the
        
        # use if 'event_sample':
        Y = sparse.csr_matrix(np.ones((N, 1)))
        
        Z = sparse.csr_matrix((event_scale, (np.arange(N, dtype='int64'), np.zeros(N, dtype='int64'))), shape=(N, 1))

        return X, Y

    
    def _create_regularizer_features(self, sim: pd.DataFrame, Teven: np.ndarray, Ueven: np.ndarray, hyp: Dict,
                                     stg: Dict,
                                     include_events: bool = False, verbose: int = 0) -> sparse.csr_matrix:
        """
            Calculate featurized data depending on which regularizer / normalizer is being used in the loss function.
            @param sim: dataset
            @param Teven: time of each event
            @param Ueven: user corresponding to each event
            @param hyp: hyperparameters
            @param stg: general settings
            @param include_events: whether or not to include the events themselves (or only use query points)
            @param verbose: print internals if > 0
            @return: sparse matrix of features for visit model
        """
        NU = stg['NU']
        NI = stg['NI']
        unique_users = sim.user_id.unique() # np.arange(NU, dtype='int16')
        assert NU == unique_users.shape[0], '%i %i' % (NU, unique_users.shape[0])
        loss_method = hyp['visit_loss_method']

        if loss_method == 'full': # or loss_method == 'event_sample':
            grid_size = hyp['lambda_grid_factor'] * (self._T_max - self._T_min)
            ts = np.arange(self._T_min, self._T_max + grid_size, grid_size)
            T = np.repeat([ts], NU, axis=0).flatten()
            U = np.repeat(unique_users, len(ts))
        elif loss_method == 'point_sample':
            # sub-sample users:
            NUsamp = int(NU * hyp['user_point_sample_factor'])
            us = np.random.choice(np.arange(NU), size=NUsamp, replace=False)
            ts = np.random.uniform(self._T_min, self._T_max, size=(NUsamp, hyp['time_sample_size']))
            U = np.repeat(us, hyp['time_sample_size'])  # each user has time_sample_size time points in regularizer
            T = ts.flatten()
        elif loss_method == 'event_sample' or loss_method == 'event_thin':
            # no featurization needed (other than include events)
            T = Teven
            U = Ueven
            #return None
        elif loss_method == 'event_score':
            T = np.concatenate([[self._T_min, self._T_max]] * sim.shape[0], axis=0)
            U = np.repeat(sim.user_id.values, 2, axis=0)
        else:
            raise NotImplementedError

        if include_events:
            T = np.concatenate([Teven, T])
            U = np.concatenate([Ueven, U])

        if verbose:
            print('T', list(T))
            print('U', list(U))

        fixed_params = NI, hyp['window_size'], Teven.mean(), Teven.min(), Teven.max()
        X = process_features_static_points(sim, T, U, fixed_params)
        
        return X

    

    def loss(self, X_regularizer: tf.Tensor, Twidth: float, N: int, NU: int, NI: int, hyp: Dict, log_file) -> Callable:
        """
            Loss function for the Poisson process. Includes several different options based on how to approximate the
            regularizer / normalizer in the Poisson process loss function.
            @param X_regularizer: data for regularization term in loss function
            @param Twidth: width of time window
            @param N: number of observed events
            @param NU: number of users
            @param NI: number of items
            @param hyp: hyperparameters
            @return: loss function
        """

        def intensity_regularizer(X_reg):
            # returns Tensor object that represents approx. integral of intensity function across all users
            # calculate regularizer as closely as possible
            Y1 = self._model(X_reg)  # height of rectangles
            ysum = kb.sum(Y1)
            grid_width_n = Twidth * hyp['lambda_grid_factor']  # width of rectangles (in normalized time)
            return ysum * grid_width_n

        def full_loss(y_is_event, y_pred):
            # y_is_event: 1 for yes, 0 for no (used for normalization)
            reg = intensity_regularizer(X_regularizer) / N
            self._history_log.append({'reg':reg})
            return -kb.log(y_pred) + reg

        def event_sample_loss(y_is_event, y_pred):
            # y_is_event: 1 for yes, 0 for no (used for normalization)
            return -y_is_event * kb.log(y_pred) + NU * y_pred * Twidth / N  # + N*kb.log(NU)

        def event_thin_loss(y_hg, y_pred):
            # y_uniform: use normalization term if and only if sampled point lies below homogeneous intensity
            u = kb.random_uniform((hyp['batch_size'],))
            use_reg = 0.5 + 0.5 * kb.sign(y_hg - u * y_pred)
            return -kb.log(y_pred) + use_reg * NU * y_pred * Twidth / N  # + N*kb.log(NU)

        def point_sample_loss(y_is_event, y_pred):
            # y_is_event: 1 for yes, 0 for no (used for normalization)
            Y1 = self._model(X_regularizer)  # height of intensity
            ysum = kb.sum(Y1)
            denom = hyp['time_sample_size'] * hyp['user_point_sample_factor'] * N
            return -kb.log(y_pred) + (ysum * Twidth) / denom
        
        def score_loss(y_dat, y_pred):
            # Nu: number of total events for this user
            return -kb.log(y_pred)

        def loss_fn(*args):
            loss_method = hyp['visit_loss_method']
            if loss_method == 'full':  # full method of regularization (note: based on rectangle approximation)
                return full_loss(*args)
            if loss_method == 'event_sample':  # use the current event only in the regularizer
                return event_sample_loss(*args)
            if loss_method == 'event_thin':  # use the current event only in the regularizer
                return event_thin_loss(*args)
            if loss_method == 'point_sample':  # sample users and times in the regularizer
                return point_sample_loss(*args)
            if loss_method == 'event_score':  
                return score_loss(*args)
            raise NotImplementedError

        return loss_fn

    def save(self, filestem: str):
        """
            Save overall model as well as sub-models.
            @param filestem: pattern for saving files, e.g. "/sim/visit_%s.h5"
            @return: None
        """
        self._model.save_weights(filestem % 'model')
        self._lambda_0_model.save_weights(filestem % 'base')
        self._lambda_u_model.save_weights(filestem % 'state')
        self._lambda_e_model.save_weights(filestem % 'hawkes')
        np.save(filestem % 'time_stats',
                [self._T_min,
                 self._T_max,
                 self._T_mean,
                 self._Tnorm_min,
                 self._Tnorm_max])

    def load(self, filestem: str, training_dataframe: pd.DataFrame):
        """
            Load overall model as well as sub-models.
            @param filestem: pattern to load the required model files, e.g. "/sim/visit_%s.h5"
            @param training_dataframe: training data that was used to fit visit model
            @return: None
        """
        self._model.load_weights(filestem % 'model')
        self._lambda_0_model.load_weights(filestem % 'base')
        self._lambda_u_model.load_weights(filestem % 'state')
        self._lambda_e_model.load_weights(filestem % 'hawkes')
        t_stats = np.load((filestem % 'time_stats') + '.npy')
        self._T_min = t_stats[0]
        self._T_max = t_stats[1]
        self._T_mean = t_stats[2]
        self._Tnorm_min = t_stats[3]
        self._Tnorm_max = t_stats[4]
        Tnorm_range = self._Tnorm_max - self._Tnorm_min
        self.time_scale = Tnorm_range / (self._T_max - self._T_min)

        # need some other info to process features on-the-fly:
        self._simulation = training_dataframe



def preprocess_visit_dat(simulation: pd.DataFrame, cell_id: int) -> (pd.DataFrame, np.ndarray):
    """
        Preprocess observed visit data to aid simulating new visits.
        @param simulation: whole dataset
        @param cell_id: which cell to look at
        @return: subset of simulation related to visits of the particular cell and the impression count per visit
    """
    sim_cell = simulation[simulation['acnt.test_cell_nbr'] == cell_id]
    visit_groups = sim_cell.sort_values('reward', ascending=False).groupby(['user_id', 'time'], sort=False)
    visit_simulation = visit_groups.first().reset_index()
    n_impressions_by_visit = visit_groups.action.count()
    # if there is a streamed item from a visit, it will be the first item of each group
    # BIG ASSUMPTION: only one streamed item per visit. todo: fix.
    print('visit model: num. unique visits in training data', visit_groups.ngroups)
    return visit_simulation, n_impressions_by_visit


def fit_sim(simulation: pd.DataFrame, hyp: Dict, stg: Dict, fit_rec: bool = True, fit_user: bool = True,
            fit_visit: bool = True, rec_model: RecModel = None) -> (RecModel, UserModel, VisitModel):
    """
    Convenience function for fitting the recommender, user, and visit models from a dataset.
    @param simulation: the dataset
    @param hyp: hyperparameters
    @param stg: general settings for training and simulation
    @param fit_rec: fit the recommender model iff true
    @param fit_user: fit the user model iff true
    @param fit_visit: fit the visit model iff true
    @param rec_model: optional pre-fitted recommender model (used in IPS weights)
    @return: tuple of recommender, user, and visit model
    """
    # S: NS-NU-NI array representing history of states
    # initial_state: starting state NU-NI array
    user_model = UserModel(stg['NI'], stg['NI'], hyp['user_model_hyp'])
    if rec_model is None: rec_model = RecModel(stg['NI'], stg['NI'], hyp['rec_model_hyp'])
    visit_model = VisitModel(stg['NI'], stg['NU'], hyp['visit_model_hyp'])

    if fit_rec: rec_model.fit(simulation, stg[
        'NI'])  # rec doesn't know about clocks, clearly not true for feature processing e.g. in UPM BB
    if fit_user: user_model.fit(simulation, rec_model, stg['NI'], ips=hyp['user_model_hyp']['train_ips'])

    if fit_visit:
        # visit model requires us to group impressions of the same visit,
        # then estimate an impression rate parameter for each user.
        visit_groups = simulation.sort_values('reward', ascending=False).groupby(['user_id', 'time'], sort=False)
        visit_simulation = visit_groups.first().reset_index()
        n_impressions_by_visit = visit_groups.action.count()
        # if there is a streamed item from a visit, it will be the first item of each group
        # assumption: only one streamed item per visit.
        print('visit model: num. unique visits in training data', visit_groups.ngroups)
        visit_model.fit(visit_simulation, n_impressions_by_visit, hyp, stg)

    return rec_model, user_model, visit_model
