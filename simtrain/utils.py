import itertools
import multiprocessing as mpr
import os
from os.path import join
import itertools

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import coo_matrix

from typing import Callable, List, Tuple, Dict

os.environ['NUMEXPR_MAX_THREADS'] = '16'


def bow(s: np.ndarray) -> str:
    """
        Converts array to bag of words string (this is how we represent a sparse state into a pandas table).
        @param s: word counts over vocabulary
        @return: string representation
    """
    i, = np.where(s > 0)
    return ':'.join(map(str, i))


def inv_bow(b: str, n: int) -> np.ndarray:
    """
        Inverse of `bow` method.
        @param b: string representation
        @param n: number of words in vocabulary
        @return: array representation of bag of words
    """
    assert isinstance(b, str), type(b)
    s = np.zeros(n)  # todo: make sparse matrix
    if len(b) > 0:
        s1 = np.array(list(map(int, b.split(':'))))
        s[s1] = 1.0
    return s


def inv_bow_sp(b: str, n: int) -> list:
    """
        Inverse of `bow` method, sparse version. Allows repeats.
        @param b: string representation
        @param n: number of words in vocabulary
        @return: sparse representation of bag of words of the form [(data,row,column)]
    """
    assert isinstance(b, str), type(b)
    s = []  # [(dat,row,col)] representation
    if len(b) > 0: s = list(map(int, b.split(':')))
    return s


def inv_bow_all(bs: List[str], n: int, dense: bool = False, m: int = None, at_rows: np.ndarray = None) -> coo_matrix:
    """
        Inverse of `bow` method, sparse version, applied to list of bags-of-words.
        Returns sparse matrix (or dense matrix if `dense=True`) of size m-by-n where m=len(bs).
        @param bs: list of bag-of-words (each represented as a string)
        @param n: number of words in vocabulary
        @param dense: return dense array instead of sparse matrix (optional)
        @param m: number of rows (optional)
        @param at_rows: specifies which rows to put each value at, otherwise sequential (optional)
        @return: matrix representation of a list of bag-of-words
    """
    row_ind = []
    col_ind = []
    data = []
    if at_rows is None: 
        at_rows = itertools.count()
        m = len(bs)
    for bi, b in zip(at_rows, bs):
        if len(b) > 0:
            s1 = list(map(int, b.split(':')))
            for si in s1:
                row_ind.append(bi)
                col_ind.append(si)
                data.append(1)
    print('stats of sparse matrix', min(row_ind), max(row_ind), min(col_ind), max(col_ind), m, n)
    X = coo_matrix((data, (row_ind, col_ind)), shape=(m, n), dtype=np.int64)
    if dense:
        return np.asarray(X.todense())
    else:
        return X


def get_dummies_int(s: pd.Series, D: int, dat=None, verbose: int = 0, dense: bool = False) -> coo_matrix:
    """
        One-hot encoding of `s`. Does what `pd.get_dummies` method does with additional ability to
        specify [0:D) vocabulary.
        @param s: series
        @param D: size of vocabulary
        @param dat: values of non-zero entries (default is 1 for each non-zero entry)
        @param verbose:
        @param dense: return dense array instead of sparse matrix
        @return: N-by-D matrix representation of `s`
    """
    N = s.shape[0]
    s_val = s.values
    assert N > 0, 'need non-empty array to one-hot encode'
    assert np.all(s_val < D), 'attempting to one-hot encode outside given dimensions'
    data = np.ones(N) if dat is None else dat
    row_ind = np.arange(N, dtype='int64')
    col_ind = s.values
    if verbose:
        print('data len', N)
        print('row_ind len', N)
        print('col_ind len', len(s.values))
    x = coo_matrix((data, (row_ind, col_ind)), shape=(N, D), dtype=np.int64)
    if dense:
        return np.asarray(x.todense())
    else:
        return x


def make_repeats(xs: np.ndarray) -> np.ndarray:
    """
    Converts an array into an array of indices where the number of repeats of each index is given by its count.
    E.g. [1, 0, 0, 2] -> [0, 3, 3]
    @param xs: array of counts
    @return: array of repeated indices
    """
    xs_pos, = np.where(xs > 0)
    return np.array([[i] * xs[i] for i in xs_pos]).flatten()


def parallelize_fnc(f: Callable[[np.ndarray, np.ndarray], pd.DataFrame],
                    splittable_data, fixed_data, partitions: int) -> pd.DataFrame:
    """
        Applies function `f` across `splittable_data` in parallel, always using the same `fixed_data`.
        @param f: callable method
        @param splittable_data: data over which to parallelize
        @param fixed_data: data that is common to each call of `f`
        @param partitions: number of partitions over which to parallelize
        @return: data frame of concatenated results
    """
    print('parallelizing across num. partitions', partitions)
    cores = mpr.cpu_count()
    pool = mpr.Pool(cores)
    data_split = np.array_split(splittable_data, partitions)
    if fixed_data is not None:
        data_split = zip(data_split, [fixed_data] * partitions)
        data = pd.concat(pool.starmap(f, data_split), axis=0)
    else:
        data = pd.concat(pool.map(f, data_split), axis=0)
    pool.close()
    pool.join()
    return data


def parallelize_fnc_groups(f: Callable[[np.ndarray, np.ndarray], pd.DataFrame],
                           splittable_data: pd.DataFrame, fixed_data, groupcol: str, partitions: int,
                           concat_mode: str = 'pandas') -> pd.DataFrame:
    """
        Applies function `f` across groupings of `splittable_data` DataFrame in parallel,
        always using the same `fixed_data`. For example, can be used to parallelize across
        users so that the data for each user go to the same partition (each partition may have multiple users).
        @param f: callable method
        @param splittable_data: data over which to parallelize
        @param fixed_data: data that is common to each call of `f`
        @param groupcol: column of the data frame `splittable_data` over which to parallelize
        @param partitions: number of partitions over which to parallelize
        @param concat_mode:
        @return: data frame of concatenated results
    """
    print('parallelizing across num. partitions', partitions)
    cores = mpr.cpu_count()

    # get set of unique col entries:
    unique_vals = np.array(splittable_data[groupcol].unique())

    # split into partitions
    val_split = np.array_split(unique_vals, partitions)

    # create groups based on the data split
    grouped_dat = splittable_data.groupby(groupcol)
    data_split = [pd.concat([grouped_dat.get_group(i) for i in split], axis=0) for split in val_split]
    # [df.reset_index(drop=True, inplace=True) for df in data_split]

    pool = mpr.Pool(cores)
    if fixed_data is not None:
        data_split = zip(data_split, [fixed_data] * partitions)
        retval = pool.starmap(f, data_split)
    else:
        retval = pool.map(f, data_split)

    if concat_mode == 'pandas':
        data = pd.concat(retval, axis=0)
    elif concat_mode == 'numpy':
        data = np.concatenate(retval, axis=0)
    elif concat_mode == 'sparse':
        data = sparse.vstack(retval)

    pool.close()
    pool.join()
    return data


def product_array(*xss: Tuple) -> Tuple:
    """
        Calculates K lists representing Cartesian product of the lists in xss.
        @param xss: K-tuple of lists
        @return: Cartesian product of `xss`
    """
    # todo: compare to meshgrid
    prod = itertools.product(*xss)
    return zip(*prod)


def generate_dense_arrays(Xs: List, Ts: List, batch_size: int, steps_per_epoch: int, W_: np.ndarray = None) -> Tuple:
    """
        Generator of minibatches of dense arrays with given type, with option to provide weights for each example.
        @param Xs: list of datasets, e.g. [inputs, outputs] or [inputs]
        @param Ts: list of types corresponding to entries in each of Xs[i]
        @param batch_size: batch size
        @param steps_per_epoch: number of minibatches per pass through the dataset
        @param W_: optional weights for each example
        @return: minibatch
    """
    N, K1 = Xs[0].shape
    assert np.all([X.shape[0] == N for X in Xs]), ','.join([str(X.shape[0]) for X in Xs])
    assert len(Xs) == len(Ts)
    while True:
        ns = np.arange(N, dtype='int64')
        shuffle_ns = np.random.permutation(ns)
        for b in range(steps_per_epoch):
            # get batch of random indices
            shuffle_ns_batch = shuffle_ns[b * batch_size:(b + 1) * batch_size]
            Xs_dense = [X[shuffle_ns_batch, :].toarray().astype(T) for (X, T) in zip(Xs, Ts)]  # 'int64'
            if W_ is not None:
                w_dense = W_[shuffle_ns_batch]
                Xs_dense.append(w_dense)
            yield tuple(Xs_dense)


def summarize_sparse_vector(val: coo_matrix) -> list:
    """
        Make readable version of a sparse matrix.
        @param val: sparse matrix
        @return: list of non-zero indices and their data
    """
    nz_val = np.where(val != 0)[0]
    return list(zip(nz_val, val[nz_val]))


def make_csr(dat: List, dims: Tuple) -> sparse.csr_matrix:
    """
        Make a csr matrix out of the given data.
        @param dat: list of data-row-column [(data,row,col)]
        @param dims: shape of resulting matrix
        @return: sparse matrix
    """
    d, r, c = [], [], []
    if len(dat) > 0: d, r, c = zip(*dat)
    return sparse.csr_matrix((d, (r, c)), shape=dims)  # `sparse` will throw error if lists not all same length


def inv_make_csr(X: sparse.csr_matrix) -> List:
    """
        Inverse of `make_csr` method. Takes a csr matrix and returns list of [(data,row,col)] tuples.
        @param X: sparse matrix
        @return: list of [(data, row, col)] tuples
    """
    Xc = X.tocoo()
    return list(zip(Xc.data, Xc.row, Xc.col))


def make_csr_from_dense_vector(x: np.ndarray, row: int, shape: Tuple) -> sparse.csr_matrix:
    """
        Takes dense np.array x and creates 2D csr matrix where the only non-zero row is x, at position row.
        @param x: dense array
        @param row: starting row for data in sparse representation
        @param shape: shape of sparse representation
        @return: 2-dimensional csr matrix
    """
    Xcoo = sparse.csr_matrix(x).tocoo()
    X = sparse.csr_matrix((Xcoo.data, (Xcoo.row + row, Xcoo.col)), shape=shape)
    return X


def onehot(i: int, N: int) -> np.ndarray:
    """
        Returns one-hot vector of size `N` where position `i` is the only 1 entry.
        @param i: position of non-zero entry
        @param N: size of array
        @return: one hot array
    """
    # todo: make sparse
    i1 = int(i)
    xs = np.zeros(N)
    xs[i1] = 1
    return xs


def lookup_title(sim: pd.DataFrame, title_id: int, inverse: bool = False) -> pd.DataFrame:
    """
        Lookup title in a dataset of titles that contains two representations of titles.
        Useful for mapping external title ids to internal contiguous title id representation.
        @param sim: dataset of titles
        @param title_id: title to look up
        @param inverse: if true, does reverse lookup
        @return: secondary id of title
    """
    if inverse:
        return sim[sim.action == title_id].original_action.iloc[0]
    else:
        return sim[sim.original_action == title_id].action.iloc[0]


def agg_results(rss: List[Dict], alpha: float = 5.0) -> Dict:
    """
        Aggregate list of results into one dictionary summarizing results.
        @param rss: list of results as dictionaries (must have identical keys)
        @param alpha: size of upper and lower bounds
        @return: summarized results
    """
    assert len(rss) > 0
    assert all([rss[0].keys() == rs.keys() for rs in rss])

    ks = rss[0].keys()
    agg = {}
    for k in ks:
        # pull out array of result for key k
        vs = np.array([rs[k] for rs in rss])
        # compute summary:
        agg[k + '_mean'] = vs.mean()
        agg[k + '_lower'] = np.percentile(vs, alpha / 2)
        agg[k + '_upper'] = np.percentile(vs, 100 - alpha / 2)
    return agg


def concat_results(rss: List[Dict]) -> Dict:
    """
    Collapse list of results (as dictionaries) into one dictionary of results where each value is a list.
    @param rss: list of results as dictionaries (must have identical keys)
    @return: all results in a single dictionary
    """
    assert len(rss) > 0
    assert all([rss[0].keys() == rs.keys() for rs in rss])

    ks = rss[0].keys()
    agg = {}
    for k in ks:
        # pull out array of result for key k
        vs = np.array([rs[k] for rs in rss])
        # compute summary:
        agg[k] = np.array(vs)
    return agg


def map_dict(d1: Dict, d2: Dict, f: Callable) -> Dict:
    """
    Return f(d1.k, d2.k), a function of two dicts, matching on key.
    @param d1: dictionary A
    @param d2: dictionary B (must have same keys as d1)
    @param f: function
    @return: dictionary where the values are an arbitrary function of the values of two input dictionaries
    """
    assert d1.keys() == d2.keys()
    ks = d1.keys()
    D = {}
    for k in ks:
        D[k] = f(d1[k], d2[k])
    return D


def init_state(simulation, NI):
    # returns user_id and np.ndarray representation of initial state for all users, ordered by user_id
    first_imps = simulation.sort_values(['time']).groupby('user_id', sort=False).first().sort_values(['user_id'])
    user_ids = np.sort(first_imps.index)
    # initial_user_state is np.array NU-x-NI int64
    return user_ids, inv_bow_all(first_imps.state.values, NI, dense=False).tocsr()


def init_state_dict(simulation, NI):
    # returns user_id and np.ndarray representation of initial state for all users, ordered by user_id
    init_state = dict([(uid,'') for uid in simulation.user_id.unique()])
    first_imps = simulation.sort_values(['time']).groupby('user_id', sort=False).first().sort_values(['user_id']).state
    for u,s in first_imps.iteritems():
        init_state[u] = inv_bow_sp(s, NI)
    return init_state