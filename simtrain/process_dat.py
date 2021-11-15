from . import SETTINGS_POLIMI as SETTINGS
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict

os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_dat(filepath: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load impressions data from `filepath`, return the data set and summary settings derived from it.
    @param filepath: where to find data file
    @return: data set, summary settings
    """
    #
    simulation = pd.read_csv(filepath)

    stg = get_settings(simulation)

    # crinkle of pandas is that empty string is saved as NaN to file, need to revert back to empty string:
    simulation = simulation.replace(np.nan, '', regex=True)

    return simulation, stg


def get_settings(simulation: pd.DataFrame) -> Dict:
    # set global variables
    # derive general settings from loaded dataset:
    stg = {
        'NI': simulation.action.max() + 1,  # num items
        'NU': simulation.user_id.nunique(),  # num users
        'T': simulation.time.max(),  # duration of simulation (in units of days)
        'NS': 100,  # num simulations
        'INF_TIME': SETTINGS.stg['INF_TIME']  # how is infinity time defined (unit of days)
    }
    return stg


def calc_tev(sim: pd.DataFrame) -> Tuple[float, float, float]:
    """
        Calculate basic time statistics from dataset of impressions.
        @param sim: DataFrame dataset of impressions
        @return: time_mean, time_min, tim_max
    """
    Tev = sim['time'].values
    Tevmean = Tev.mean()
    Tevmin = Tev.min()
    Tevmax = Tev.max()
    return Tevmean, Tevmin, Tevmax
