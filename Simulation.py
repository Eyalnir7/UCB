import math
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle

CHANGE_P = 4  # used in run simulation as CHANGE_P/horizon for the probability of change in some round


def draw(p, deterministic):
    """
    draw from ber(p) or return p if deterministic
    """
    if deterministic:
        return p
    if random.random() > p:
        return 0.0
    else:
        return 1.0


def get_changes(horizon, changes_times, changes_values, change_probability, max_changes, num_arms):
    """
    aux function for run_simulation
    :return: the changes_times and changes_values for the simulation or None if bad input
    """
    if changes_times is not None:
        if changes_values is not None:
            return changes_times, changes_values
        return None
    if change_probability is None:
        change_probability = CHANGE_P / horizon
    num_changes = np.random.binomial(horizon, change_probability)
    changes_times = np.random.rand(num_changes)
    if max_changes is not None:
        changes_times = random.sample(changes_times, max_changes)
    random_pairs = np.random.rand(len(changes_times), num_arms).tolist()
    return changes_times, random_pairs


def run_simulation(horizon, algo, initial_arms, changes_times=None, changes_values=None, change_probability=None,
                   max_changes=None, N=1, save=False, deterministic=False):
    """
    :param deterministic: whether the arms are bernoulli or deterministic
    :param save: whether to save the log file or not
    :param N: number of simulation to average on
    :param horizon: number of rounds
    :param algo: the algorithm that chooses the arms
    :param changes_times: list of values between 0 and 1 such that a random change will
    happen at round changes[i]*horizon.
    :param max_changes: if changes_times is None, you can input a max number of changes to allow
    :param change_probability: probability of change in arms of every round. if None use 4/horizon. only used if
    changes times is None. Values for after change are assign uniformly
    :param changes_values: list of lists of arms values for each change (not including initial arms)
    :param initial_arms: list of values for initial arms
    :return: dataframe containing information on the run and the array describing the changes
    """
    num_arms = len(initial_arms)
    changes_times, changes_values = get_changes(horizon, changes_times, changes_values, change_probability, max_changes,
                                                num_arms)

    arms_range = range(num_arms)
    columns = [f"ucb{arm}" for arm in arms_range] + [f"lcb{arm}" for arm in arms_range] + \
              [f"mu{arm}" for arm in arms_range] + [f"c{arm}" for arm in arms_range]

    data = np.zeros((horizon, len(columns)))
    changes_values.insert(0, initial_arms)
    changes_times.append(1)

    cumulant_regret = np.zeros(horizon)

    rangeN = range(N)
    if N != 1:
        rangeN = tqdm(rangeN)

    for _ in rangeN:
        algo.reset()
        phase = 0
        current_regret = np.zeros(horizon)
        for t in range(horizon):
            chosen_arm = algo.select_arm()
            if t == math.ceil(changes_times[phase] * horizon):
                phase += 1
            reward = draw(changes_values[phase][chosen_arm], deterministic)
            algo.update(chosen_arm, reward)
            row = algo.ucb_values + algo.lcb_values + algo.values + algo.counts
            if t != 0:
                current_regret[t] += current_regret[t-1] + np.max(changes_values[phase]) - reward
            else:
                current_regret[t] += np.max(changes_values[phase]) - reward
            data[t] += (1 / N) * np.array(row)
        cumulant_regret += (1/N) * current_regret

    data = pd.DataFrame(list(data), columns=columns)
    if save:
        with open(f"simulations/{horizon}_{changes_times}", 'wb') as file:
            pickle.dump((data, changes_times, changes_values), file)
    return data, changes_times, changes_values, cumulant_regret
