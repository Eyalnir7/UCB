import random
import pandas as pd
from tqdm import tqdm
import numpy as np

from UCB1 import UCB1

CHANGE_P = 4  # used in run simulation as CHANGE_P/horizon for the probability of change in some round


def draw(p):
    """
    draw from ber(p)
    """
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
    numbers = np.arange(1, horizon + 1)
    selected = np.random.rand(horizon) < change_probability
    selected_numbers = numbers[selected].tolist()
    if max_changes is not None:
        selected_numbers = random.sample(selected_numbers, max_changes)
    random_pairs = np.random.rand(len(selected_numbers), num_arms).tolist()
    return selected_numbers, random_pairs


def run_simulation(horizon, algo, initial_arms, changes_times=None, changes_values=None, change_probability=None,
                   max_changes=None):
    """
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
    data = np.empty(horizon, dtype=object)
    changes_times, changes_values = get_changes(horizon, changes_times, changes_values, change_probability, max_changes,
                                                len(initial_arms))
    phase = 0  # the current phase
    # for t in range(horizon):
    #     chosen_arm = algo.select_arm()
    #     if t > change * horizon:
    #         reward = draw(new_arms[chosen_arm])
    #     else:
    #         reward = draw(arms[chosen_arm])
    #     algo.update(chosen_arm, reward)
    #     row = algo.ucb_values + algo.lcb_values + algo.values + algo.counts
    #     data[t] = row
    # return data

