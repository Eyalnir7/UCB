import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

class SWUCB:
    def __init__(self, n_arms, window_size, radius_function):
        """
        Important: the count of an arm are allowed to be zero. This must be taken into consideration when implementing
        :param n_arms: amount of arms in the simulation
        :param window_size: window size for the sliding window
        :param radius_function: how the "confidence bounds" should be calculated
        """
        self.n_arms = n_arms
        self.window_size = window_size
        self.radius_function = radius_function

        # Initialize counts, values, and bounds for each arm
        self.counts = [0 for _ in range(n_arms)]
        self.values = [0.0 for _ in range(n_arms)]
        self.ucb_values = [0.0 for _ in range(n_arms)]
        self.lcb_values = [0.0 for _ in range(n_arms)]
        self.counts_log = [[] for _ in range(n_arms)]

        # Initialize a sliding window for each arm
        self.rewards_window = []

        # Time step
        self.t = 1

    def select_arm(self):
        if self.t <= self.window_size:
            for arm in range(self.n_arms):
                if self.counts[arm] == 0:
                    return arm

        # Select arm based on max of UCB reward within the sliding window
        return self.ucb_values.index(max(self.ucb_values))

    def update(self, chosen_arm, reward):
        """
        Important: the count of an arm are allowed to be zero. This must be taken into consideration when implementing
        the radius function
        :param chosen_arm: the arm the algorithm picked in the select_arm function
        :param reward: a sample from the arm from that was taken in round t
        """
        # Update counts, values, and rewards window for the chosen arm
        self.counts[chosen_arm] += 1
        self.counts_log[chosen_arm].append(self.t)
        self.rewards_window.append((chosen_arm, reward))
        flag = False
        if len(self.rewards_window) > self.window_size:
            forgot_arm, forgot_reward = self.rewards_window.pop(0)
            self.counts[forgot_arm] -= 1
            filtered_window = [t[1] for t in self.rewards_window if t[0] == forgot_arm]
            self.values[forgot_arm] = np.mean(filtered_window)
            flag = True

        if flag and forgot_arm != chosen_arm or not flag:
          filtered_window = [t[1] for t in self.rewards_window if t[0] == chosen_arm]
          self.values[chosen_arm] = np.mean(filtered_window)

        # Update UCB and LCB values for all arms
        for arm in range(self.n_arms):
            if self.counts[arm] > 0:
                radius = self.radius_function(self.t, self.counts[arm])
                self.ucb_values[arm] = self.values[arm] + radius
                self.lcb_values[arm] = self.values[arm] - radius

        # Increment time step
        self.t += 1

    def reset(self):
        # Reset counts, values, and bounds for each arm
        self.counts = [0 for _ in range(self.n_arms)]
        self.values = [0.0 for _ in range(self.n_arms)]
        self.ucb_values = [0.0 for _ in range(self.n_arms)]
        self.lcb_values = [0.0 for _ in range(self.n_arms)]

        # Reset sliding window for each arm
        self.rewards_window = []

        # Reset time step
        self.t = 1

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
    changes_times = np.random.randint(1, horizon, size=num_changes)
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
    :param changes_times: list of values in [1, horizon] such that a change will occur in those rounds. For example if
    we have round 4 in the list, then in round 4 the rewards will be chosen from a different distribution
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
    changes_values.insert(0, initial_arms)
    changes_times.append(horizon+1)

    rangeN = range(N)
    if N != 1:
        rangeN = tqdm(rangeN)

    for _ in rangeN:
        algo.reset()
        phase = 0
        current_regret = np.zeros(horizon)
        for t in range(horizon):
            if t == changes_times[phase]-1:
                phase += 1
            chosen_arm = algo.select_arm()
            reward = draw(changes_values[phase][chosen_arm], deterministic)
            algo.update(chosen_arm, reward)


T = 10000
arms = [0.5,0]
results_list = []

for tau in tqdm(range(T)):
  radius_function = lambda t, c: np.sqrt(2*np.log(tau) / np.max([c, 1]))
  swucb = SWUCB(2, tau, radius_function)
  run_simulation(T, initial_arms=arms, N=1,changes_times=[],
                                               changes_values=[],
                                               save=False,
                                               algo=swucb,
                                               deterministic=True)
  results_list.append(swucb.counts_log)
  if tau == T//2:
    df = pd.DataFrame(results_list)
    df.to_csv('simulation_results_checkpoint.csv', index=False)

df = pd.DataFrame(results_list)
df.to_csv('simulation_results.csv', index=False)