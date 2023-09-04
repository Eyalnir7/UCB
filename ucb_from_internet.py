import math
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from UCB1 import UCB1



def draw(p):
    if random.random() > p:
        return 0.0
    else:
        return 1.0


def run():
    data = np.empty(horizon, dtype=object)
    last_change = 1
    last_chosen = 0
    for t in range(horizon):
        change_arm = 0
        chosen_arm = algo.select_arm()
        if chosen_arm != last_chosen:
            change_arm = 1
            last_change = t
            last_chosen = chosen_arm
        if t > change*horizon:
            reward = draw(new_arms[chosen_arm])
        else:
            reward = draw(arms[chosen_arm])
        algo.update(chosen_arm, reward)
        row = algo.ucb_values + algo.lcb_values + algo.values + algo.counts + [change_arm]
        data[t] = row
    return data, last_change


new_arms = [0.6, 0.7]
horizon = 10000
change_round = [0.1]
arms = [0.6, 0.4]
algo = UCB1(2, len(arms))
N = 100
H = 10000
l = 100
window_size = 2000

columns = [f"ucb{arm}" for arm in range(len(arms))] + [f"lcb{arm}" for arm in range(len(arms))] + \
          [f"mu{arm}" for arm in range(len(arms))] + [f"c{arm}" for arm in range(len(arms))] + ["changes"]

for change in change_round:
    last_change_hist = np.zeros(N)
    data, last_change_hist[0] = run()
    sim_log = pd.DataFrame(list(data), columns=columns).mul(1 / N)
    one_sim = pd.DataFrame(list(data), columns=columns)
    ones_positions = one_sim.iloc[len(one_sim) // 2:][one_sim['changes'] == 1].index
    differences = [ones_positions[i + 1] - ones_positions[i] for i in range(len(ones_positions) - 1)]
    average_difference = sum(differences) / len(differences) if differences else 0
    x_values = range(100, len(one_sim) - window_size + 1)
    y_values = []
    for start_index in x_values:
        end_index = start_index + window_size
        ones_count = one_sim['changes'][start_index:end_index].sum()
        y_values.append(ones_count)

    plt.plot(x_values, y_values)
    plt.axvline(x=change * horizon)
    plt.show()

    for n in tqdm(range(1, N)):
        algo.reset()
        data, last_change_hist[n] = run()
        # one_sim = pd.DataFrame(list(data), columns=columns)
        # ones_positions = one_sim.iloc[len(one_sim) // 2:][one_sim['changes'] == 1].index
        # differences = [ones_positions[i + 1] - ones_positions[i] for i in range(len(ones_positions) - 1)]
        # average_difference += sum(differences) / len(differences) if differences else 0
        sim_log = sim_log.add(one_sim.mul(1 / N))
    # average_difference = average_difference/N
    # print(f"average difference {average_difference}")

    xrange = range(horizon)

    # plt.plot(xrange, sim_log["c0"], label=f"arm0: {arms[0]} -> {new_arms[0]}")
    # plt.plot(xrange, sim_log["c1"], label=f"arm1: {arms[1]} -> {new_arms[1]}")
    # # plt.plot(xrange, [x ** 0.5 for x in xrange], label="sqrt(x)")
    # plt.axvline(x=change*horizon)
    # plt.title(f"avg of counts across N simulations as function of round for horizon {horizon}")
    # plt.legend()
    # plt.show()

    # plt.plot(xrange, sim_log["ucb0"] - sim_log["ucb1"])
    # plt.axvline(x=change * horizon)
    # plt.title(f"avg difference of ucb over N simulations of {horizon} rounds")
    # plt.show()

    # plt.ylim(0, 1)
    # for arm in range(len(arms)):
    #     plt.plot(xrange, sim_log[f"mu{arm}"], label=f"arm{arm}")
    #     # plt.fill_between(xrange, sim_log[f"lcb{arm}"], sim_log[f"ucb{arm}"], alpha=0.2)
    #     plt.plot(xrange, sim_log[f"ucb{arm}"], label=f"ucb{arm}")
    plt.plot(xrange[3000:], sim_log[3000:]["ucb0"], label="ucb0")
    plt.legend()
    plt.axvline(x=change * horizon)
    plt.title(f"mean of the mean and ucb of arms for {N} simulations of {horizon} rounds")
    plt.show()

    # plt.hist(last_change_hist, bins=20)
    # plt.title(f"last change of arm for {N} simulations of {horizon} rounds")
    # plt.show()

    # last_change_avg_hist = np.zeros(H - l)
    # first_change_avg_hist = np.zeros(H - l)
    # for h in tqdm(range(l, H)):
    #     last_change_avg = 0
    #     first_change_avg = 0
    #     for n in range(N):
    #         first_change = 0
    #         last_change = 1
    #         last_chosen = 0
    #         algo.reset()
    #         for t in range(h):
    #             chosen_arm = algo.select_arm()
    #             if chosen_arm != last_chosen:
    #                 if first_change == 0 and t > change*h:
    #                     first_change = t
    #                 last_change = t
    #                 last_chosen = chosen_arm
    #             if t > change*h:
    #                 reward = draw(new_arms[chosen_arm])
    #             else:
    #                 reward = draw(arms[chosen_arm])
    #             algo.update(chosen_arm, reward)
    #         last_change_avg += last_change
    #         first_change_avg += first_change
    #     last_change_avg = last_change_avg / N
    #     last_change_avg_hist[h - l] = last_change_avg
    #     first_change_avg = first_change_avg / N
    #     first_change_avg_hist[h - l] = first_change_avg
    #
    # xrange = range(l, H)
    # plt.plot(xrange, last_change_avg_hist)
    # plt.title(f"avg round of last change. arms: [{arms[0]}, {arms[1]}]")
    # plt.show()
    #
    # halfx = [x ** 0.5 for x in xrange]
    # plt.plot(xrange, first_change_avg_hist, label="first change per horizon")
    # plt.plot(xrange, halfx, label="1/2 * horizon")
    # plt.title(f"avg round of first change. arms: [{arms[0]}, {arms[1]}] -> [{new_arms[0]}, {new_arms[1]}]")
    # plt.legend()
    # plt.show()
    #
    # plt.plot(xrange, [x - y for x, y in zip(first_change_avg_hist, halfx)], label="y=1/2 * x")
    # plt.show()
