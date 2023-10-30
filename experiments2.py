import Simulation as sim
from DUCB import DUCB
from graphs import Graph
from UCB import UCB
import pickle
import matplotlib.pyplot as plt
import numpy as np


def radius_function(t, c):
    return np.sqrt(np.sqrt(t) / c)


ucb = UCB(2, radius_function)
horizon = 10000
N = 1
arms0 = [0.3, 0]
change_times = [2000, 6000]
change_values = [[0.4, 0.7], [0.9, 0.7]]

ducb = DUCB(2, 2, 1-0.25*np.sqrt(2/10000))

data_sqrt, ct_sqrt, cv_sqrt = sim.run_simulation(horizon, initial_arms=arms0, N=N, changes_times=change_times,
                                                 changes_values=change_values, save=False, algo=ucb, deterministic=True)

data_d, ct_d, cv_d = sim.run_simulation(horizon, initial_arms=arms0, N=N, changes_times=change_times,
                                                 changes_values=change_values, save=False, algo=ducb, deterministic=True)

# def extract_data(path):
#     """
#     :param path: path to file with pickled data
#     :return: data, change_times, change_values
#     """
#     with open(path, 'rb') as file:
#         d, ct, cv = pickle.load(file)
#         return d, ct, cv

# data, change_times, change_values = extract_data("simulations/10000000_[1]")
# print("done reading")
#
# g = Graph(data, change_times, change_values, len(arms0), N)
# g.plot_cumulative_count()
# print("done ploting cumulative count")
# g.plot_means()
# print("done ploting means")
# wrong_arm_picked_list = g.wrong_arm_picked_list()
# print(g.calculate_regret())
# print(wrong_arm_picked_list)
# print(len(wrong_arm_picked_list))
# differences = [wrong_arm_picked_list[i] - wrong_arm_picked_list[i - 1] for i in range(1, len(wrong_arm_picked_list))]
# print(differences)
# plt.plot(range(len(differences)), differences)
# plt.show()
# #
# # plt.hist(wrong_arm_picked_list, bins=len(wrong_arm_picked_list))
# # plt.show()
#
# def is_strictly_increasing(lst):
#     flag = True
#     last = 0
#     for i in range(1, len(lst)):
#         if lst[i] < lst[i - 1]:
#             flag = False
#             last = i
#     return flag, last
#
# #todo: plot
# print(is_strictly_increasing(differences))

# data, change_times, change_values = sim.run_simulation(10000, initial_arms=[0.5, 0.3], N=1, changes_times=[0.3, 0.5],
#                         changes_values=[[0.5, 0.9], [0.5, 0.3]], save=True, algo=alg, deterministic=True)
#
# g = Graph(data, change_times, change_values, 2, 1)
# g.plot_cumulative_count()
# g.plot_means()
