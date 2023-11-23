import Simulation as sim
from DUCB import DUCB
from UCB1 import UCB1
from graphs import Graph
from UCB import UCB
import pickle
import matplotlib.pyplot as plt
import numpy as np

horizon = 100000
arms0 = [0.3, 0]


def radius_function(t, c):
    return np.sqrt(np.log(horizon) / c)


ucb_finite = UCB(2, radius_function)
data_ucb, ct_ucb, cv_ucb, regret_ucb = sim.run_simulation(horizon, initial_arms=arms0, N=1,
                                                          changes_times=[],
                                                          changes_values=[[]],
                                                          save=False,
                                                          algo=ucb_finite,
                                                          deterministic=True)

g = Graph(data_ucb, ct_ucb, cv_ucb, 2, 1)
g.plot_means()
wrong_picked = np.array(g.wrong_arm_picked_list())+1
print(wrong_picked)
print(len(wrong_picked))
xrange = range(horizon)
plt.plot(xrange, regret_ucb, label="ucb")
plt.title("cumulant regret")
plt.legend()
plt.show()


# ucb = UCB(2, radius_function)
# horizon = 10000
# N = 100
# arms0 = [0.3, 0]
# change_times = [0.2, 0.6]
# change_values = [[0.4, 0.7], [0.9, 0.7]]
#
# ducb = DUCB(2, 2, 1 - 0.25 * np.sqrt(2 / 10000))
#
# data_sqrt, ct_sqrt, cv_sqrt, regret_sqrt = sim.run_simulation(horizon, initial_arms=arms0, N=N,
#                                                               changes_times=list(change_times),
#                                                               changes_values=list(change_values),
#                                                               save=False,
#                                                               algo=ucb,
#                                                               deterministic=False)
#
# data_d, ct_d, cv_d, regret_d = sim.run_simulation(horizon, initial_arms=arms0, N=N,
#                                                   changes_times=list(change_times),
#                                                   changes_values=list(change_values),
#                                                   save=False,
#                                                   algo=ducb,
#                                                   deterministic=False)
#
# data_ucb, ct_ucb, cv_ucb, regret_ucb = sim.run_simulation(horizon, initial_arms=arms0, N=N,
#                                                           changes_times=list(change_times),
#                                                           changes_values=list(change_values),
#                                                           save=False,
#                                                           algo=UCB1(2, 2),
#                                                           deterministic=False)
#
# xrange = range(horizon)
# plt.plot(xrange, regret_sqrt, label="sqrt_ucb")
# plt.plot(xrange, regret_d, label="ducb")
# plt.plot(xrange, regret_ucb, label="ucb")
# plt.title("cumulant regret")
# plt.legend()
# plt.show()
#
# g_ucb = Graph(data_ucb, ct_ucb, cv_ucb, 2, N)
# g_ucb.plot_means()
# g_sqrt = Graph(data_sqrt, ct_sqrt, cv_sqrt, 2, N)
# g_sqrt.plot_means()
# g_d = Graph(data_d, ct_d, cv_d, 2, N)
# g_d.plot_means()
# print("end")
