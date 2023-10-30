import Simulation as sim
from graphs import Graph
from UCB1 import UCB1
import pickle


alg = UCB1(2, 2)
horizon = 10000000
N = 150
arms0 = [0.3, 0.6]
change_times = []
change_values = [[]]
data, _, _ = sim.run_simulation(horizon, initial_arms=arms0, N=N, changes_times=change_times,
                                changes_values=change_values, save=True, algo=alg)


# def extract_data(path):
#     """
#     :param path: path to file with pickled data
#     :return: data, change_times, change_values
#     """
#     with open(path, 'rb') as file:
#         d, ct, cv = pickle.load(file)
#         return d, ct, cv
#
#
# data, change_times, change_values = extract_data("simulations/10000_[0.5, 1]")

g = Graph(data, change_times, change_values, len(arms0), N)
g.plot_cumulative_count()
g.plot_means()
