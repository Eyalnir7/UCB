import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, data, changes_times, changes_values, num_arms, N):
        self.data = data
        self.changes_times = changes_times
        self.changes_values = changes_values
        self.num_arms = num_arms
        self.horizon = len(data)
        self.xrange = range(self.horizon)
        self.distinct_colors = sns.color_palette("husl", num_arms)
        self.N = N

    def plot_cumulative_count(self, arms_subset=None):
        print("plot cumulative count")
        if arms_subset is None:
            arms_subset = range(self.num_arms)
        for i in arms_subset:
            plt.plot(self.xrange, self.data[f"c{i}"], label=f"arm{i}", color=self.distinct_colors[i])

        for i, t in enumerate(self.changes_times):
            end = math.ceil(t-1)
            plt.axvline(x=end)
        plt.title(f"avg of counts across N simulations as function of round for horizon {self.horizon}")
        plt.legend()
        plt.show()

    def plot_changing_means(self):
        start = 1
        for i, t in enumerate(self.changes_times):
            end = math.ceil(t)
            plt.axvline(x=end)
            for arm in range(self.num_arms):
                plt.axhline(y=self.changes_values[i][arm], xmin=start / self.horizon, xmax=end / self.horizon,
                            color=self.distinct_colors[arm],
                            linestyle=":")
            start = end

    def plot_means(self):
        plt.ylim(0, 1)
        for arm in range(self.num_arms):
            plt.plot(self.xrange, self.data[f"mu{arm}"], label=f"arm{arm}", color=self.distinct_colors[arm])
            plt.plot(self.xrange, self.data[f"ucb{arm}"], label=f"ucb{arm}", color=self.distinct_colors[arm],
                     linestyle='dashed')

        self.plot_changing_means()
        plt.legend()
        plt.title(f"means, expectations and ucb")
        plt.show()

    def calculate_regret(self):
        regret = 0
        start = 0
        for phase, t in enumerate(self.changes_times):
            end = math.ceil(t)
            if end == self.horizon:
                end -= 1
            max_arm = np.argmax(self.changes_values[phase])
            max_arm_value = self.changes_values[phase][max_arm]
            for arm in range(self.num_arms):
                if arm == max_arm:
                    continue
                start_count = 0
                if start != 0:
                    start_count = self.data[f"c{arm}"][start - 1]
                wrong_arm_count = self.data[f"c{arm}"][end] - start_count
                regret += wrong_arm_count * (max_arm_value - self.changes_values[phase][arm])
            start = end
        return regret

    def wrong_arm_picked_list(self):
        """
        :return: list of rounds where the wrong arm was picked.
        Should only be used with N = 1 (deterministic case analysis)
        """
        start = 0
        rounds = []
        for phase, t in enumerate(self.changes_times):
            end = math.ceil(t)
            max_arm = np.argmax(self.changes_values[phase])
            current_rounds = np.zeros(end - start)
            for arm in range(self.num_arms):
                if arm == max_arm:
                    continue
                begin = start - 1
                if start == 0:
                    begin = start
                if arm == 0:
                    current_rounds[0] = 1
                current_rounds = current_rounds | \
                                 ((self.data[f'c{arm}'][begin:end] - self.data[f'c{arm}'][begin:end].shift(1)) == 1)
            rounds += self.data.index[current_rounds].tolist()
            start = end
        return rounds

    def plot_ucb_difference(self):
        """
        plots the difference in ucb
        Should only be used with N = 1 (deterministic case analysis) and two arms
        """
        if self.N != 1 or self.num_arms != 2:
            print("function plot_ucb_difference should only be used with N=1 and num_arms = 2")
            return
        plt.plot(self.xrange, self.data["ucb0"]-self.data["ucb1"])
        plt.title(f"ucb0 - ucb1")
        plt.show()


    def plot_cumulative_wrong_arm_count(self):
        pass
