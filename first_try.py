import math
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

arms = [0.8, 0.2]
T = 100000


class MAB:
    def __init__(self, arms, T):
        self.arms = arms
        self.T = T

    def sample_result(self, arm):
        if 0 <= arm < len(self.arms):
            r = np.random.rand()
            return 1 if r < self.arms[arm] else 0
        else:
            return 0

    def run(self, algorithm):
        result = self.T
        for i in tqdm(range(self.T)):
            arm = algorithm.choose_arm()
            sample = self.sample_result(arm)
            result -= sample
            algorithm.get_outcome(sample, arm)
        return result


class Algorithm(ABC):
    @abstractmethod
    def choose_arm(self):
        pass

    @abstractmethod
    def get_outcome(self, sample, arm):
        pass


class UCB(Algorithm):
    def __init__(self, T, K, alpha):
        self.alpha = alpha
        self.K = K
        self.T = T
        self.history = []
        self.arms = {arm: (0, 0) for arm in range(K)}
        self.ucb_index = np.zeros(K)
        self.lcb_index = np.zeros(K)
        self.t = 0
        self.last_change = 1
        self.means = [[] for arm in range(K)]
        self.ucb_index_history = [[] for arm in range(K)]
        self.lcb_index_history = [[] for arm in range(K)]

    def choose_arm(self):
        if self.t < self.K:
            return self.t
        arm = np.argmax(self.ucb_index)
        if arm != self.history[self.t - 1][0]:
            self.last_change = self.t
        return arm

    def get_outcome(self, sample, arm):
        self.t += 1
        update = ((self.arms[arm][0] * self.arms[arm][1] + sample) / float((self.arms[arm][1] + 1)), self.arms[arm][1] + 1)
        self.arms[arm] = update
        self.ucb_index[arm] = self.arms[arm][0] + (self.alpha * math.log(self.t) / float((2 * self.arms[arm][1]))) ** 0.5
        self.lcb_index[arm] = self.arms[arm][0] - (self.alpha * math.log(self.t) / float((2 * self.arms[arm][1]))) ** 0.5
        self.history.append((arm, sample))
        for i in range(self.K):
            self.means[i].append(self.arms[i][0])
            self.ucb_index_history[i].append(self.ucb_index[i])
            self.lcb_index_history[i].append(self.lcb_index[i])

    def reset(self):
        self.history = []
        self.arms = {arm: (0, 0) for arm in range(self.K)}
        self.ucb_index = np.zeros(self.K)
        self.t = 0
        self.last_change = 1
        self.means = [[] for arm in range(self.K)]
        self.ucb_index_history = [[] for arm in range(self.K)]
        self.lcb_index = np.zeros(self.K)


simulation = MAB(arms, T)
ucb = UCB(T, len(arms), 2)
# print(simulation.run(ucb))
# h0 = []
# h1 = []
# sum0, sum1 = 0, 0
# for t in ucb.history:
#     sum0 += 1 if t[0] == 0 else 0
#     sum1 += 1 if t[0] == 1 else 0
#     h0.append(sum0)
#     h1.append(sum1)
# plt.plot(range(T), h0, label="arm0")
# plt.plot(range(T), h1, label="arm1")
# plt.legend()
# plt.xlabel("time step")
# plt.ylabel("sum of times played")
# plt.show()
N = 1
last_change = np.zeros(N)
for i in tqdm(range(N)):
    ucb.reset()
    simulation.run(ucb)
    last_change[i] = ucb.last_change
last_change_mean = last_change.mean()
print(last_change_mean)
#plt.hist(last_change, bins=50)
plt.show()
plt.ylim(0, 1)
x = range(T)
for i in range(ucb.K):
    plt.plot(x, ucb.means[i], label=f"arm {i}")
    plt.fill_between(x, ucb.lcb_index_history[i], ucb.ucb_index_history[i], alpha=0.2)
plt.legend()
plt.show()
plt.ylim(0, 1.5)
for i in range(ucb.K):
    plt.plot(x, ucb.ucb_index_history[i])
plt.show()
hmm = []
hmmm = []
for t in x[:-1]:
    if ucb.ucb_index_history[1][t]-ucb.ucb_index_history[0][t] < 0 and ucb.ucb_index_history[1][t+1]-ucb.ucb_index_history[0][t+1] > 0:
        while ucb.ucb_index_history[1][t]-ucb.ucb_index_history[0][t] > 0:
            hmm.append((ucb.ucb_index_history[0][t], ucb.ucb_index_history[1][t], t))
            hmm.append((ucb.means[0][t], ucb.means[1][t], t))
print(hmm)
print(hmmm)