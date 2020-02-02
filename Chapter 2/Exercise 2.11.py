import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from fractions import Fraction

np.random.seed(1123)

class Bandit:

    def __init__(self, k_arms=10, epsilon=0.1, sample_avg=False, alpha=0.1, initial_est=0.0,
                 true_reward=0.0, stationary=True, mu=0.0, sigma=0.01, UCB_param=None, gradient=False):

        self.k = k_arms
        self.eps = epsilon
        self.sample_avg = sample_avg
        self.alpha = alpha
        self.initial_est = initial_est
        self.true_reward = true_reward
        self.stationary = stationary
        self.mu = mu
        self.sigma = sigma
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.average_reward = 0

        self.actions = np.arange(self.k)

    def reset(self):

        if self.stationary:
            self.q_true = np.random.randn(self.k)
        else:
            self.q_true = np.full(self.k, self.true_reward)

        self.q_estimation = np.full(self.k, self.initial_est)

        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.time = 0

    def act(self):
        
        if self.UCB_param is not None:
            UCB_estimate = self.q_estimation + \
                self.UCB_param * np.sqrt((np.log(self.time + 1) / (self.action_count + 1e-7)))
            
            q_best = np.max(UCB_estimate)
            return np.random.choice(np.where(UCB_estimate == q_best)[0])
        
        if self.gradient:
            q_exp = np.exp(self.q_estimation)
            self.pi_prob = q_exp / np.sum(q_exp)

            return np.random.choice(self.actions, p=self.pi_prob)

        if np.random.rand() < self.eps:
            return np.random.choice(self.actions)
        else:
            q_best = np.max(self.q_estimation)
            return np.random.choice(np.where(q_best == self.q_estimation)[0])

    def random_walk(self):
        self.q_true += np.random.randn(self.k)*self.sigma + self.mu
        self.best_action = np.argmax(self.q_true)

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.action_count[action] += 1
        self.time += 1
        baseline = self.average_reward
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_avg:
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        if self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            self.q_estimation += self.alpha * (reward - self.average_reward) * (one_hot - self.pi_prob)
        else:
            self.q_estimation[action] += self.alpha * (reward - baseline)

        if not self.stationary:
            self.random_walk()

        return reward

def play(bandits, runs, time):
    avg_start_time = 100000
    rewards = np.zeros((len(bandits), runs, time - avg_start_time))

    for i, bandit in enumerate(bandits):
        for r in trange(runs, desc="Bandit: {}".format(i+1)):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)

                if bandit.time > avg_start_time:
                    rewards[i, r, bandit.time - avg_start_time - 1] = reward

    mean_rewards = rewards.mean(axis=1).mean(axis=1)
    return mean_rewards

def plot(x, rewards, save_file):
    
    X_tick = [str(Fraction(item).limit_denominator()) for item in x]

    plt.figure(figsize=(10, 10))
    plt.plot(X_tick, rewards, label="E-Greedy")
    #plt.xticks(np.unique(x), X_tick)
    plt.xlabel("Epsilons")
    plt.ylabel("Average Reward Over Last 100,000 Steps.")
    plt.legend()

    plt.savefig(save_file)
    plt.close()

if __name__ == '__main__':
    
    var = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
    runs = 20
    time = 200000
    
    print("Performance Plot for E-Greedy")
    x_eg = var[ : 6]
    bandits = [Bandit(epsilon=eps, stationary=False) for eps in x_eg]
    rewards_eg = play(bandits, runs, time)

    print("Performance Plot for UCB")
    x_UCB = var[3 : ]
    bandits = [Bandit(UCB_param=c, stationary=False) for c in x_UCB]
    rewards_UCB = play(bandits, runs, time)
    
    print("Performance Plot for Gradient Bandit")
    x_g = var[2 : ]
    bandits = [Bandit(alpha=a, gradient=True, stationary=False) for a in x_g]
    rewards_g = play(bandits, runs, time)

    
    X_tick_eg = [str(Fraction(item).limit_denominator()) for item in x_eg]
    X_tick_UCB = [str(Fraction(item).limit_denominator()) for item in x_UCB]
    X_tick_g = [str(Fraction(item).limit_denominator()) for item in x_g]

    plt.figure(figsize=(10, 10))
    plt.plot(X_tick_eg, rewards_eg, label="E-Greedy")
    plt.plot(X_tick_UCB, rewards_UCB, label="UCB")
    plt.plot(X_tick_g, rewards_g, label="Gradient Bandit")
    plt.xlabel("Epsilon, UCB_Param, Alpha")
    plt.ylabel("Average Reward Over Last 100,000 Steps.")
    #plt.yticks(np.arange(1, 1.5, step=0.1))
    plt.legend()
    plt.savefig("./performance_plot.png")
    plt.close()


