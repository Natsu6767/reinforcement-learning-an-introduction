"""
Exercise 2.5 (programming) 
Design and conduct an experiment to demonstrate the
difficulties that sample-average methods have for nonstationary problems. Use a modified
version of the 10-armed testbed in which all the q*(a) start out equal and then take
independent random walks (say by adding a normally distributed increment with mean
zero and standard deviation 0.01 to all the q*(a) on each step). Prepare plots like
Figure 2.2 for an action-value method using sample averages, incrementally computed,
and another action-value method using a constant step-size parameter, alpha = 0.1. Use
epsilon  = 0.1 and longer runs, say of 10,000 steps.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(1123)

class Bandit:

    """    
    @k_arms: number of arms/action.
    @epsilon: probability of exploring.
    @stationary: if True, the true reward for each action remains the same distribution.
    @true_reward: starting true reward for each action.
    @initial_est: our initial estimation of the reward for all of the actions.
    @sample_avg: if True, use sample average to update estimations instead of constant step size.
    @step_size: constant step size for updating estimations.
    @sigma: the standard deviation of the normal distribution to sample from for random walk when action rewards are not stationary.
    @mu: the mean of the normal distribution to sample from for random walk when action rewards are not stationary.
    """

    def __init__(self, k_arms=10, epsilon=0, stationary=False, true_reward=0.0, 
                  initial_est=0.0, sample_avg=True, step_size=0.1, sigma=0.01, mu=0.0):

        self.k = k_arms
        self.epsilon = epsilon
        self.initial = initial_est
        self.true_reward = true_reward
        self.step_size = step_size
        self.sample_avg = sample_avg
        self.stationary = stationary
        self.actions = np.arange(self.k)
        self.average_reward = 0
        self.sigma = sigma
        self.mu = mu

    # resets the bandit. Initial state.
    def reset(self):
        # Real reward for each action
        if self.stationary:
            self.q_true = np.random.randn(self.k)
        else:
            self.q_true =  np.full(self.k, self.true_reward)
        # Reward estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial
        # What the best action given the current q_tue is.
        self.best_action = np.argmax(self.q_true)
        # To count the number of time a particular action was selected.
        self.action_count = np.zeros(self.k)
        self.time = 0

    # select the action. Exploitation vs Exploration.
    def act(self):

        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        # Exploitation
        else:
            q_best = np.max(self.q_estimation)
            return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # Change the q_true distributions when non stationary.
    def random_walk(self):
        self.q_true += np.random.randn(self.k) * self.sigma + self.mu
        self.best_action = np.argmax(self.q_true)

    # Update the estimations.
    def step(self, action):

        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        
        if self.sample_avg:
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        else:
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])


        if not self.stationary:
            self.random_walk()
        return reward

# Simulate the games.
def play(bandits, runs, time):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    
    for i, bandit in enumerate(bandits):
        for r in tqdm(range(runs), desc='Bandit_{}. Epsilon: {}'.format(i+1, bandit.epsilon)):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1

    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)

    return mean_best_action_counts, mean_rewards

def plot(epsilons, best_action_counts, rewards, save_file):
    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, reward in zip(epsilons, rewards):
        plt.plot(reward, label='epsilons = %0.2f' %(eps))
    plt.xlabel('steps')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts*100, label='epsilons = %0.2f' %(eps))
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()

    plt.savefig('./'+save_file)
    plt.close()


if __name__ == '__main__':

    epsilons = [0.0, 0.1, 0.01]
    runs = 2000
    time = 10000

    print("Using Sample Average\n")
    bandits = [Bandit(epsilon=eps, sample_avg=True, stationary=False) for eps in epsilons]
    best_action_counts_sa, rewards_sa = play(bandits, runs=runs, time=time)
    
    plot(epsilons, best_action_counts_sa, rewards_sa, 'sample_avg.png')
    
    print("Using Constant Step Size")
    bandits = [Bandit(epsilon=eps, sample_avg=False, stationary=False) for eps in epsilons]
    best_action_counts_cs, rewards_cs = play(bandits, runs=runs, time=time)
    
    plot(epsilons, best_action_counts_cs, rewards_cs, 'constant_step.png')

    plt.figure(figsize=(10, 30))
    i = 0
    for r_sa, r_cs in zip(rewards_sa, rewards_cs):
        i += 1
        plt.subplot(3, 1, i)
        plt.title("Epsilon = %0.2f " %(epsilons[i-1]))
        plt.plot(r_sa, label="Sample Average")
        plt.plot(r_cs, label="Constant Step Size")
        plt.xlabel('Steps')
        plt.ylabel('Average Reward')
        plt.legend()

    plt.savefig("./sa_cs_comparisons.png")
    


