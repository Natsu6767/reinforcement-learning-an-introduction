import numpy as np
import matplotlib.pyplot as plt
import tqdm

RIGHT = 1
LEFT = 0
ACTIONS = [LEFT, RIGHT]

class ShortCorridor:
    def __init__(self):
        self.start_state = 0
        self.goal = 3
        self.reversed_state = 1

        self.reset()

    def reset(self):
        self.cur_state = self.start_state

    def step(self, action):
        state = self.cur_state
        if self.cur_state == self.reversed_state:
            if action == LEFT:
                self.cur_state += 1
            else:
                self.cur_state += -1
        else:
            if action == LEFT:
                self.cur_state += -1
            else:
                self.cur_state += 1

        reward = -1
        end = False
        
        self.cur_state = max(self.cur_state, self.start_state)

        if self.cur_state == self.goal:
            reward = 0
            end = True

        return end, state, reward


class Reinforce:
    def __init__(self, episodes, alpha, gamma=1.0):
        self.episodes = episodes
        self.alpha_theta = alpha
        self.gamma = gamma
        
        self.x = np.array([[0, 1], 
                           [1, 0]])
        self.theta = np.array([-1.47, 1.47])#  np.random.randn(2)#, dtype=np.float)

        self.rewards = list()
        
    def get_pi(self):
        h = np.dot(self.theta, self.x)
        #For numerical stability.
        t = np.exp(h - np.max(h))
        pmf = t / np.sum(t)
        
        imin = np.argmin(pmf)
        epsilon = 0.05
        
        if pmf[imin] < epsilon:
            pmf[:] = 1 - epsilon
            pmf[imin] = epsilon

        return pmf

    def choose_action(self):
        pmf = self.get_pi()
        
        action = np.random.binomial(1, pmf[1])

        return ACTIONS[action]

    def generate_episode(self, s_corridor):
        s_corridor.reset()
        trajectory = list()
        ep_reward = 0
        while True:
            action = self.choose_action()
            end, state, reward = s_corridor.step(action)
        
            ep_reward += reward
            trajectory.append((state, action, reward))

            if end:
                break
        self.rewards.append(ep_reward)
        return trajectory

    def train(self):
        s_corridor = ShortCorridor()
        for ep in range(self.episodes):
            trajectory = self.generate_episode(s_corridor)

            T = len(trajectory)
            G = 0

            for i, (state, action, reward) in enumerate(reversed(trajectory), 1):
                G = reward + self.gamma * G
                pmf = self.get_pi()
                ln_pi_gradient = self.x[ :, action] - np.dot(self.x , pmf)

                update = self.alpha_theta * np.power(self.gamma, T-i) * G * ln_pi_gradient
                self.theta += update
            

class BaselineReinforce(Reinforce):
    def __init__(self, episodes, alpha_theta, alpha_w, gamma=1.0):
        super().__init__(episodes, alpha_theta, gamma)

        self.alpha_w = alpha_w
        self.w = 0

    def train(self):
        s_corridor = ShortCorridor()
        for ep in range(self.episodes):
            trajectory = self.generate_episode(s_corridor)
            
            T = len(trajectory)
            G = 0

            for i, (state, action, reward) in enumerate(reversed(trajectory), 1):
                G = reward + self.gamma * G
                delta = G - self.w

                self.w += self.alpha_w * delta

                pmf = self.get_pi()
                ln_pi_gradient = self.x[ :, action] - np.dot(self.x , pmf)

                self.theta += self.alpha_theta * np.power(self.gamma, T-i) * delta * ln_pi_gradient


def figure_13_1():
    runs = 100
    episodes = 1000
    alphas = [2e-3, 2e-4, 2e-5]

    pbar = tqdm.tqdm(alphas)
    for alpha in pbar:
        pbar.set_description("Alpha = {}".format(alpha))
        avg_run_rewards = np.zeros(episodes, dtype=np.float)

        for r in tqdm.tqdm(range(1, runs+1)):
            reinforce_agent = Reinforce(episodes, alpha)
            reinforce_agent.train()

            avg_run_rewards += (1.0 / r) * (reinforce_agent.rewards - avg_run_rewards)

        plt.plot(avg_run_rewards, label=r"$\alpha = {}$".format(alpha))

    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward on Episode")
    plt.title("Performance of REINFORCE on Short-Corridor Grid World")
    plt.legend(loc="best")
    plt.savefig("test_figure_13_1.png")
    plt.close()

def figure_13_2():
    runs = 100
    episodes = 1000

    plt.figure()
    
    
    print("\nShort-Corridor REINFORCE")
    avg_run_rewards = np.zeros(episodes, dtype=np.float)
    alpha = 2e-4
    for r in tqdm.tqdm(range(1, runs+1)):
        reinforce_agent = Reinforce(episodes, alpha)
        reinforce_agent.train()

        avg_run_rewards += (1.0 / r) * (reinforce_agent.rewards - avg_run_rewards)

    plt.plot(avg_run_rewards, label=r"$REINFORCE, \alpha={}$".format(alpha))
    
    print("\nShort-Corridor REINFORCE with Baseline")
    avg_run_rewards = np.zeros(episodes, dtype=np.float) 
    alpha_theta = alpha*10
    alpha_w = alpha*100
    for r in tqdm.tqdm(range(1, runs+1)):
        baseline_reinforce_agent = BaselineReinforce(episodes, alpha_theta, alpha_w)
        baseline_reinforce_agent.train()

        avg_run_rewards += (1.0 / r) * (baseline_reinforce_agent.rewards - avg_run_rewards)
    plt.plot(avg_run_rewards, label=r"$REINFORCE \ with \ Baseline, \alpha={}, \ w={}$".format(alpha_theta, alpha_w))

    plt.xlabel("Episode")
    plt.ylabel("Total Reward on Episode")
    plt.title("Short-Corridor REINFORCE with and without Baseline Comparison")
    plt.legend(loc="best")
    plt.savefig("Figure_13_2.png")
    plt.close()

if __name__ == "__main__":
    print("\nFigure 13.1")
    figure_13_1()
    print("\nFigure 13.2")
    #figure_13_2()
