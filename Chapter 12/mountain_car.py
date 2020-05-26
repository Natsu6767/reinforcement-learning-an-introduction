import numpy as np
import matplotlib.pyplot as plt
import tqdm
import copy
import multiprocessing as mp

from functools import partial
from mpl_toolkits.mplot3d import Axes3D
from tiles3 import IHT, tiles


FORWARD = 1
ZERO = 0
REVERSE = -1
ACTIONS = [FORWARD, ZERO, REVERSE]


def bound(x, low, high):
    return np.clip(x, low, high)


class MountainCar:
    def __init__(self):
        self.x_min = -1.2
        self.x_max = 0.5

        self.v_min = -0.07
        self.v_max = 0.07

        self.start_min = -0.6
        self.start_max = -0.4
        
        self.reset()
    
    def reset(self):
        self.pos = np.random.RandomState().uniform(self.start_min, self.start_max)
        self.velocity = 0

    def get_state(self):
        return (self.pos, self.velocity)

    def update(self, action):
        new_velocity = self.velocity + 0.001*action - 0.0025*np.cos(3*self.pos)
        self.velocity = bound(new_velocity, self.v_min, self.v_max)

        new_pos = self.pos + self.velocity
        self.pos = bound(new_pos, self.x_min, self.x_max)
        
        if self.pos == self.x_min:
            self.velocity = 0

    def get_reward(self):
        if self.pos == self.x_max:
            return True, 0
        else:
            return False, -1

    def step(self, action):
        self.update(action)
        end, reward = self.get_reward()

        return end, reward, (self.pos, self.velocity)

class ValueFunction:
    def __init__(self, mcar, replacing=True, clear_trace=False, true_update=False, num_tiling=8, max_size=4096):
        self.num_tiling = num_tiling
        self.replacing = replacing
        self.clear_trace = clear_trace
        self.true_update = true_update
        
        self.iht = IHT(max_size)
        self.weights = np.zeros(max_size, dtype=np.float)
        
        self.x_scale = self.num_tiling / (mcar.x_max - mcar.x_min)
        self.v_scale = self.num_tiling / (mcar.v_max - mcar.v_min)
        
        self.e_trace = np.zeros_like(self.weights)

    def get_active_tiles(self, state, action):
        x, v = state
        active_tiles = tiles(self.iht, self.num_tiling, \
                             [self.x_scale*x, self.v_scale*v], [action])

        return active_tiles

    def value(self, state, action):
        active_tiles = self.get_active_tiles(state, action)

        return np.sum(self.weights[active_tiles])

    def update(self, state, action, alpha, gamma, lmbda,  delta, q_diff):
        alpha /= self.num_tiling
        self.update_trace(state, action, alpha, gamma, lmbda)
            
        if False:#self.true_update:
            self.weights += alpha * (delta + q_diff) * self.e_trace
            active_tiles = self.get_active_tiles(state, action)
            for tile in active_tiles:
                self.weights[tile] -= alpha * q_diff    
        else:
            self.weights += alpha * delta * self.e_trace

    def update_trace(self, state, action, alpha, gamma, lmbda):
        self.e_trace *= gamma * lmbda
        
        active_tiles = self.get_active_tiles(state, action)

        if self.clear_trace:
            clear_tiles = list()
            for a in ACTIONS:
                if a == action:
                    continue
                clear_tiles.extend(self.get_active_tiles(state, a))
            self.e_trace[clear_tiles] = 0

        if self.replacing:
            self.e_trace[active_tiles] = 1
        elif self.true_update:
            self.e_trace[active_tiles] += (1 - alpha * gamma * lmbda * \
                                           np.sum(self.e_trace[active_tiles]))
        else:
            self.e_trace[active_tiles] += 1

    def cost_to_goal(self, state):
        costs = list()
        
        for action in ACTIONS:
            costs.append(self.value(state, action))

        return -np.max(costs)


def sarsa_lambda(episodes, lmbda, replacing, clear_trace, true_update,  alpha, epsilon=0.0, gamma=1.0):
    mcar = MountainCar()
    value_function = ValueFunction(mcar, replacing, clear_trace, true_update)

    steps_per_episode = np.zeros(episodes, dtype=np.int)
    rewards_per_episode = np.zeros(episodes, dtype=np.int)

    def get_action(state, epsilon=epsilon):
        if np.random.RandomState().rand() > epsilon:
            max_value = -value_function.cost_to_goal(state)
            action_choices = [a for a in ACTIONS if \
                              value_function.value(state, a) == max_value]
            
            return np.random.RandomState().choice(action_choices)
        else:
            return np.random.RandomState().choices(ACTIONS)

    for ep in range(episodes):
        mcar.reset()

        state = mcar.get_state()
        action = get_action(state)
        
        q_value = value_function.value(state, action)
        q_old = 0
        
        while True:
            """
            if steps_per_episode[ep] > 300:
                import pdb;
                pdb.set_trace()
            """
            steps_per_episode[ep] += 1
            if steps_per_episode[ep] > 5000:
                print("STEPS LIMIT EXCEED")
                break

            end, reward, next_state = mcar.step(action)
            
            rewards_per_episode[ep] += reward

            q_value = value_function.value(state, action)
            delta = reward - q_value

            if end:
                next_q_value = 0
            else:
                next_action = get_action(next_state)
                next_q_value = value_function.value(next_state, next_action)
                delta += gamma * next_q_value

            q_diff = q_value - q_old

            value_function.update(state, action, alpha, gamma, lmbda, delta, q_diff)

            action = next_action
            state = next_state
            q_old = next_q_value

            if end:
                break
    
    return alpha, np.mean(steps_per_episode), np.mean(rewards_per_episode)


def figure_12_10():
    episodes = 50
    runs = 100

    lmbdas = [0, 0.68, 0.84, 0.92, 0.96, 0.98, 0.99]

    n_points = 24
    alphas=[np.linspace(0.5, 1.8, n_points), 
            np.linspace(0.4, 1.8, n_points), 
            np.linspace(0.3, 1.8, n_points), 
            np.linspace(0.3, 1.8, n_points), 
            np.linspace(0.3, 1.8, n_points),
            np.linspace(0.3, 1.6, n_points),
            np.linspace(0.3, 1.5, n_points)]
    
    """ 
    for l in [lmbdas[1]]:
        alpha_to_idx = {alpha : i for i, alpha in enumerate(alphas[1])}
        run_steps = np.zeros(len(alphas[1]))
        pbar = tqdm.tqdm(alphas[1])
        for alpha in pbar:
            pbar.set_description("Alpha={}".format(alpha))
            for r in tqdm.tqdm(range(1, runs+1)):
                alpha, ep_steps, _ = sarsa_lambda(episodes, l, False, False, True, alpha)
                #alpha, ep_steps = semiGradientSARSA(episodes, 1, alpha)
                i = alpha_to_idx[alpha]
                run_steps[i] += (1.0 / r) * (ep_steps - run_steps[i])
        
        plt.plot(alphas[1], run_steps, label=r"$\lambda = {}$".format(l))
    
    #This is the actual loop. The above is for testing one value of lamnda and corresponding alphas.
    """
    pbar = tqdm.tqdm(lmbdas)
    for lmbda, alphas_ in zip(pbar, alphas):
        pbar.set_description("Lambda = {}".format(lmbda))
        alpha_to_idx = {alpha : i for i, alpha in enumerate(alphas_)}
        run_steps = np.zeros(len(alphas_), dtype=np.float)
        
        for r in tqdm.tqdm(range(1, runs+1)):
            with mp.Pool() as p:
                foo = partial(sarsa_lambda, episodes, lmbda, True, False, False)
                alpha_steps = p.map(foo, alphas_)

            for alpha, avg_ep_steps, _ in alpha_steps:
                i = alpha_to_idx[alpha]
                run_steps[i] += (1.0 / r) * (avg_ep_steps - run_steps[i])

        plt.plot(alphas_, run_steps, label="Lambda = {}".format(lmbda))
    
    plt.xlabel(r"$\alpha \ X \ Number \ of \ Tilings \ (8)$")
    plt.ylabel("Steps per Episode")
    plt.title(r"$Mountain \ Car \ Sarsa(\lambda) \ with \ Replacing \ Traces$")
    plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1), ncol=1)
    plt.tight_layout()
    plt.savefig("Figure_12_10.png")
    plt.close()

def figure_12_11():
    episodes = 20
    runs = 100

    lmbda = 0.9

    n_points = 10
    alphas = [np.linspace(0.2, 2.0, n_points),
              np.linspace(0.2, 2.0, n_points),
              np.linspace(0.2, 2.0, n_points),
              np.linspace(0.2, 0.6, 3)]
            

    labels = [r"$True \ Online \ Sarsa(\lambda)$",
              r"$Sarsa(\lambda) \ with \ replacing traces$",
              r"$Sarsa(\lambda) \ with \ replacing \ traces \ and \ clearing \ traces$",
              r"$Sarsa(\lambda) \ with \ accumulating \ traces$"]
    
    methods_params = [(False, False, True),
                      (True, False, False),
                      (True, True, False),
                      (False, False, False)]

    
    plot_markers = ["s", "v", "o", "*"]

    pbar = tqdm.tqdm(methods_params)

    for method_params, label, marker, alphas_ in zip(pbar, labels, plot_markers, alphas):
        run_rewards = np.zeros(len(alphas_), dtype=np.float)
        alpha_to_idx = {alpha: i for i, alpha in enumerate(alphas_)}
        for r in tqdm.tqdm(range(1, runs+1)):
            with mp.Pool() as p:
                foo = partial(sarsa_lambda, episodes, lmbda, method_params[0], 
                              method_params[1], method_params[2])
                alpha_rewards = p.map(foo, alphas_)

            for alpha, _, ep_rewards in alpha_rewards:
                i = alpha_to_idx[alpha]
                run_rewards[i] += (1.0 / r) * (ep_rewards - run_rewards[i])

        plt.plot(alphas_, run_rewards, label=label, marker=marker)

    plt.xlabel(r"$\alpha \ X \ Number \ of \ Tilings \ (8)$")
    plt.ylabel("Rewards per Episode")
    plt.title(r"$Mountain \ Car \ Sarsa(\lambda) \ Comparisons$")
    plt.ylim([-550, -150])
    plt.legend(loc="best")
    plt.savefig("test_Figure_12_11.png")
    plt.close()


if __name__ == "__main__":
    print("\nFigure 12.10")
    #figure_12_10()
    print("\nFigure 12.11")
    figure_12_11()
