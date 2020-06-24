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
        state = (self.pos, self.velocity)
        self.update(action)
        end, reward = self.get_reward()

        return end, state, action, reward, (self.pos, self.velocity)

class ValueFunction:
    def __init__(self, mcar, num_tiling=8, max_size=4096):
        self.num_tiling = num_tiling
        
        self.iht = IHT(max_size)
        self.weights = np.zeros(max_size, dtype=np.float)
        
        self.x_scale = self.num_tiling / (mcar.x_max - mcar.x_min)
        self.v_scale = self.num_tiling / (mcar.v_max - mcar.v_min)

    def get_active_tiles(self, state, action):
        x, v = state
        active_tiles = tiles(self.iht, self.num_tiling, \
                             [self.x_scale*x, self.v_scale*v], [action])

        return active_tiles

    def value(self, state, action):
        active_tiles = self.get_active_tiles(state, action)

        return np.sum(self.weights[active_tiles])

    def update(self, state, action, delta):
        active_tiles = self.get_active_tiles(state, action)
        delta /= self.num_tiling

        for tile in active_tiles:
            self.weights[tile] += delta

    def cost_to_goal(self, state):
        costs = list()
        
        for action in ACTIONS:
            costs.append(self.value(state, action))

        return -np.max(costs)

def semiGradientSARSA(episodes, n_steps, alpha, epsilon=0.0, gamma=1.0, mcar=None, value_function=None):
    v_func_return = True
    if mcar is None:
        mcar = MountainCar()
    if value_function is None:
        v_func_return = False
        value_function = ValueFunction(mcar)

    steps_per_episode = np.zeros(episodes, dtype=np.int)
    
    N = n_steps + 1
    state_memory = np.zeros((N, 2), dtype=np.float)
    reward_memory = np.zeros(N, dtype=np.int)
    action_memory = np.zeros(N, dtype=np.int)

    def get_action(state, epsilon=epsilon):
        if np.random.RandomState().rand() > epsilon:
            max_value = -value_function.cost_to_goal(state)
            action_choices = [a for a in ACTIONS if \
                              value_function.value(state, a) == max_value]
            
            return np.random.RandomState().choice(action_choices)
        else:
            return np.random.RandomState().choice(ACTIONS)
     
    
    for ep in range(episodes):
        mcar.reset()
        t = 0
        
        state = mcar.get_state()
        action = get_action(state)

        state_memory[t % N] = state
        action_memory[t % N] = action

        T = np.inf

        while True:
            steps_per_episode[ep] += 1
            t += 1
            if t < T:
                end, state, action, reward, next_state = mcar.step(action)

                reward_memory[t % N] = reward
                state_memory[t % N] = next_state

                if end:
                    T = t
                else:
                    action = get_action(next_state)
                    action_memory[t % N] = action

            tau = t - n_steps

            if tau >= 0:
                G = 0.0
                for i in range(tau+1, min(tau+n_steps, T) + 1):
                    G += np.power(gamma, i-tau-1) * reward_memory[i % N]

                if (tau + n_steps) < T:
                    index = (tau + n_steps) % N
                    mem_state = state_memory[index]
                    mem_action = action_memory[index]
                    G += np.power(gamma, n_steps) * value_function.value(mem_state, mem_action)

                update_state = state_memory[tau % N]
                update_action = action_memory[tau % N]

                delta = alpha * (G - value_function.value(update_state, update_action))
                value_function.update(update_state, update_action, delta)

            if tau >= T - 1:
                break

    if v_func_return:
        return value_function
    else:
        return alpha, np.mean(steps_per_episode)

def get_values_for_3d_plot(x, y, value_function):
    values = np.zeros((x.shape[0], y.shape[0]), dtype=np.float)

    for i, pos in enumerate(x):
        for j, v in enumerate(y):
            values[i, j] = value_function.cost_to_goal((pos, v))

    return values

def figure_10_1():
    episodes = 9000
    plot_eps = [1, 12, 104, 1000, 9000]
    alpha = 0.5

    mcar = MountainCar()
    value_function = ValueFunction(mcar)

    X = np.linspace(mcar.x_min, mcar.x_max)
    Y = np.linspace(mcar.v_min, mcar.v_max)
    x, y = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(15, 20))
    plot_idx = 1

    for eps in tqdm.tqdm(range(1, episodes+1)):
        value_function = semiGradientSARSA(1, 1, alpha, mcar=mcar, value_function=value_function)

        if eps in plot_eps:
            ax = fig.add_subplot(3, 2, plot_idx, projection='3d')
            z = get_values_for_3d_plot(X, Y, value_function)
            ax.plot_wireframe(x, y, z)
            ax.set_xlabel("Position")
            ax.set_ylabel("Velocity")
            ax.set_zlabel("Cost to Go")
            ax.set_title("Episode: {}".format(eps), fontsize=15)
            plot_idx += 1
    
    plt.suptitle("Mountain Car Cost-to-Go Functions", fontsize=30)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig("Figure_10_1.png")
    plt.close()


def figure_10_2():
    episodes = 500
    alphas = [0.1, 0.2, 0.5]
    runs = 100
    
    alpha_to_idx = {alpha: i for i, alpha in enumerate(alphas)}
    
    run_steps = np.zeros((len(alphas), episodes), dtype=np.float)
    
    for r in tqdm.tqdm(range(1, runs+1), desc="Runs"):
        with mp.Pool(processes=8) as p:
            foo = partial(semiGradientSARSA, episodes, 1)
            alpha_steps = p.map(foo, alphas)
        
        for alpha, ep_steps in alpha_steps:
            i = alpha_to_idx[alpha]
            run_steps[i] += (1/r) * (ep_steps - run_steps[i])
    
    for i, alpha in enumerate(alphas):
        plt.plot(run_steps[i], label="Alpha = {} / 8".format(alpha))
   
    plt.xlabel("Episodes")
    plt.ylabel("Steps per Episode (log scale)")
    plt.yscale("log", basey=2)
    plt.title("Mountain Car Semi-Gradient SARSA")
    plt.legend(loc="best")
    plt.savefig("Figure_10_2.png")
    plt.close()

def figure_10_3():
    episodes = 500
    n_steps = [8, 1]
    alphas = [0.3, 0.5]
    runs = 100

    mcar = MountainCar()
    
    plt.figure()

    pbar = tqdm.tqdm(n_steps)
    for n, alpha in zip(pbar, alphas):
        pbar.set_description("n = {}, alpha = {}".format(n, alpha))
        run_steps = 0.0

        for r in tqdm.tqdm(range(1, runs+1)):
            _, ep_steps = semiGradientSARSA(episodes, alpha=alpha, n_steps=n)
            run_steps += (1/r) * (ep_steps - run_steps)

        plt.plot(run_steps, label="n={}".format(n))
    
    plt.xlabel("Episodes")
    plt.ylabel("Steps per Episode (log scale)")
    plt.yscale("log", basey=2)
    plt.title("Mountain Car Semi-Gradient n-step SARSA")
    plt.legend(loc="best")
    plt.savefig("Figure_10_3.png")
    plt.close()

def figure_10_4():
    episodes = 50
    runs = 100

    n_steps = np.power(2, np.arange(5))

    n_points = 24
    alphas=[np.linspace(0.5, 1.8, n_points), 
            np.linspace(0.3, 1.8, n_points), 
            np.linspace(0.3, 1.5, n_points), 
            np.linspace(0.1, 1.0, n_points), 
            np.linspace(0.1, 0.7, n_points)]
    
    pbar = tqdm.tqdm(n_steps)
    for n, alphas_ in zip(pbar, alphas):
        pbar.set_description("n_steps = {}".format(n))
        alpha_to_idx = {alpha : i for i, alpha in enumerate(alphas_)}
        run_steps = np.zeros(len(alphas_), dtype=np.float)

        for r in tqdm.tqdm(range(1, runs+1)):
            p = mp.Pool()
            foo = partial(semiGradientSARSA, episodes, n)
            alpha_steps = p.map(foo, alphas_)
            p.close()
            p.join()

            for alpha, ep_steps in alpha_steps:
                i = alpha_to_idx[alpha]
                run_steps[i] += (1/r) * (ep_steps - run_steps[i])

        plt.plot(alphas_, run_steps, label="n={}".format(n))

    plt.xlabel("Alpha X Number of Tilings (8)")
    plt.ylabel("Steps per Episode")
    plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1), ncol=1)
    plt.tight_layout()
    plt.savefig("Figure_10_4.png")
    plt.close()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    print("\nFigure 10.1")
    figure_10_1()
    print("\nFigure 10.2")
    figure_10_2()
    print("\nFigure 10.3")
    figure_10_3()
    print("\nFigure 10.4")
    figure_10_4() 
