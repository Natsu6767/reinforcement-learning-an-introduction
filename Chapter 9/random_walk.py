import numpy as np
import matplotlib.pyplot as plt
import tqdm
import copy


RIGHT = 1
LEFT = -1
ACTIONS = [RIGHT, LEFT]


class RandomWalk:
    def __init__(self, n_states, step_size=100):
        self.n_states = n_states
        self.step_size = step_size

        self.start = self.n_states // 2

        self.reset()

    def reset(self):
        self.curr_pos = copy.deepcopy(self.start)

    def step(self, action):
        end = False
        reward = 0
        
        steps = np.random.randint(1, self.step_size+1)

        self.curr_pos += action * steps
        self.curr_pos = max(min(self.curr_pos, self.n_states), -1)

        if self.curr_pos == self.n_states:
            end = True
            reward = 1
        elif self.curr_pos == -1:
            end = True
            reward = -1
        
        return end, reward, self.curr_pos

def get_true_value(rwalk):
    state_values = np.zeros(rwalk.n_states+2, dtype=np.float)
    state_values[0] = -1
    state_values[-1] = 1

    for i in range(1, rwalk.n_states+1):
        state_values[i] = 2*i / (rwalk.n_states + 1) - 1
    
    while True:
        old_values = np.copy(state_values)
        for state in range(1, rwalk.n_states+1):
            state_values[state] = 0
            for action in ACTIONS:
                for step in np.arange(1, rwalk.step_size+1):
                    next_state = state + step * action
                    next_state = max(min(next_state, rwalk.n_states+1), 0)

                    state_values[state] += 1.0 / (2 * rwalk.step_size) * state_values[next_state]
        
        error = np.max(np.abs(state_values - old_values))
        if error < 1e-2:
            break

    return state_values[1 : -1]


class BaseValueFunctions:
    def __init__(self, order, func, n_states):
        self.order = order
        self.n_states = n_states

        self.weights = np.zeros(self.order+1, dtype=np.float)
        self.bases = list()

        if func == "Polynomial":
            for i in range(self.order+1):
                self.bases.append(lambda s, n=i: np.power(s, n))
        elif func == "Fourier":
            for i in range(self.order+1):
                self.bases.append(lambda s, n=i: np.cos(n * np.pi * s))

    def value(self, state):
        state /= float(self.n_states)

        features = np.asarray([func(state) for func in self.bases])
        
        return np.dot(features, self.weights)

    def update(self, state, delta):
        state /= float(self.n_states)

        derivative_value = np.asarray([func(state) for func in self.bases])
        self.weights += delta * derivative_value


class TilingValueFunction:
    def __init__(self, n_states, num_tiling, tile_width, tiling_offset_vector):
        self.fundamental_unit = tile_width // num_tiling
        self.n_tiles = n_states // tile_width + 1
        self.num_tiling = num_tiling
        self.tile_width = tile_width

        self.weights = np.zeros((num_tiling, self.n_tiles), dtype=np.float)
        
        ap_n = np.arange(1, num_tiling + 1)
        self.tile_index = -tile_width + (ap_n - 1) * self.fundamental_unit*tiling_offset_vector

        
    def value(self, state):
        value = 0
        for tiling in range(self.num_tiling):
            tile = (state - self.tile_index[tiling]) // self.tile_width

            value += self.weights[tiling, tile]

        return value

    def update(self, state, delta):
        delta /= self.num_tiling
        for tiling in range(self.num_tiling):
            tile = (state - self.tile_index[tiling]) // self.tile_width
            derivative = 1
            self.weights[tiling, tile] += delta * derivative


def gradient_mc(rwalk, value_function, episodes, alpha, true_values=None, dynamic_alpha=False):
    state_values = np.zeros(rwalk.n_states, dtype=np.float)
    
    if true_values is not None:
        ep_errors = np.zeros(episodes, dtype=np.float)

    for e in range(episodes):
        rwalk.reset()
        trajectory = list()

        while True:
            state = rwalk.curr_pos
            action = np.random.choice(ACTIONS)
            end, reward, next_state = rwalk.step(action)

            trajectory.append((state, reward))
            if end:
                break
        
        G = 0.0
        for i, (state, reward) in enumerate(reversed(trajectory)):
            G = reward + G
            alpha_ = alpha
            if dynamic_alpha:
                alpha_ = 500*alpha / (i + 1)

            delta = (G - value_function.value(state))
            delta *= alpha_

            value_function.update(state, delta)
            state_values[state] = value_function.value(state)
        
        if true_values is not None:
            ep_errors[e] = np.sqrt(np.mean(np.square(state_values - true_values)))

    
    if true_values is not None:
        return ep_errors

    return state_values

def polynomial_vs_fourier(n_states):
    rwalk = RandomWalk(n_states)

    base_functions = ["Polynomial", "Fourier"]
    alphas = [1e-4, 5e-5]
    linestyles = ["-", "-"]
    orders = [5, 10, 15]
    episodes = 5000
    runs = 30

    true_values = get_true_value(rwalk)
    
    plt.figure()
    
    pbar = tqdm.tqdm(base_functions, leave=True)
    for base_function, alpha, ls in zip(pbar, alphas, linestyles):
        for order in orders:
            pbar.set_description("{} basis with Order {}".format(base_function, order))
            r_errors = 0
            for r in tqdm.tqdm(range(1, runs+1), desc="Runs"):
                base_value_fucntion = BaseValueFunctions(order, base_function, n_states)
                
                ep_error = gradient_mc(rwalk, base_value_fucntion, episodes, alpha, true_values)

                r_errors += (1.0 / r) * (ep_error - r_errors)

            plt.plot(r_errors, ls=ls, label="{} basis with Order {}".format(base_function, order))

    plt.xlabel("Episodes")
    plt.ylabel("RMS error average over {} runs".format(runs))
    plt.title("Polynomial vs Fourier Basis on {} state Random Walk".format(n_states))
    plt.legend(loc="best")
    plt.savefig("polynomial_vs_fourier.png")
    plt.close()

def tiling_vs_no_tiling(n_states):
    rwalk = RandomWalk(n_states)

    num_tilings = [1, 50]
    tile_width = 200
    tiling_offset_vector = 1
    alpha = 1e-4
    runs = 30
    episodes = 5000

    true_values = get_true_value(rwalk)

    plt.figure()
    pbar = tqdm.tqdm(num_tilings)
    for num_tiling in pbar:
        pbar.set_description("Num Tiling: {}".format(num_tiling))
        r_errors = 0

        for r in tqdm.tqdm(range(1, runs+1), desc="Runs"):
            value_function = TilingValueFunction(n_states, num_tiling, tile_width, tiling_offset_vector)
            ep_error = gradient_mc(rwalk, value_function, episodes, alpha, true_values, True)

            r_errors += (1.0 / r) * (ep_error - r_errors)

        plt.plot(r_errors, label="Num Tiling = {}".format(num_tiling))

    plt.xlabel("Episodes")
    plt.ylabel("RMS error average over {} runs".format(runs))
    plt.title("Effect of Tiling on {} state Random Walk".format(n_states))
    plt.legend(loc="best")
    plt.savefig("tiling_on_random_walk.png")
    plt.close()

if __name__ == "__main__":
    print("Polynomial vs Fourier")
    #polynomial_vs_fourier(1000)
    print("\nTiling vs No Tiling")
    tiling_vs_no_tiling(1000)
