import numpy as np
import matplotlib.pyplot as plt
import tqdm
import copy

class RandomWalk:
    def __init__(self, n_states, step_size=100):
        self.n_states = n_states
        self.step_size = step_size

        self.start = n_states // 2
        self.reset()

    def reset(self):
        self.curr_pos = copy.deepcopy(self.start) 
    
    def step(self, action):
        step = np.random.randint(1, self.step_size+1)

        end = False
        self.curr_pos += action*step
        self.curr_pos = max(min(self.curr_pos, self.n_states), -1)

        if self.curr_pos == self.n_states:
            end = True
            reward = 1
        elif self.curr_pos == -1:
            end = True
            reward = -1
        else:
            reward = 0

        return end, reward, self.curr_pos


LEFT = -1
RIGHT = 1
ACTIONS = [LEFT, RIGHT]

def get_true_value(rwalk):
    state_values = np.zeros(rwalk.n_states+2, dtype=np.float)
    
    state_values[0] = -1
    state_values[-1] = 1
    
    for i in range(1, rwalk.n_states+1):
        #(i - (n+1-i))
        state_values[i] = 2*i / (rwalk.n_states + 1) - 1
    
    while True:
        old_values = np.copy(state_values)

        for state in range(1, rwalk.n_states+1):
            state_values[state] = 0
            for action in ACTIONS:
                for step in range(1, rwalk.step_size+1):
                    next_state = state + step*action

                    next_state = max(min(next_state, rwalk.n_states+1), 0)
                    state_values[state] += 1.0 / (2 * rwalk.step_size) * state_values[next_state]

        error = np.sum(np.abs(state_values - old_values))
        if error < 1e-2:
            break
    
    state_values[0] = state_values[-1] = 0
    return state_values[1 : -1]


def state_aggregation_mc(rwalk, group_size=100):
    state_values = np.zeros(rwalk.n_states, dtype=np.float)
    weights = np.zeros(rwalk.n_states//group_size, dtype=np.float)
    
    true_values = get_true_value(rwalk)

    alpha = 2e-5
    episodes = int(1e5)

    pbar = tqdm.tqdm(range(episodes))
    for e in pbar:
        rwalk.reset()
        
        trajectory = list()
        while True:
            state = rwalk.curr_pos
            action = np.random.choice(ACTIONS)
            end, reward, next_state = rwalk.step(action)

            trajectory.append([state, reward])
            if end:
                break

        G = 0
        for state, reward in reversed(trajectory):
            G = reward + G
            g_index = state // group_size
            weights[g_index] += alpha*(G - weights[g_index])
            state_values[state] = weights[g_index]


    return state_values

def state_aggregation_td(rwalk, true_values,  group_size, episodes, alphas,  n_steps=1):
    state_values = np.zeros((len(alphas), rwalk.n_states), dtype=np.float)
    weights = np.zeros((len(alphas), rwalk.n_states//group_size), dtype=np.float)

    gamma = 1.0

    states_mem = np.zeros(n_steps+1, dtype=np.int)
    rewards_mem = np.zeros(n_steps+1, dtype=np.int)
    
    eps_errors = np.zeros((len(alphas), episodes), dtype=np.float)
    
    #import pdb; pdb.set_trace();

    pbar = tqdm.tqdm(range(episodes))
    for e in pbar:
        rwalk.reset()

        T = np.inf
        t = 0

        states_mem[t] = rwalk.curr_pos
        action = np.random.choice(ACTIONS)
        
        while True:
            t += 1
            if t < T:
                end, reward, next_state = rwalk.step(action)

                index = t % (n_steps + 1)
                rewards_mem[index] = reward
                states_mem[index] = next_state

                if end:
                    T = t
                else:
                    action = np.random.choice(ACTIONS)

            tau = t - n_steps
            if tau >= 0:
                G = 0.0
                for i in range((tau + 1), min(tau + n_steps, T) + 1):
                    G += np.power(gamma, i - (tau + 1)) * rewards_mem[i % (n_steps + 1)]

                if (tau + n_steps) < T:
                    s2 = states_mem[(tau + n_steps) % (n_steps + 1)]
                    G += np.power(gamma, n_steps) * weights[ :, s2//group_size]

                s1 = states_mem[tau % (n_steps + 1)]

                weights[ :, s1//group_size] += \
                    alphas * (G - weights[ :, s1//group_size])

                state_values[ :, s1] = weights[ :, s1//group_size]
            
            if tau >= T-1:
                break
        
        eps_errors[ :, e] = np.sqrt(np.mean(np.square(state_values - true_values), axis=-1)) 

    return state_values, np.mean(eps_errors, axis=-1)


def approx_mc(n_states, group_size):
    rwalk = RandomWalk(n_states, group_size)

    print("Calculating True Values")
    true_values = get_true_value(rwalk)
    print("\n\nCalculating the State Values using Approximate MC method.")
    state_values = state_aggregation_mc(rwalk, group_size)
    
    plt.plot(np.arange(1, n_states+1), true_values, label="True Value")
    plt.plot(state_values, label="Approximate MC Value")
    plt.xlabel("States")
    plt.ylabel("Value")
    plt.title("Gradient Monte-Carlo on {}-state Random Walk".format(n_states))
    plt.legend(loc="best")
    plt.savefig("gradient_mc_random_walk.png")
    plt.close()


def approx_td(n_states, group_size):
    rwalk = RandomWalk(n_states, group_size)
    true_values = get_true_value(rwalk)

    print("Calculating True Values")
    true_values = get_true_value(rwalk)
    print("\n\nCalculating the State Values using Approximate TD(0) menthod.")
    alphas = [2e-4]
    episodes = int(1e5)
    group_size = 100
    state_values, _ = state_aggregation_td(rwalk, true_values, group_size, episodes, alphas, 1)

    plt.figure(figsize=(10, 20))
    
    plt.subplot(2, 1, 1)

    plt.plot(np.arange(1, n_states+1), true_values, label="True Value")
    plt.plot(state_values.squeeze(), label="Approximate TD Value")
    plt.title("Semi-Gradient TD(0) on 1000 state Random Walk")
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.legend(loc="best")

    print("\nComputing Error Plots for different alphas")
    alphas = np.linspace(0, 1, num=1001)
    n_steps = np.power(2, np.arange(0, 10))
    runs = 20
    episodes = 10
    group_size = 50
    rwalk = RandomWalk(n_states, group_size)
    
    plt.subplot(2, 1, 2)

    pbar = tqdm.tqdm(n_steps)
    for n in pbar:
        run_errors = np.zeros((runs, len(alphas)), dtype=np.float)
        for r in range(runs):
            pbar.set_description("n_steps: {}, Run: {}".format(n, r+1))

            _, run_errors[r] = state_aggregation_td(rwalk, true_values, group_size, episodes, alphas, n)

        run_errors = np.mean(run_errors, axis=0)
        plt.plot(alphas, run_errors, label="n = {}".format(n))

    plt.title("RMS Error for n-step TD")
    plt.xlabel("Alpha")
    plt.ylabel("Average RMS error over 1000 states and first 10 episodes.")
    plt.ylim(top=0.60)
    plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1), ncol=1)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    plt.suptitle("Bootstrapping with state aggregation on the 1000-state random walk task")
    plt.savefig("TD_random_walk.png")
    plt.close()



if __name__ == "__main__":
    approx_mc(1000, 100)
    approx_td(1000, 100)
