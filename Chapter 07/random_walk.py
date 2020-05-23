import numpy as np
import matplotlib.pyplot as plt
import tqdm

class RandomWalk:
    def __init__(self, n_states):
        self.n_states = n_states

        self.start = self.n_states // 2
        self.t_state = self.n_states

        self.reset()

    def reset(self):
        self.cur_state = self.start

    def step(self, action):
        state = self.cur_state
        reward = 0
        end = False

        self.cur_state += action
        
        if self.cur_state < 0:
            reward = -1
            end = True
        elif self.cur_state >= self.t_state:
            reward = 1
            end = True

        return end, state, action, reward, self.cur_state


def n_TD(rwalk, n, alphas, epsilon=0.1,  gamma=1.0, eps=10):
    LEFT = -1
    RIGHT = 1
    ACTIONS = [LEFT, RIGHT]
    
    true_values = np.zeros(rwalk.n_states, dtype=np.float)
    for i in range(1, rwalk.n_states+1):
         true_values[i-1] = i / (rwalk.n_states+1) + (-1 + i/(rwalk.n_states+1))
    
    #State and Reward memory arrays.
    states_mem = np.zeros(n+1, dtype=np.int)
    rewards_mem = np.zeros(n+1, dtype=np.int)
    
    state_values = np.zeros((alphas.shape[0], rwalk.n_states), dtype=np.float)
    
    def policy(state):
        return np.random.choice(np.arange(0, len(ACTIONS)))
    
    eps_errors = np.zeros((alphas.shape[0], eps), dtype=np.float)

    for e in range(eps):
        rwalk.reset()
        states_mem[0] = rwalk.cur_state
        action = policy(rwalk.cur_state)
        
        t = 0
        T = np.inf
        while True:
            t += 1
            if t < T:
                end, _, _, reward, s2 = rwalk.step(ACTIONS[action])

                index = t % (n + 1)
                rewards_mem[index] = reward
                states_mem[index] = s2

                if end:
                    T = t
                else:
                    action = policy(rwalk.cur_state)
            
            tau = t - n

            if tau >= 0:
                G = 0.0
                for i in range((tau + 1), min(tau + n, T) + 1 ):
                    G += np.power(gamma, i - (tau + 1)) * rewards_mem[i % (n + 1)]
                
                if (tau + n) < T:
                    s2 = states_mem[(tau + n) % (n + 1)]
                    G += np.power(gamma, n) * state_values[ :, s2]

                s1 = states_mem[tau % (n + 1)]

                state_values[ :, s1] += \
                    alphas * (G - state_values[ :, s1])

            if tau >= T - 1:
                break
        
        #Calculate RMS and average the errors over the states.
        eps_errors[ : , e] = np.sqrt(np.mean(np.square(state_values - true_values), axis=-1))

    #Average errors over episodes and return.
    return np.mean(eps_errors, axis=-1)

def play():
    
    n_states=19
    rwalk = RandomWalk(n_states=n_states)

    eps = 10
    runs = 100
    epsilon = 0.1
    gamma = 1.0
    alphas = np.linspace(0, 1, num=1001, endpoint=True)
    n_steps = np.power(2, np.arange(0, 10)) 
    
    plt.figure()
    
    pbar = tqdm.tqdm(n_steps)
    for n in pbar:
        errors = np.zeros((alphas.shape[0], runs), dtype=np.float)

        for r in range(runs):
            errors[ :, r] = n_TD(rwalk, n, alphas, epsilon, gamma, eps)
        
        #Average the errors over the runs.
        errors = np.mean(errors, axis=-1)
        #Plot
        plt.plot(alphas, errors, label="n = {}".format(n))
    
    plt.xlabel("Alphas")
    plt.ylabel("Average RMS error over {} states \nand first {} episodes".format(n_states, eps))
    plt.title("Performance of n-step TD on 19-state Random Walk.")
    #Set ylims to make plot look meaningful.
    plt.ylim([0.25, 0.60])
    plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1), ncol=1)
    plt.tight_layout()
    plt.savefig("Random_Walk_n_step_TD.png")
    plt.close()

if __name__ == "__main__":
    play()



