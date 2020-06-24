import numpy as np
import matplotlib.pyplot as plt
import tqdm


LEFT = -1
RIGHT = 1
ACTIONS = [LEFT, RIGHT]


class RandomWalk:
    def __init__(self, n_states):
        self.n_states = n_states

        self.start = n_states // 2
        self.t_state = n_states

        self.true_values = self.get_true_value()

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
        elif self.cur_state == self.t_state:
            reward = 1
            end = True

        return end, state, reward, self.cur_state

    def get_true_value(self):
        values = np.zeros(self.n_states + 2, dtype=np.float)
        values[0] = -1
        values[-1] = 1
        
        while True:
            old_values = np.copy(values)
            for state in range(1, self.n_states+1):
                values[state] = 0
                for action in ACTIONS:
                    next_state = state + action
                    values[state] += (1.0 / 2) * values[next_state]

            error = np.max(np.abs(values - old_values))
            if error < 1e-6:
                break

        return values[1 : -1]

def off_line_lambda(rwalk, lmbda, alphas, episodes, gamma=1.0):
    state_values = np.zeros((rwalk.n_states, len(alphas)), dtype=np.float)
    errors = np.zeros(len(alphas), dtype=np.float)
    
    trueValue=np.arange(-18,20,2)/20

    for ep in range(1, episodes+1):
        rwalk.reset()
        trajectory = list()
        #Generating Episode
        while True:
            action = np.random.choice(ACTIONS)

            end, state, reward, next_state = rwalk.step(action)
            trajectory.append((state, reward, next_state))

            if end:
                break

        T = len(trajectory)

        #Calculating the Lambda Returns
        for t in range(0, T):
            #G_t
            G = 0
            state, _, _ = trajectory[t]
            
            returns = 0
            for i in range(t, T-1):
                _, reward, next_state = trajectory[i]
                returns += np.power(gamma, i-t) * reward
                
                G += (1 - lmbda) * np.power(lmbda, i-t) * \
                    (returns + np.power(gamma, i-t+1) * state_values[next_state])
            
            _, reward, _, = trajectory[-1]
            returns += np.power(gamma, T-t-1) * reward
            G += np.power(lmbda, T-t-1) * returns

            state_values[state] += alphas * (G - state_values[state])
       
        errors += (1.0 / ep) * (np.sqrt(np.mean(np.square(state_values - rwalk.true_values[:, None]), axis=0))\
                               - errors)
    
    return errors

def semi_gradient_td_lambda(rwalk, lmbda, alphas, episodes, gamma=1.0):
    state_values = np.zeros((rwalk.n_states, len(alphas)), dtype=np.float)

    errors = np.zeros(len(alphas), dtype=np.float)

    for ep in range(1, episodes+1):
        rwalk.reset()
        z = np.zeros_like(state_values)

        while True:
            action = np.random.choice(ACTIONS)
            end, state, reward, next_state = rwalk.step(action)

            z *= gamma*lmbda
            z[state] += 1
            
            G = reward
            if not end:
                G += gamma * state_values[next_state]

            delta = G - state_values[state]
            state_values += alphas * delta * z
            #Unstability arises when both lambda and alpha are high.
            state_values = np.minimum(state_values, 50)

            if end:
                break

        errors += (1.0 / ep) * (np.sqrt(np.mean(np.square(state_values - rwalk.true_values[:, None]), axis=0))\
                                - errors)

    return errors

def online_td_lambda(rwalk, lmbda, alphas, episodes, gamma=1.0):
    state_values = np.zeros((rwalk.n_states, len(alphas)), dtype=np.float)

    errors = np.zeros(len(alphas), dtype=np.float)

    for ep in range(1, episodes+1):
        rwalk.reset()
        z = np.zeros_like(state_values)
        V_old = 0
        while True:
            action = np.random.choice(ACTIONS)
            end, state, reward, next_state = rwalk.step(action)

            V = state_values[state]
            if not end:
                V_ = state_values[next_state]
            else:
                V_ = 0

            delta = reward + gamma * V_ - V

            z *= gamma*lmbda
            z[state] += (1 - alphas * gamma * z[state])
            
            state_values += alphas * (delta + V - V_old) * z
            state_values[state] -= alphas * (V - V_old)
            
            V_old = V_

            if end:
                break

        errors += (1.0 / ep) * (np.sqrt(np.mean(np.square(state_values - rwalk.true_values[:, None]), axis=0))\
                                - errors)

    return errors




def figure_12_3():
    episodes = 10
    runs = 20
    lmbdas = [0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = np.linspace(0, 1, 100)

    rwalk = RandomWalk(19)

    plt.figure()

    for lmbda in tqdm.tqdm(lmbdas):
        run_errors = 0
        for r in range(1, runs+1):
            errors = off_line_lambda(rwalk, lmbda, alphas, episodes)
            run_errors += (1.0 / r) * (errors - run_errors)

        plt.plot(alphas, run_errors, label=r"$\lambda={}$".format(lmbda))

    plt.xlabel("Alphas")
    plt.ylabel("RMS error at the end of the episode over the first {} episodes".format(episodes))
    plt.title(r"$Off-Line \ \lambda-Return \ Algorithm$")
    plt.ylim(top=0.60)
    plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1), ncol=1)
    plt.tight_layout()
    plt.savefig("Figure_12_3.png")
    plt.close()

def figure_12_6():
    episodes = 10
    runs = 20
    lmbdas = [0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = np.linspace(0, 1, 100)

    rwalk = RandomWalk(19)

    plt.figure()

    for lmbda in tqdm.tqdm(lmbdas):
        run_errors = 0
        for r in range(1, runs+1):
            errors = semi_gradient_td_lambda(rwalk, lmbda, alphas, episodes)
            run_errors += (1.0 / r) * (errors - run_errors)

        plt.plot(alphas, run_errors, label=r"$\lambda={}$".format(lmbda))

    plt.xlabel("Alphas")
    plt.ylabel("RMS error at the end of the episode over the first {} episodes".format(episodes))
    plt.title(r"$Semi-Gradient \ TD(\lambda) \ Algorithm$")
    plt.ylim([0.25, 0.60])
    plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1), ncol=1)
    plt.tight_layout()
    plt.savefig("Figure_12_6.png")
    plt.close()

def figure_12_8():
    episodes = 10
    runs = 20
    lmbdas = [0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1]
    alphas = np.linspace(0, 1, 100)

    rwalk = RandomWalk(19)

    plt.figure()

    for lmbda in tqdm.tqdm(lmbdas):
        run_errors = 0
        for r in range(1, runs+1):
            errors = online_td_lambda(rwalk, lmbda, alphas, episodes)
            run_errors += (1.0 / r) * (errors - run_errors)

        plt.plot(alphas, run_errors, label=r"$\lambda={}$".format(lmbda))

    plt.xlabel("Alphas")
    plt.ylabel("RMS error at the end of the episode over the first {} episodes".format(episodes))
    plt.title(r"$Online \ TD(\lambda) \ Algorithm$")
    plt.ylim(top=0.60)
    plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1), ncol=1)
    plt.tight_layout()
    plt.savefig("Figure_12_8.png")
    plt.close()



if __name__ == "__main__":
    print("\nFigure 12.3")
    figure_12_3()
    print("\nFigure 12.6")
    figure_12_6()
    print("\nFigure 12.8")
    figure_12_8()
