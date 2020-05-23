import numpy as np
import matplotlib.pyplot as plt
import tqdm
import copy
import seaborn as sns

REJECT = 0
ACCEPT = 1
ACTIONS = [REJECT, ACCEPT]

class Servers:
    def __init__(self, num_servers):
        self.servers = num_servers
        self.free_prob = 0.06
        self.priority = [1, 2, 4, 8]

        self.free_servers = num_servers

        self.p_customer = self.get_customer()

    def get_state(self):
        return (self.free_servers, self.p_customer)

    def get_customer(self):
        return np.random.randint(len(self.priority))

    def step(self, action):
        state = (self.free_servers, self.p_customer)

        servers_freed = np.random.binomial(self.servers - self.free_servers, self.free_prob)
        self.free_servers += servers_freed
        reward = 0
        if self.free_servers > 0:
            reward = action * self.priority[self.p_customer]
            self.free_servers -= action

        #Next State
        self.p_customer = self.get_customer()
        nxt_state = (self.free_servers, self.p_customer)

        return state, action, reward, nxt_state

def differential_semi_gradient_sarsa(server, num_steps, alpha=0.01, beta=0.01, epsilon=0.1):
    q_values = np.zeros((server.servers+1, len(server.priority), len(ACTIONS)), dtype=np.float)
    average_reward = 0
    

    def get_action(state, e=epsilon):
        if np.random.rand() > e:
            values = q_values[state[0], state[1]]
            action_choices = [action for i, action in enumerate(ACTIONS) if values[i] == np.max(values)]
            return np.random.choice(action_choices)
        else:
            return np.random.choice(ACTIONS)

    start_state = server.get_state()
    action = get_action(start_state)

    for step in tqdm.tqdm(range(num_steps)):
        state, a1, reward, nxt_state = server.step(action)
        action = get_action(nxt_state)

        delta = reward - average_reward + q_values[nxt_state[0], nxt_state[1], action] - \
            q_values[state[0], state[1], a1]

        average_reward += beta * delta
        q_values[state[0], state[1], a1] += alpha * delta

    return q_values

if __name__ == "__main__":
    server = Servers(10)
    steps = int(2e7)

    q_values = differential_semi_gradient_sarsa(server, steps)

    fig = plt.figure(figsize=(10, 20))

    ax = fig.add_subplot(2, 1, 1)
    _ = sns.heatmap(np.transpose(np.argmax(q_values, axis=-1)), cmap="YlGnBu", ax=ax, xticklabels=range(1, 11),
                    yticklabels=server.priority)
    
    ax.set_title("Policy")
    ax.set_xlabel("Number of Free Seervers")
    ax.set_ylabel("Priority")

    ax = fig.add_subplot(2, 1, 2)
    q_values[0, :, ACCEPT] = -np.inf

    for i, priority in enumerate(server.priority):
        ax.plot(np.max(q_values[:, i, :], axis=-1), label="Priority={}".format(priority))

    ax.set_title("Value Function")
    ax.set_xlabel("Number of Free Servers")
    ax.set_ylabel("Differential Value of Best Action")

    plt.savefig("access_control.png")
    plt.close()

