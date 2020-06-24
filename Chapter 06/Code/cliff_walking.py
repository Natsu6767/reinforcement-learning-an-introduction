import numpy as np
import matplotlib.pyplot as plt
import tqdm

class CliffWalking:
    def __init__(self, nrows=4, ncols=12):
        self.nrows = nrows
        self.ncols = ncols
        
        self.start = np.array([0, 0], dtype=np.int)
        self.goal = np.array([0, self.ncols-1], dtype=np.int)
        self.cliff = [c for c in range(self.start[1]+1, self.goal[1]-1)]
    
        self.reset()

    def reset(self):
        self.curr_pos = self.start.copy()

    def step(self, action):
        state = self.curr_pos.copy()
        self.curr_pos += action

        self.curr_pos[0] = np.clip(self.curr_pos[0], 0, self.nrows-1)
        self.curr_pos[1] = np.clip(self.curr_pos[1], 0, self.ncols-1)

        reward = -1
        
        if(self.curr_pos == self.goal).all():
            reward = 0

        if(self.curr_pos[0] == 0 and self.curr_pos[1] in self.cliff):
            reward = -100
            self.reset()

        return state, action, reward, self.curr_pos

def td_control(cwalk, actions, eps=500, epsilon=0.1, alpha=0.5, gamma=1.0, sarsa=True):
    state_action_values = np.zeros((cwalk.nrows, cwalk.ncols, len(actions)), dtype=np.float)

    def e_policy(pos):
        if np.random.rand() > epsilon:
            max_val = np.max(state_action_values[pos[0], pos[1]])
            action_choices = [a for a in range(len(actions)) if state_action_values[pos[0], pos[1], a] == max_val]
            return np.random.choice(action_choices)
        else:
            return np.random.choice(np.arange(0, len(actions)))

    ep_rewards = list()

    for e in range(eps):
        cwalk.reset()
        action = e_policy(cwalk.curr_pos)
        reward_sum = 0
        while True:
            s1, a1, reward, s2 = cwalk.step(actions[action])
            a1 = action

            action = e_policy(cwalk.curr_pos)

            if sarsa:
                G = reward + gamma*state_action_values[s2[0], s2[1], action]
            else:
                G = reward + gamma*np.max(state_action_values[s1[0], s2[1]])

            state_action_values[s1[0], s1[1], a1] +=\
                alpha*(G - state_action_values[s1[0], s1[1], a1])

            if reward == 0:
                break
            reward_sum += reward
        ep_rewards.append(reward_sum)

    return state_action_values, ep_rewards

def play():
    cwalk = CliffWalking()
    
    UP = np.array([1, 0], dtype=np.int)
    RIGHT = np.array([0, 1], dtype=np.int)
    DOWN = np.array([-1, 0], dtype=np.int)
    LEFT = np.array([0, -1], dtype=np.int)

    ACTIONS = [UP, RIGHT, DOWN, LEFT]
    runs = 1000
    eps = 500
    print("\nUsing SARSA")
    sarsa_avg_rewards = np.zeros((runs, eps), dtype=np.float)
    for r in tqdm.tqdm(range(runs)):
        sarsa_sa_values, sarsa_rewards = td_control(cwalk, ACTIONS, eps=eps, sarsa=True)
        sarsa_avg_rewards[r] = sarsa_rewards

    sarsa_avg_rewards = np.mean(sarsa_avg_rewards, axis=0)
    
    print("\nUsing Q-Learning")
    qlearning_avg_rewards = np.zeros((runs, eps), dtype=np.float)
    for r in tqdm.tqdm(range(runs)):
        qlearning_sa_values, qlearning_rewards = td_control(cwalk, ACTIONS, eps=eps, sarsa=False)
        qlearning_avg_rewards[r] = qlearning_rewards

    qlearning_avg_rewards = np.mean(qlearning_avg_rewards, axis=0)
    
    plt.figure()
    plt.plot(sarsa_avg_rewards, label="Sarsa")
    plt.plot(qlearning_avg_rewards, label="Q-leanring")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episode")
    plt.ylim([-100, 0])
    plt.title("Cliff Walking")
    plt.legend()
    plt.savefig("Cliff_Walking_Compare.png")
    plt.close()

    """
    #Testing

    #SARSA
    time_taken = 0
    cwalk.reset()
    rewards = 0
    while True:
        time_taken += 1
        pos = cwalk.curr_pos
        action = np.argmax(sarsa_sa_values[pos[0], pos[1]], axis=-1)
        _, _, r, _ = cwalk.step(ACTIONS[action])
        
        if r == 0:
            break
        rewards += r
    
    print("\n")
    print("*"*25)
    print("Time Steps for SARSA: {}".format(time_taken))
    print("Reward Obtained for SARSA: {}".format(rewards))
    print("*"*25)

    #Q-Learning
    time_taken = 0
    cwalk.reset()
    rewards = 0
    while True:
        time_taken += 1
        pos = cwalk.curr_pos
        action = np.argmax(sarsa_sa_values[pos[0], pos[1]], axis=-1)
        _, _, r, _ =  cwalk.step(ACTIONS[action])
        
        if r == 0:
            break
        rewards += r
    
    print("\n")
    print("*"*25)
    print("Time Steps for Q-Learning: {}".format(time_taken))
    print("Reward Obtained for Q-Learning: {}".format(rewards))
    print("*"*25)
    print("\n")
    """

if __name__ == "__main__":
    play()
