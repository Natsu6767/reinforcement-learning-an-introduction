import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random
import copy

class Maze:
    def __init__(self, shortcut):
        self.shortcut = shortcut

        self.xmax = 8
        self.ymax = 5

        self.start = (3, 0)
        self.goal = (8, 5)
        
        if shortcut:
            self.orig_obstacle = [(x, 2) for x in range(1, self.xmax+1)]
        else:
            self.orig_obstacle = [(x, 2) for x in range(self.xmax)]

        self.reset()
    
    def reset(self):
        self.curr_pos = np.array(self.start)
    
    def obstacle_reset(self):
        self.obstacle = copy.deepcopy(self.orig_obstacle)

    def shortcut_update(self):
        self.obstacle = self.obstacle[ : -1]

    def block_update(self):
        self.obstacle = self.obstacle[1 :]
        self.obstacle.append((self.xmax, 2))

    def step(self, action):
        state = copy.deepcopy(self.curr_pos)
        end = False
        reward = 0

        self.curr_pos += action
        self.curr_pos[0] = np.clip(self.curr_pos[0], 0, self.xmax)
        self.curr_pos[1] = np.clip(self.curr_pos[1], 0, self.ymax)

        if tuple(self.curr_pos) in self.obstacle:
            self.curr_pos -= action

        if (self.curr_pos == self.goal).all():
            reward = 1
            end = True

        return end, state, action, reward, self.curr_pos


def dyna_q(maze, time_steps, plus=True, shortcut=True, n_planning=0, time_weight=1e-3, alpha=1, epsilon=0.1, gamma=0.95):
    UP = np.array([0, 1])
    RIGHT = np.array([1, 0])
    DOWN = np.array([0, -1])
    LEFT = np.array([-1, 0])

    ACTIONS = [UP, RIGHT, DOWN, LEFT]

    state_action_values = np.zeros((maze.xmax+1, maze.ymax+1, len(ACTIONS)), dtype=np.float)
    model = dict()

    def e_policy(state, e=epsilon):
        if np.random.rand() >= e:
            try:
                values = state_action_values[state[0], state[1]]
                action_choices = [i for i, value in enumerate(values) if value == np.max(values)]
                return np.random.choice(action_choices)
            except:
                import pdb; pdb.set_trace()
        else:
            return np.random.choice(np.arange(0, len(ACTIONS)))

    time_reward = np.zeros(time_steps, dtype=np.int)
    state_action_time_intervals = np.zeros((maze.xmax+1, maze.ymax+1, len(ACTIONS)), dtype=np.int)
    
    time = 0

    # episode loop
    while True:
        maze.reset()

        while True:
            if shortcut and time == 3000:
                maze.shortcut_update()
            if not shortcut and time == 1000:
                maze.block_update()

            action = e_policy(maze.curr_pos)
            end, s1, _, reward, s2 = maze.step(ACTIONS[action])
            a1 = action

            G = reward + gamma*np.max(state_action_values[s2[0], s2[1]])
            state_action_values[s1[0], s1[1], a1] += \
                alpha * (G - state_action_values[s1[0], s1[1], a1])

            time_reward[time] = reward

            if plus:
                if (tuple(s1), a1) not in model.keys():
                    for a, _ in enumerate(ACTIONS):
                        model[(tuple(s1), a)] = (0, tuple(s1))
            model[(tuple(s1), a1)] = (reward, tuple(s2))
            
            if plus:
                #Incrementing counter for not observed state and resetting counter of observed state
                state_action_time_intervals[s1[0], s1[1], a1] = time
                #state_action_mask = np.ma.array(state_action_time_intervals, mask=False)
                #state_action_mask.mask[s1[0], s1[1], a1] = True
                #state_action_mask += 1
                #state_action_time_intervals = copy.deepcopy(state_action_mask.data)

            for j in range(n_planning):
                s1_m, a1_m = random.choice(list(model.keys()))
                reward_m, s2_m = model[(s1_m, a1_m)]
                
                
                
                if plus:
                    reward_m += time_weight * np.sqrt(time - state_action_time_intervals[s1_m[0], s1_m[1], a1_m])
                
                G = reward_m + gamma*np.max(state_action_values[s2_m[0], s2_m[1]])
                state_action_values[s1_m[0], s1_m[1], a1_m] += \
                    alpha * (G - state_action_values[s1_m[0], s1_m[1], a1_m])
            
            time += 1
            if end or time >= time_steps:
                break
        if time >= time_steps:
            break

    time_cum_reward = np.cumsum(time_reward, axis=0)

    return time_cum_reward


def play(shortcut, alpha, n, k):
    maze = Maze(shortcut)
    if shortcut:
        time_steps = 6000
    else:
        time_steps = 3000
    
    n_planning = n
    runs = 10
    plus = [True, False]
    labels = ["Dyna-Q+", "Dyna-Q"]
    plt.figure()
    y_max = 0
    for p, lb in zip(plus, labels):
        cum_rewards = np.zeros((runs, time_steps), dtype=np.float)
        print("{} {} Maze".format(lb, "Shortcut" if shortcut else "Blocking"))
        for r in tqdm.tqdm(range(runs)):   
            maze.obstacle_reset()
            cum_rewards[r] = dyna_q(maze, time_steps, p, shortcut, n_planning, time_weight=k, alpha=alpha)
        
        y_max = max(np.mean(cum_rewards, axis=0).max(), y_max)
        plt.plot(np.mean(cum_rewards, axis=0), label=lb)

    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward")
    plt.legend(loc="best")
    if shortcut:
        plt.vlines(3000, 0, y_max, ls="--")
        plt.title("Shortcut Maze")
        plt.savefig("shortcut_maze.png")
    else:
        plt.vlines(1000, 0, y_max, ls="--")
        plt.title("Blocking Maze")
        plt.savefig("blocking_maze.png")
    plt.close()


if __name__ == "__main__":
    play(shortcut=False, alpha=1.0, n=50, k=1e-3)
    play(shortcut=True, alpha=1, n=50, k=1e-3)
