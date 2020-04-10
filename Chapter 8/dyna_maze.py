import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random
import copy

class Maze:
    def __init__(self):
        self.start = (0, 3)
        self.goal = (8, 5)

        self.ymax = 5
        self.xmax = 8

        self.block = [(2, 2),
                      (2, 3),
                      (2, 4),
                      (5, 1),
                      (7, 3),
                      (7, 4),
                      (7, 5)]
        self.reset()

    def reset(self):
        self.curr_pos = np.array(self.start)

    def step(self, action):
        state = copy.deepcopy(self.curr_pos)
        reward = 0
        end = False

        self.curr_pos += action
        
        self.curr_pos[0] = np.clip(self.curr_pos[0], 0, self.xmax)
        self.curr_pos[1] = np.clip(self.curr_pos[1], 0, self.ymax)
        
        if tuple(self.curr_pos) in self.block:
            self.curr_pos -= action

        if (self.curr_pos == self.goal).all():
            reward = 1
            end = True

        return end, state, action, reward, self.curr_pos

def tab_dyna_q(maze, eps, n_planning=0, epsilon=0.1, alpha=0.1, gamma=1.0):
    
    UP = np.array([0, 1])
    RIGHT = np.array([1, 0])
    DOWN = np.array([0, -1])
    LEFT = np.array([-1, 0])

    ACTIONS = [UP, RIGHT, DOWN, LEFT]
    
    state_action_values = np.zeros((maze.xmax+1, maze.ymax+1, len(ACTIONS)), dtype=np.float)
    model = dict()

    def e_policy(state, e=epsilon):
        if np.random.rand() >= e:
            values = state_action_values[state[0], state[1]]
            action_choices = [i for i, value in enumerate(values) if value == np.max(values)]

            return np.random.choice(action_choices)
        else:
            return np.random.choice(range(0, len(ACTIONS)))
    
    steps_per_ep = np.zeros(eps, dtype=np.int)
    for e in range(eps):
        maze.reset()
        while True:
            steps_per_ep[e] += 1
            action = e_policy(maze.curr_pos)

            end, s1, _, reward, s2 = maze.step(ACTIONS[action])
            a1 = action

            G = reward + gamma*np.max(state_action_values[s2[0], s2[1]])
            state_action_values[s1[0], s1[1], a1] += \
                alpha * (G - state_action_values[s1[0], s1[1], a1])

            key = tuple((tuple(s1), a1))
            model[key] = (reward, tuple(s2))
            
            for j in range(n_planning):
                s1_m, a1_m = random.choice(list(model.keys()))
                reward_m, s2_m = model[(s1_m, a1_m)]

                G_m = reward_m + gamma*np.max(state_action_values[s2_m[0], s2_m[1]])
                state_action_values[s1_m[0], s1_m[1], a1_m] += \
                    alpha * (G_m - state_action_values[s1_m[0], s1_m[1], a1_m])
            if end:
                break
    
    return steps_per_ep

def play():
    maze = Maze()
    eps = 50
    n_planning_list = [0, 5, 50]
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.95
    
    runs = 30
    
    linestyles = ["--", "-", "--"]
    plt.figure()
    #run_seeds1 = np.linspace(1, 500, runs, dtype=np.int)
    #run_seeds2 = np.linspace(1000, 10000, runs, dtype=np.int)

    pbar = tqdm.tqdm(n_planning_list)
    for n_planning, ls in zip(pbar, linestyles):
       

        pbar.set_description("Planning Steps: {}".format(n_planning))
        pbar.refresh()

        avg_run_steps = np.zeros((runs, eps), dtype=np.float)
        for r in range(runs):
            avg_run_steps[r] = tab_dyna_q(maze, eps, n_planning, epsilon, alpha, gamma)

        avg_run_steps = np.mean(avg_run_steps, axis=0)

        plt.plot(range(2, eps+1) ,avg_run_steps[1 :], ls=ls, label="{} planning steps".format(n_planning))

    plt.xlabel("Episodes")
    plt.ylabel("Steps per Episode")
    plt.title("Dyna-Q Performance")
    plt.ylim([14, 900])
    plt.xticks([0, 2, 10, 20, 30, 40, 50], labels=["", 2, 10, 20, 30, 40, 50])
    plt.yticks([0,14, 200, 400, 600, 800], labels=["", 14, 200, 400, 600, 800])
    plt.legend(loc="best")
    plt.savefig("dyna_maze_performance.png")

if __name__ == "__main__":
    play()
