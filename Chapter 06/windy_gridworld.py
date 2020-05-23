import numpy as np
import matplotlib.pyplot as plt
import tqdm

UP = np.array([1, 0], dtype=np.int)
DOWN = np.array([-1, 0], dtype=np.int)
RIGHT = np.array([0, 1], dtype=np.int)
LEFT = np.array([0, -1], dtype=np.int)
UR = np.array([1, 1], dtype=np.int)
DR = np.array([-1, 1], dtype=np.int)
DL = np.array([-1, -1], dtype=np.int)
UL = np.array([1, -1], dtype=np.int)
NM = np.array([0, 0], dtype=np.int)

ACTIONS = [UP, DOWN, RIGHT, LEFT]
ACTIONS_KING = [UP, UR, RIGHT, DR, DOWN, DL, LEFT, UL]
ACTIONS_KING_PLUS = [UP, UR, RIGHT, DR, DOWN, DL, LEFT, UL, NM]

class Gridworld:
    def __init__(self, nrows=7, ncols=10, start=(3, 0), goal=(3, 7), stochastic=False):
        self.nrows = nrows
        self.ncols = ncols
        self.start = np.array(start, dtype=np.int)
        self.goal = np.array(goal, dtype=np.int)
        self.stochastic = stochastic
        self.wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0], dtype=np.int)
        
        self.reset()

    def reset(self):
        self.curr_pos = self.start.copy()

    def step(self, action):
        state = self.curr_pos.copy()
        
        if self.stochastic:
            wind_move = self.wind[self.curr_pos[1]] + 1*np.random.choice([-1, 0, 1])
        else:
            wind_move = self.wind[self.curr_pos[1]]
        
        wind_move = np.append(wind_move, [0])
        
        self.curr_pos += action + wind_move
        self.curr_pos[0] = np.clip(self.curr_pos[0], 0, self.nrows-1)
        self.curr_pos[1] = np.clip(self.curr_pos[1], 0, self.ncols-1)

        reward = -1
        if np.array_equal(self.curr_pos, self.goal):
            reward = 0
        
        return state, action, reward, self.curr_pos



def play(stochastic, eps=int(200), epsilon=0.1, alpha=0.5, gamma=1.0):
    gworld = Gridworld(stochastic=stochastic)
    
    ACTION_LIST = [ACTIONS, ACTIONS_KING, ACTIONS_KING_PLUS]
    label_list = ["Four Action", "King Actions", "King Actions + No Move"]


    plt.figure()

    for action_list, label in zip(ACTION_LIST, label_list):
        state_action_values = np.zeros((gworld.nrows, gworld.ncols, len(action_list)), dtype=np.float)

        def policy(pos, epsilon):
            if np.random.rand() > epsilon:
                max_value = np.max(state_action_values[pos[0], pos[1]])
                action_choices = [a for a in range(len(action_list)) if state_action_values[pos[0], pos[1], a] == max_value]
                
                return np.random.choice(action_choices)
            else:
                return np.random.choice(np.arange(len(action_list)))
        
        ep_time_plot = list()
        time = 0
        pbar = tqdm.tqdm(range(eps))
        for e in pbar:
            gworld.reset()
            
            action = policy(gworld.curr_pos, epsilon)
            while True:
                time += 1
                pbar.set_description("Time Step: {}".format(time))
                pbar.refresh()
                ep_time_plot.append(e)
                
                s1, _, reward, s2 = gworld.step(action_list[action])
                a1 = action
                action = policy(gworld.curr_pos, epsilon)

                G = reward + gamma*state_action_values[s2[0], s2[1], action]
                state_action_values[s1[0], s1[1], a1] +=\
                    alpha*(G - state_action_values[s1[0], s1[1], a1])

                if reward == 0:
                    break
        
        final_time = 0
        gworld.reset()
        while True:
            final_time += 1
            action = policy(gworld.curr_pos, epsilon=0.0)
            _, _, reward, _, =  gworld.step(action_list[action])
            if reward == 0:
                break
        
        print("\n")
        print('*'*25)
        print("Time steps for final path: {}".format(final_time))
        print("*"*25)
        print("\n")

        plt.plot(ep_time_plot, label=label)
    
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.legend()
    if stochastic:
        plt.title("Stochastic Windy Gridworld")
        plt.savefig("Windy_Gridworld_Stochastic_Compare.png")
    else:
        plt.title("Windy Gridworld")
        plt.savefig("Windy_Gridworld_Compare.png")

if __name__ == "__main__":
    print("\nNormal Windy Gridworld")
    #play(stochastic=False, eps=250)
    print("\nStochastic Windy Gridworld!")
    play(stochastic=True, eps=1000)
