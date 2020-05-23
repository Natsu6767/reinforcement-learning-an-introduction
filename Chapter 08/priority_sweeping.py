import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random
import heapq
import copy

class SimpleModel:
    def __init__(self, placeholder=None):
        self.model = dict()

    def feed(self, state, action, reward, next_state):
        self.model[(tuple(state), action)] = (reward, tuple(next_state))

    def sample(self):
        state, action = random.choice(list(self.model.keys()))
        reward, next_state = self.model[(state, action)]

        return state, action, reward, next_state

class PriorityModel(SimpleModel):
    def __init__(self, theta):
        SimpleModel.__init__(self)
        
        self.theta = theta

        self.predecessors = dict()
        self.pqueue = list()
        self.task_finder = dict()
        self.counter = 0


    def is_empty(self):
        if self.task_finder:
            return False
        else:
            return True
    
    def get_value_from_model(key):
        return self.model[key]

    def feed(self, state, action, reward, next_state):
        SimpleModel.feed(self, state, action, reward, next_state)

        if tuple(next_state) not in self.predecessors.keys():
            self.predecessors[tuple(next_state)] = set()
        
        item = (tuple(state), action)
        if item not in self.predecessors[tuple(next_state)]:
            self.predecessors[tuple(next_state)].add(item)
    
    def insert_pq(self, priority, item):
        if priority > self.theta:
            if item in self.task_finder.keys():
                task = self.task_finder.pop(item)
                task[-1] = None
            task = [-priority, self.counter, item]
            self.task_finder[item] = task
            heapq.heappush(self.pqueue, task)
            self.counter+=1

    def pop_queue(self):
        while not len(self.pqueue) == 0:
            _, _, item = heapq.heappop(self.pqueue)
            if item is not None:
                del self.task_finder[item]
                return item
        raise KeyError('Pop from an empty priority queue')

    def sample(self):
        state, action = self.pop_queue()
        reward, next_state = self.model[(state, action)]

        return state, action, reward, next_state

class Maze:
    def __init__(self, factor=1):
        self.factor = factor
        
        self.width = 9 * self.factor
        self.height = 6 * self.factor

        self.start = (0 * self.factor, 3 * self.factor)
        self.goal = (8 * self.factor, 5 * self.factor)

        self.init_obstacles = [(2, 2),
                          (2, 3),
                          (2, 4),
                          (5, 1),
                          (7, 3),
                          (7, 4),
                          (7, 5)]
        
        self.obstacles = list()
        for ob_state in self.init_obstacles:
            self.obstacles.extend(self.extend_state(ob_state))

        self.state_space_size = (self.width, self.height)

        self.reset()

    def extend_state(self, state):
        new_state = [state[0] * self.factor, state[1] * self.factor]
        extend_state = list()
        for i in range(self.factor):
            for j in range(self.factor):
                extend_state.append((new_state[0] + i, new_state[1] + j))

        return extend_state

    def reset(self):
        self.curr_pos = copy.deepcopy(self.start)
    
    def step(self, action):
        state = copy.deepcopy(self.curr_pos)
        self.curr_pos += action
        reward = 0
        end = False

        self.curr_pos[0] = np.clip(self.curr_pos[0], 0, self.width-1)
        self.curr_pos[1] = np.clip(self.curr_pos[1], 0, self.height-1)

        if tuple(self.curr_pos) in self.obstacles:
            self.curr_pos -= action

        if (self.curr_pos == self.goal).all():
            end = True
            reward = 1

        return end, state, action, reward, self.curr_pos


UP = np.array((0, 1))
RIGHT = np.array((1, 0))
DOWN = -UP
LEFT = -RIGHT

ACTIONS = [UP, RIGHT, DOWN, LEFT]

def dyna_q(model, maze, n_planning=0, priority_sweeping=False, epsilon=0.1, alpha=0.1, gamma=1.0):
    state_action_values = np.zeros((maze.width, maze.height, len(ACTIONS)), dtype=np.float)

    def e_policy(state, e=epsilon):
        if np.random.rand() >= e:
            values = state_action_values[state[0], state[1]]
            action_choices = [a for a, value in enumerate(values) if value == np.max(values)]
            return np.random.choice(action_choices)
        else:
            return np.random.choice(np.arange(0, len(ACTIONS)))

    count_updates = 0

    #Episode Loop
    while not check_path(state_action_values, maze):
        maze.reset()
        #import pdb; pdb.set_trace()
        while True:
            action = e_policy(maze.curr_pos)
            end, s1, _, reward, s2 = maze.step(ACTIONS[action])
            a1 = action

            G = reward + gamma*np.max(state_action_values[s2[0], s2[1]])
            if not priority_sweeping:
                state_action_values[s1[0], s1[1], a1] += \
                    alpha * (G - state_action_values[s1[0], s1[1], a1])

            model.feed(s1, action, reward, s2)

            if priority_sweeping:
                priority = np.abs(G - state_action_values[s1[0], s1[1], a1])
                model.insert_pq(priority, (tuple(s1), action))

            for i in range(n_planning):
                if priority_sweeping and model.is_empty():
                    break

                #import pdb; pdb.set_trace()
                s1_m, a1_m, reward_m, s2_m = model.sample()

                G_m = reward_m + gamma*np.max(state_action_values[s2_m[0], s2_m[1]])
                state_action_values[s1_m[0], s1_m[1], a1_m] += \
                    alpha * (G_m - state_action_values[s1_m[0], s1_m[1], a1_m])
                
                if priority_sweeping:
                    for state_pre, action_pre in model.predecessors[tuple(s1_m)]:
                        reward_pre, _ = model.model[(state_pre, action_pre)]
                        priority = np.abs(reward_pre + gamma*np.max(state_action_values[s1_m[0], s1_m[1]]) - \
                                          state_action_values[state_pre[0], state_pre[1], action_pre])
                        model.insert_pq(priority, (tuple(state_pre), action_pre))

                count_updates+=1
            count_updates+=1
            if end:
                break

    return count_updates

def check_path(q_values, maze):
    steps = 0
    maze.reset()
    max_steps = 14 * maze.factor * 1.2

    while True:
        values = q_values[maze.curr_pos[0], maze.curr_pos[1]]
        actions_choices = [a for a, value in enumerate(values) if value == np.max(values)]
        action = np.random.choice(actions_choices)
        
        steps +=1
        if steps > max_steps:
            return False

        end, _, _, _, _ = maze.step(ACTIONS[action])
        if end:
            break

    return  True

def play():
    n_planning = 5
    epsilon = 0.1
    alpha = 0.5
    gamma = 0.95

    theta = 1e-4

    num_of_mazes = 8
    runs = 5
    
    methods = [SimpleModel, PriorityModel]
    labels = ["Dyna-Q", "Priority Sweeping"]

    plt.figure()
    maze_res_updates = np.zeros((num_of_mazes, runs, 2))
    pbar = tqdm.tqdm(range(1, num_of_mazes+1))
    for i, factor in enumerate(pbar):
        maze = Maze(factor=factor)
        for j, (method, lb) in enumerate(zip(methods, labels)):
            for r in range(runs):
                pbar.set_description("Maze Res: x{} Method: {}".format(factor, lb))
                
                model = method(theta)

                maze_res_updates[i, r, j] = dyna_q(model, maze, n_planning, bool(j), epsilon, alpha, gamma)

    maze_res_updates = np.mean(maze_res_updates, axis=1)
    for i, lb in enumerate(labels):
        plt.plot(maze_res_updates[ : , i], label=lb)

    plt.xlabel("Gridworld Size (#states)")
    plt.ylabel("Updates until optimal solution")
    plt.title("Priority Sweeping vs Dyna-Q")
    plt.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
    plt.xticks(np.arange(num_of_mazes), np.power(np.arange(1, num_of_mazes+1), 2)*47)
    plt.legend(loc="best")
    plt.savefig("priority_sweeping_on_mazes.png")
    plt.close()

if __name__ == "__main__":
    play()
