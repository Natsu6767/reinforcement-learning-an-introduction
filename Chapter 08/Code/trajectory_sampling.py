import numpy as np
import matplotlib.pyplot as plt
import tqdm

ACTIONS = [0, 1]
TERMINATION_PROB = 0.1
EPSILON = 0.1

class Task:
    def __init__(self, n_states, branches):
        self.n_states = n_states
        self.b = branches

        self.transitions = np.random.randint(self.n_states, size=(self.n_states, len(ACTIONS), self.b))
        self.rewards = np.random.randn(self.n_states, len(ACTIONS), self.b)

    def step(self, state, action):
        if np.random.rand() >= TERMINATION_PROB:
            b = np.random.randint(self.b)
            next_state = self.transitions[state, action, b]
            reward = self.rewards[state, action, b]

            return reward, next_state
        else:
            return 0, self.n_states

def uniform(task, eval_interval, max_steps):
    steps = 0
    q_values = np.zeros((task.n_states, len(ACTIONS)), dtype=np.float)
    
    eval_step_list = list()
    eval_value_list = list()
    while True:
        for state in range(task.n_states):
            for action in ACTIONS:
                
                next_states = task.transitions[state, action]
                rewards = task.rewards[state, action]

                q_max = np.max(q_values[next_states], axis=1)

                q_values[state, action] = (1 - TERMINATION_PROB) * np.mean(rewards + q_max) + TERMINATION_PROB*0
                
                if not steps % eval_interval:
                    value = eval_policy(task, q_values)
                    eval_step_list.append(steps)
                    eval_value_list.append(value)
                
                steps +=1
                if steps >= max_steps:
                    return eval_step_list, eval_value_list

def on_policy(task, eval_interval, max_steps):
    steps = 0
    q_values = np.zeros((task.n_states, len(ACTIONS)), dtype=np.float)
    state = 0

    eval_step_list = list()
    eval_value_list = list()

    for steps in range(max_steps):
        if np.random.rand() < EPSILON:
            action = np.random.choice(ACTIONS)
        else:
            values = q_values[state]
            action_choices = [a for a, value in enumerate(values) if value == np.max(values)]
            action = np.random.choice(action_choices)

        next_states = task.transitions[state, action]
        rewards = task.rewards[state, action]

        q_max = np.max(q_values[next_states], axis=1)

        q_values[state, action] = (1 - TERMINATION_PROB)*np.mean(rewards + q_max) + TERMINATION_PROB*0
        
        _, state = task.step(state, action)
        if state == task.n_states:
            state = 0

        if not steps % eval_interval:
            value = eval_policy(task, q_values)
            eval_step_list.append(steps)
            eval_value_list.append(value)

    return eval_step_list, eval_value_list

def eval_policy(task, q_values):
    runs = 1000
    returns = list()

    for r in range(runs):
        ep_rewards = 0
        state = 0

        while state < task.n_states:
            values = q_values[state]
            action_choices = [a for a, value in enumerate(values) if value == np.max(values)]
            action = np.random.choice(action_choices)

            reward, state = task.step(state, action)
            ep_rewards += reward

        returns.append(ep_rewards)
    
    return np.mean(returns)

if __name__ == "__main__":
    branches = [1, 3, 10]
    num_states = 1000
    max_steps = 20000

    methods = [uniform, on_policy]
    labels = ["Uniform", "On-Policy"]
    line_styles = ["--", "-"]

    num_tasks = 75
    x_ticks = 100

    plt.figure(figsize=(10, 20))
    
    print("\nFor 1,000 states")
    plt.subplot(2, 1, 1)

    pbar = tqdm.tqdm(branches)
    for b in pbar:
        values_uniform = list()
        values_on_policy = list()

        for i in range(num_tasks):
            pbar.set_description("Task: {}".format(i+1))

            task = Task(num_states, b)
            
            steps, value = methods[0](task, max_steps//x_ticks, max_steps)
            values_uniform.append(value)

            steps, value = methods[1](task, max_steps//x_ticks, max_steps)
            values_on_policy.append(value)

        values_uniform = np.mean(values_uniform, axis=0)
        values_on_policy = np.mean(values_on_policy, axis=0)
        values = [values_uniform, values_on_policy]
        
        for label, ls, value in zip(labels, line_styles, values):
            plt.plot(steps, value, ls=ls, label="b={} {}".format(b, label))

    plt.title("1,000 States")
    plt.xlabel("Computation time, in expected updates")
    plt.ylabel("Value of start state under greedy policy")
    plt.legend(loc="best")

    print("\nFor 10,000 states.")

    plt.subplot(2, 1, 2)
    num_states = 10000
    max_steps = 200000
    b = 1

    values_uniform = list()
    values_on_policy = list()

    for i in tqdm.tqdm(range(num_tasks)):
        task = Task(num_states, b)
        
        steps, value = methods[0](task, max_steps//x_ticks, max_steps)
        values_uniform.append(value)

        steps, value = methods[1](task, max_steps//x_ticks, max_steps)
        values_on_policy.append(value)

    values_uniform = np.mean(values_uniform, axis=0)
    values_on_policy = np.mean(values_on_policy, axis=0)
    values = [values_uniform, values_on_policy]
        
    for label, ls, value in zip(labels, line_styles, values):
        plt.plot(steps, value, ls=ls, label="b={} {}".format(b, label))

    plt.title("10,000 States")
    plt.xlabel("Computation time, in expected updates")
    plt.ylabel("Value of start state under greedy policy")
    plt.legend(loc="best")
    
    plt.suptitle("Uniform vs On-Policy Trajectory Sampling")
    plt.savefig("Trajectory_Sampling.png")
    plt.close()
