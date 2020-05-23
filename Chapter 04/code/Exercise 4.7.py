import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import seaborn as sns
import time

from functools import partial
from scipy.stats import poisson
from mpl_toolkits.mplot3d import Axes3D

# Problem Constants
MAX_CARS = 20
MAX_MOVE = 5

E_REQ_1 = 3
E_REQ_2 = 4

E_RET_1 = 3
E_RET_2 = 2

RENT_CREDIT = 10
MOVE_COST = -2
PARKING_COST = -4
MAX = 11

poisson_cache = dict()

def poisson_prob(n, lam):
    global poisson_cache
    key = n*10 + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)

    return poisson_cache[key]

class PolicyIteration:
    def __init__(self, n_parallel=8, gamma=0.9, delta=1e-2):
        self.n_parallel = n_parallel
        self.gamma = gamma
        self.delta = delta

        self.values = np.zeros((MAX_CARS+1, MAX_CARS+1))
        self.policy = np.zeros(self.values.shape, dtype=np.int)
        self.actions = np.arange(-MAX_MOVE, MAX_MOVE+1)
        self.a2i = {a: i for i, a in np.ndenumerate(self.actions)}
        self.iterations = 0

    def solve(self):
        self.plot()
        total_start_time = time.time()
        print("STARTING!")
        while True:
            start_time = time.time()
            self.values = self.policy_evaluation(self.values, self.policy)
            elapsed_time = time.time() - start_time
            print("Elapsed Time: {} seconds.".format(elapsed_time))
            
            start_time = time.time()
            policy_change, self.policy = self.policy_improvement(self.values, self.policy, self.actions)
            elapsed_time = time.time() - start_time
            print("Elapsed Time: {} seconds.".format(elapsed_time))

            self.iterations += 1
            self.plot()

            if policy_change == 0:
                break
        total_elapsed_time = time.time() - total_start_time
        print("The policy has converged after {} iterations in {} seconds.".format(self.iterations, total_elapsed_time))


    def bellman(self, values, action, state):
        expected_return = 0

        if action > 0:
            expected_return += MOVE_COST * (action -1)
        else:
            expected_return += MOVE_COST * abs(action)

        for req1 in range(MAX):
            for req2 in range(MAX):

                num_cars_first_loc = min(state[0] - action, MAX_CARS)
                num_cars_second_loc = min(state[1] + action, MAX_CARS)

                #Requets
                real_requests_first_loc = min(num_cars_first_loc, req1)
                real_requests_second_loc = min(num_cars_second_loc, req2)

                reward = (real_requests_first_loc + real_requests_second_loc) * RENT_CREDIT
                
                if num_cars_first_loc > 10:
                    reward += PARKING_COST
                if num_cars_second_loc > 10:
                    reward += PARKING_COST

                num_cars_first_loc -= real_requests_first_loc
                num_cars_second_loc -= real_requests_second_loc
                
                prob = poisson_prob(req1, E_REQ_1) * poisson_prob(req2, E_REQ_2)

                for ret1 in range(MAX):
                    for ret2 in range(MAX):

                        num_cars_first_loc_ = min(num_cars_first_loc + ret1, MAX_CARS)
                        num_cars_second_loc_ = min(num_cars_second_loc + ret2, MAX_CARS)

                        prob_ = poisson_prob(ret1, E_RET_1) * poisson_prob(ret2, E_RET_2) * prob

                        expected_return += prob_ * (reward + self.gamma * values[num_cars_first_loc_, num_cars_second_loc_])

        return expected_return


    
    def policy_evaluation(self, values, policy):
        k = np.arange(MAX_CARS + 1)

        states = []
        for i in k:
            for j in k:
                states.append((i, j))
        
        while True:
            new_values = np.copy(values)
           
            with mp.Pool(processes=self.n_parallel) as p:
                foo = partial(self.expected_return_pe, values, policy)
                results = p.map(foo, states)

            for v, i, j in results:
                new_values[i, j] = v

            difference = np.abs(new_values - values).sum()
            values = new_values
            print("The difference between the values is ", difference)
            
            if difference < self.delta:
                print("The values have converged!")
                return new_values
   
    def policy_improvement(self, values, policy, actions):
        new_policy = np.copy(policy)

        expected_action_returns = np.zeros((MAX_CARS+1, MAX_CARS+1, np.size(actions)))

        k = np.arange(MAX_CARS + 1)
        states = []
        for i in k:
            for j in k:
                states.append((i, j))

        func = dict()

        with mp.Pool(processes=self.n_parallel) as p:
            for action in actions:
                func[action] = partial(self.expected_return_pi, values, action)
                results = p.map(func[action], states)

                for v, i, j, a in results:
                    expected_action_returns[i, j, self.a2i[a]] = v

        for i in range(expected_action_returns.shape[0]):
            for j in range(expected_action_returns.shape[1]):
                new_policy[i, j] = actions[np.argmax(expected_action_returns[i, j])]

        policy_change = (new_policy != policy).sum()
        print("Number policies changed: ", policy_change)

        return policy_change, new_policy


    def expected_return_pe(self, values, policy, state):
        action = policy[state[0], state[1]]
        expected_return = self.bellman(values, action, state)
        
        return expected_return, state[0], state[1]

    def expected_return_pi(self, values, action, state):
        
        if ((state[0] < action) or (state[1] < -action)):
            return -np.inf, state[0], state[1], action
        else:
            expected_return = self.bellman(values, action, state)
            return expected_return, state[0], state[1], action
    
    def plot(self, final=False):

        if final:
            X = np.arange(0, self.values.shape[0])
            Y = np.arange(0, self.values.shape[1])
            X, Y = np.meshgrid(X, Y)

            fig = plt.figure(figsize=(20, 15))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot_wireframe(Y, X, self.values, cstride=1, rstride=1)
            
            ax.set_title("Optimal Value Function", fontsize=15)
            ax.set_ylabel("#Cars at first location", fontsize=15)
            ax.set_xlabel("#Cars at second location", fontsize=15)
        
            plt.savefig("./car/Optimal Value Function.png") 
            plt.close()
            plt.clf()
        else:
            fig = sns.heatmap(np.flipud(self.policy), cmap="YlGnBu")
            fig.set_ylabel('# cars at first location', fontsize=12)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel('# cars at second location', fontsize=12)
            fig.set_title('policy {}'.format(self.iterations), fontsize=20)
            fig.figure.savefig("./car/Policy {}.png".format(self.iterations))
            plt.close()
            plt.clf()

if __name__ == "__main__":
    solver = PolicyIteration()
    solver.solve()
    solver.plot(final=True)

