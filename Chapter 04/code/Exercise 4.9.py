import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from functools import partial


if __name__ == "__main__":
    
    GOAL = 100
    values = np.zeros(GOAL+1)
    values[GOAL] = 1.0
    HEAD_PROB = 0.55
    STATES = np.arange(GOAL + 1)
    
    sweeps = []
    iterations = 0
    while True:
        new_values = np.copy(values)

        sweeps.append((iterations, values))
        
        for state in STATES[1 : GOAL]:
            actions = np.arange(min(state, GOAL - state) + 1)
            action_return = []
            
            for a in actions:
                action_return.append(
                HEAD_PROB*values[state + a] + (1 - HEAD_PROB)*values[state - a])

            new_values[state] = np.max(action_return)


        difference = np.abs(new_values - values).sum()
        print("Difference between values: ", difference)
        values = new_values
        iterations += 1
        if difference < 1e-6:
            print("Values have Converged!")
            sweeps.append((iterations, values))
            break

    policy = np.zeros(GOAL + 1)
    
    intervals = iterations // 10
    sweeps_ = list()
    k = -intervals
    while k < iterations-intervals:
        k += intervals
        sweeps_.append(sweeps[k])
        

    if not k == iterations -1:
        sweeps_.append(sweeps[iterations-1])

    sweeps = sweeps_

    for state in STATES[1 : GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        action_return = []

        for a in actions:
            action_return.append(
            HEAD_PROB*values[state + a] + (1 - HEAD_PROB)*values[state - a])

        policy[state] = actions[np.argmax(np.round(action_return[1:], 5)) + 1]

    print("Optimal Policy Found!")
    print(policy)

    fig = plt.figure(figsize=(20, 30))
    fig.suptitle("Head Probability: {}".format(HEAD_PROB), fontsize=30)
    plt.subplot(2, 1, 1)
    for sweep, state_value in sweeps:
        plt.plot(state_value, label='Sweep {}'.format(sweep))
    plt.xlabel('Capital', fontsize=25)
    plt.ylabel('Value estimates', fontsize=25)
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel("Capital", fontsize=25)
    plt.ylabel("Final Policy (stake)", fontsize=25)

    plt.savefig("Gambler_055.png")
    plt.close()
