import numpy as np
import matplotlib.pyplot as plt
import tqdm
import copy


def e_greedy(state, q_value, epsilon):
    if np.random.rand() >= epsilon:
        values = q_value[state]
        action_choices = [action for action, value_ in enumerate(values) if \
                          value_ == np.max(values)]
        return np.random.choice(action_choices)
    else:
        return np.random.choice([a for a, _ in enumerate(q_value[state])])


if __name__ == "__main__":
    STATE_A = 0
    STATE_B = 1
    STATE_T = 2

    LEFT = 0
    RIGHT = 1
    ACTIONS_A = np.array([LEFT, RIGHT])

    ACTIONS_B = np.arange(0, 10)

    q_values = [np.zeros(len(ACTIONS_A)), np.zeros(len(ACTIONS_B)), np.zeros(1)]

    runs = int(1e4)
    eps = 300
    epsilon = 0.1
    alpha = 0.1
    gamma = 1.0

    q_perc_left_actions = np.zeros((runs, eps), dtype=np.float)
    dq_perc_left_actions = np.zeros((runs, eps), dtype=np.float)
    
    for r in tqdm.tqdm(range(runs)):
        
        #Q-Learning
        q = copy.deepcopy(q_values)
        #Double Q-Learning
        dq1 = copy.deepcopy(q_values)
        dq2 = copy.deepcopy(q_values)
        
        for e in range(eps):
            #Variables with subscript q are for Q-Learning
            #Variables with subscript dq are for Double Q-Learning
            state_q = STATE_A
            reward_q = 0
            
            state_dq = STATE_A
            reward_dq = 0
            
            #Q-Learning Episode
            while state_q != STATE_T:
                action_q = e_greedy(state_q, q, epsilon)
                
                #Transitions for State A
                if state_q == STATE_A and action_q == LEFT:
                    q_perc_left_actions[r, e] += 1
                    next_state_q = STATE_B
                    reward_q = 0
                elif state_q == STATE_A and action_q != LEFT:
                    next_state_q = STATE_T
                    reward_q = 0
                
                #Transitions for State B
                if state_q == STATE_B:
                    next_state_q = STATE_T
                    reward_q = np.random.randn()*1 + (-0.1)
                
                #Q-Update
                G_q = reward_q + gamma*np.max(q[next_state_q])
                q[state_q][action_q] += alpha*(G_q - q[state_q][action_q])
                
                #Update State
                state_q = next_state_q
            
            #Double Q-Leanrning Episode
            while state_dq != STATE_T:
                joint_q = [x/2 for x in (dq1+dq2)]
                action_dq = e_greedy(state_dq, joint_q, epsilon)
            
                #Transitions for State A
                if state_dq == STATE_A and action_dq == LEFT:
                    dq_perc_left_actions[r, e] += 1
                    next_state_dq = STATE_B
                    reward_dq = 0
                elif state_dq == STATE_A and action_dq != LEFT:
                    next_state_dq = STATE_T
                    reward_dq = 0
                
                #Transitions for State B
                if state_dq == STATE_B:
                    next_state_dq = STATE_T
                    reward_dq = np.random.randn()*1 + (-0.1)
                
                #Chose which q_value will used for what.
                if np.random.rand() >= 0.5:
                    action_q = dq2
                    val_q = dq1
                else:
                    action_q = dq1
                    val_q = dq2
                
                #Double Q-Update
                values = val_q[next_state_dq]
                best_action = np.random.choice([action for action, val_ in\
                                                enumerate(values) if val_ == np.max(values)])
                
                G_dq = reward_dq + gamma*action_q[next_state_dq][best_action]
                val_q[state_dq][action_dq] += alpha*(G_dq - val_q[state_dq][action_dq])
                
                #Update State
                state_dq = next_state_dq
    
    #Calculate the percentages.
    q_perc_left_actions = np.mean(q_perc_left_actions, axis=0)*100
    dq_perc_left_actions = np.mean(dq_perc_left_actions, axis=0)*100
    
    #Plotting the graph
    plt.figure()
    plt.plot(q_perc_left_actions, label="Q-Learning")
    plt.plot(dq_perc_left_actions, label="Double Q-Learning")
    plt.plot(np.ones(eps)*5, ls="--", label="Optimal")
    plt.xlabel("Episodes")
    plt.ylabel("% Left Action from A")
    plt.legend()
    plt.title("Maximization Bias")
    plt.savefig("Maximization_Bias_Plot.png")
    plt.close()
