import numpy as np
import matplotlib.pyplot as plt
import tqdm

class RandomWalk:
    def __init__(self, n_states=5, start=2):
        self.n_states = n_states
        self.start = start
        self.cur_state = start


    def reset(self):
        self.cur_state = self.start

    def step(self):
        reward = 0
        end = False
        state = self.cur_state

        if np.random.rand() >= 0.5:
            self.cur_state += 1
        else:
            self.cur_state -= 1

        if self.cur_state == self.n_states:
            reward = 1
            end = True

        if self.cur_state < 0:
            end = True

        return end, reward, state, self.cur_state


def left_6_2(eps=100, alpha=0.1, gamma=1.0):
    rwalk = RandomWalk()
    sv_list = list()

    state_values = np.repeat(0.5, rwalk.n_states)
    sv_list.append(state_values.copy())

    for e in tqdm.tqdm(range(eps)):
        rwalk.reset()
        if e in [1, 10]:
            sv_list.append(state_values.copy())

        while True:
            end, reward, i_state, f_state = rwalk.step()
            
            if end:
                G = reward + gamma*0
                state_values[i_state] += alpha*(G - state_values[i_state])
                break

            G = reward + gamma*state_values[f_state]
            state_values[i_state] += alpha*(G - state_values[i_state])

    sv_list.append(state_values.copy())
    
    #Ploting
    labels = ["0 eps", "1 eps", "10 eps", "100 eps"]

    plt.figure()
    
    y_true = [1/6, 2/6, 3/6, 4/6, 5/6]
    plt.plot(y_true, marker='^', label="True")

    for val, label in zip(sv_list, labels):
        plt.plot(val, marker="8", label=label)
    
    
    plt.xticks(range(5), ['A', 'B', 'C', 'D', 'E'])
    plt.xlabel("State")
    plt.ylabel("Estimated Value")
    plt.legend()
    plt.savefig("Random_Walk_TD_Plots.png")


def right_6_2(runs=100, eps=100, gamma=1.0):
    rwalk = RandomWalk()
    true_state_values = np.array([1/6, 2/6, 3/6, 4/6, 5/6])
    
    #Temporal Difference
    alphas_td = [0.15, 0.10, 0.05]
    td_sv_avg_error = list()

    for alpha in tqdm.tqdm(alphas_td):
        td_sv_alpha = np.zeros((runs, eps), dtype=np.float)

        for r in range(runs):
            td_sv = np.ones(rwalk.n_states, dtype=np.float)*0.5
            for e in range(eps):
                
                rwalk.reset()

                while True:
                    end, reward, i_state, f_state = rwalk.step()

                    if end:
                        G = reward + gamma*0
                        td_sv[i_state] += alpha*(G - td_sv[i_state])
                        break

                    G = reward + gamma*td_sv[f_state]
                    td_sv[i_state] += alpha*(G - td_sv[i_state])

                td_sv_alpha[r, e] = np.sqrt(np.mean(np.square(td_sv - true_state_values)))
            
        
        td_sv_avg_error.append(np.mean(td_sv_alpha, axis=0))
    
    #Monte-Carlo
    alphas_mc = [0.04, 0.03, 0.02, 0.01]
    mc_sv_avg_error = list()

    for alpha in tqdm.tqdm(alphas_mc):
        mc_sv_alpha = np.zeros((runs, eps), dtype=np.float)

        for r in range(runs):
            mc_sv = np.ones(rwalk.n_states, dtype=np.float)*0.5
            
            for e in range(eps):    
                rwalk.reset()
                trajectory = list()
                
                while True:
                    end, reward, state, _ = rwalk.step()
                    
                    trajectory.append([state, reward])
                    
                    if end:
                        break

                G = 0.0
                for state, reward in reversed(trajectory):
                    G = reward + gamma*G
                    mc_sv[state] += alpha*(G - mc_sv[state])
                
                mc_sv_alpha[r, e] = np.sqrt(np.mean(np.square(mc_sv - true_state_values)))
        
        mc_sv_avg_error.append(np.mean(mc_sv_alpha, axis=0))

    #Plotting
    plt.figure()
    
    for val, label in zip(td_sv_avg_error, alphas_td):
        plt.plot(val, label="TD Alpha="+str(label))
    
    for val, label in zip(mc_sv_avg_error, alphas_mc):
        plt.plot(val, ls='--', label="MC Alphas="+str(label))
    
    plt.xlabel("Walk/Episodes")
    plt.ylabel("Empirical RMS error, averaged over states")
    plt.legend()
    plt.savefig("Random_Walk_TD_MC_Compare.png")

def figure_6_2(runs=100, eps=100, gamma=1.0):
    rwalk = RandomWalk()
    true_state_values = np.array([1/6, 2/6, 3/6, 4/6, 5/6])
    
    #Temporal Difference
    alpha = 0.001
    td_sv_avg_error = np.zeros((runs, eps), dtype=np.float)*0.5
    
    for r in tqdm.tqdm(range(runs)):
        batch_trajectory = list()
        td_state_values = np.ones(rwalk.n_states, dtype=np.float)*0.5
        for e in range(eps):
            rwalk.reset()

            while True:
                end, reward, i_state, f_state = rwalk.step()
                
                batch_trajectory.append([end, reward, i_state, f_state])
                if end:
                    break
            
            increments = np.zeros_like(td_state_values)
            while True:
                old_sv = td_state_values
                for end, reward, i_state, f_state in batch_trajectory:
                    if end:
                        G = reward + gamma*0
                    else:
                        G = reward + gamma*td_state_values[f_state]

                    increments[i_state] += (G - td_state_values[i_state])

                increments *= alpha
                if(np.sum(np.abs(increments)) < 1e-3):
                    break
                td_state_values += increments

            td_sv_avg_error[r, e] = np.sqrt(np.mean(np.square(td_state_values - true_state_values)))

    td_sv_avg_error = np.mean(td_sv_avg_error, axis=0)
    
    #Monte-Carlo
    mc_sv_avg_error = np.zeros((runs, eps), dtype=np.float)

    for r in tqdm.tqdm(range(runs)):
        batch_trajectory = list()
        mc_state_values = np.ones(rwalk.n_states, dtype=np.float)*0.5
        
        for e in range(eps):
            ep_trajectory = list()
            rwalk.reset()

            while True:
                end, reward, state, _ = rwalk.step()
                ep_trajectory.append([reward, state])
                if end:
                    break
            
            batch_trajectory.append(ep_trajectory)
            
            while True:
                increments = np.zeros_like(mc_state_values)
                for trajectory in batch_trajectory:
                    G = 0.0
                    for reward, state in reversed(trajectory):
                        G = reward + gamma*G
                        increments[state] += (G - mc_state_values[state])
                
                increments *= alpha
                if(np.sum(np.abs(increments)) < 1e-3):
                    break
                mc_state_values += increments
            mc_sv_avg_error[r, e] = np.sqrt(np.mean(np.square(mc_state_values - true_state_values)))

    mc_sv_avg_error = np.mean(mc_sv_avg_error, axis=0)
    
    plt.figure()
    plt.plot(td_sv_avg_error, label="TD Alpha="+str(alpha))
    plt.plot(mc_sv_avg_error, ls="--", label="MC Alpha="+str(alpha))
   
    plt.title("Batch Training")
    plt.xlabel("Walks/Episodes")
    plt.ylabel("RMS error, averaged over states")
    plt.legend()
    plt.savefig("Batch_Training_Random_Walk.png")



if __name__ == "__main__":
    left_6_2()
    right_6_2()
    figure_6_2()
