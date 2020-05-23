import numpy as np
import matplotlib.pyplot as plt
import tqdm


def simulate(b_steps, sample=True):
    """
    Assume that MDP consits of a single state A, and
    all transitions from this state result in a reward
    r ~ N(1, 1) and lead to the terminal state. The value of the
    terminal is taken as 0 (True Value).
    """

    rewards = np.random.rand(b_steps) + 0.5
    true_state_value = np.mean(rewards)

    errors = list()
    estimate_value = 0
    
    if sample:
        for t in range(1, 2 * b_steps + 1):
            errors.append(np.sqrt(np.mean(np.square(estimate_value - true_state_value))))
            r = np.random.choice(rewards)
            estimate_value += (1 / t) * (r - estimate_value)
    else:
        expectation = 0
        count = 0
        for t in range(1, 2 * b_steps + 1):
            errors.append(np.sqrt(np.mean(np.square(estimate_value - true_state_value)))) 
            expectation += (1 / b_steps) * rewards[t % b_steps]
            
            if not t % b_steps:
                count += 1
                estimate_value += (1 / count) * (expectation - estimate_value)
                expectation = 0
    
    return errors

if __name__ == "__main__":
    runs = 1000
    b_steps = [2, 10, 100, 1000, 10000]

    plt.figure()

    
    for b in tqdm.tqdm(b_steps):
        run_errors = np.zeros((runs, 2*b), dtype=np.float)
        for r in range(runs):
            run_errors[r] = simulate(b, sample=True)

        avg_errors = np.mean(run_errors, axis=0)
        x_axis = np.arange(len(avg_errors), dtype=np.float) / float(b)
        plt.plot(x_axis, avg_errors, label="b = {}".format(b))
    
    b = 100
    run_errors = np.zeros((runs, 2*b), dtype=np.float)
    for r in range(runs):
        run_errors[r] = simulate(b, sample=False)
    avg_errors = np.mean(run_errors, axis=0)
    x_axis = np.arange(len(avg_errors), dtype=np.float) / float(b)
    plt.plot(x_axis, avg_errors, ls="--", label="Expected Update")

    plt.xlabel("Number of Computations")
    plt.ylabel("RMS error in value estimate")
    plt.title("Sample vs Expected Update")
    plt.xticks([0, 1.0, 2.0], ['0', 'b', '2b'])
    plt.legend(loc="best")
    plt.savefig("Sample_vs_Expected_Updates.png")
    plt.close()
