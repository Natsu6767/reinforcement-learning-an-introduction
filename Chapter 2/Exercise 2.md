**Exercise 2.1**

*In ![equation](https://latex.codecogs.com/png.latex?\varepsilon)-greedy action selection, for the case of two actions and ![equation](https://latex.codecogs.com/png.latex?\varepsilon) = 0.5, what is
the probability that the greedy action is selected?*

![equation](https://latex.codecogs.com/png.latex?P(greedy)&space;=&space;P(pick\,greedy|exploit)P(exploit)&space;&plus;&space;P(pick\,greedy|exploration)P(exploration)\newline~~~~~~~~~~~~~~~~~~~&space;=&space;1*(1-\varepsilon)&space;&plus;&space;\frac{1}{A}*\varepsilon\newline~~~~~~~~~~~~~~~~~~~&space;=&space;1&space;-&space;\frac{(A&space;-&space;1)}{A}\varepsilon)

Since there are 2 actions, A = 2 . Hence,

![equation](https://latex.codecogs.com/png.latex?P(greedy)&space;=&space;0.75)



**Exercise 2.2:**

*Bandit example Consider a k-armed bandit problem with k = 4 actions,
denoted 1, 2, 3, and 4. Consider applying to this problem a bandit algorithm using
![equation](https://latex.codecogs.com/png.latex?\varepsilon)-greedy action selection, sample-average action-value estimates, and initial estimates
of Q<sub>1</sub>(a) = 0, for all a. Suppose the initial sequence of actions and rewards is A<sub>1</sub> = 1,
R<sub>1</sub> = −1, A<sub>2</sub> = 2, R<sub>2</sub> = 1, A<sub>3</sub> = 2, R<sub>3</sub> = −2, A<sub>4</sub> = 2, R<sub>4</sub> = 2, A<sub>5</sub> = 3, R<sub>5</sub> = 0. On some
of these time steps the ![equation](https://latex.codecogs.com/png.latex?\varepsilon) case may have occurred, causing an action to be selected at
random. On which time steps did this definitely occur? On which time steps could this
possibly have occurred?*

Time steps at which ![equation](https://latex.codecogs.com/png.latex?\varepsilon) case definitely occurred: 3, 4, 5.

Time steps at which ![equation](https://latex.codecogs.com/png.latex?\varepsilon) case may have occurred: All time steps.

At time step 2 since Q<sub>2</sub>(1) = -1 < Q<sub>2</sub>(2) = 0, A<sub>2</sub> = 2 could simply have been picked due to greedy policy. Since, all the other actions also have Q<sub>2</sub> = 0, the choice of action would have been random.



**Exercise 2.3**

*In the comparison shown in Figure 2.2, which method will perform best in
the long run in terms of cumulative reward and probability of selecting the best action?
How much better will it be? Express your answer quantitatively.*

The ![equation](https://latex.codecogs.com/png.latex?\varepsilon)-greedy method with ![equation](https://latex.codecogs.com/png.latex?\varepsilon)= 0.01 would perform better. However, it would take a long time to reach the optimal solution. Once this method has found the optimal solution it would exploit it 99% (1 - ![equation](https://latex.codecogs.com/png.latex?\varepsilon)) of the time. On the other hand, for ![equation](https://latex.codecogs.com/png.latex?\varepsilon)= 0.1, it would only exploit the optimal solution only 90% of the time.



**Exercise 2.4**

*If the step-size parameters α<sub>n</sub> are not constant, then the estimate Q<sub>n</sub> is a weighted average of previously received rewards with a weighting different from that given by **Q<sub>n+1</sub>= Q<sub>n</sub>+ α[R<sub>n</sub>− Q<sub>n</sub>]**. What is the weighting on each prior reward for the general case, analogous to the above, in terms of the sequence of step-size parameters?*

![equation](https://latex.codecogs.com/png.latex?Q_{n&plus;1}&space;=&space;Q_{n}&space;&plus;&space;\alpha_{n}(R_{n}&space;-&space;Q_{n})\newline~~~~~~~~~~~~&space;=&space;(1&space;-&space;\alpha_{n})Q_{n}&space;&plus;&space;\alpha_{n}R_{n}\newline~~~~~~~~~~~~&space;=&space;(1&space;-&space;\alpha_{n})[Q_{n-1}&space;&plus;&space;\alpha_{n-1}(R_{n-1}&space;-&space;Q_{n-1})]&space;&plus;&space;\alpha_{n}R_{n}\newline~~~~~~~~~~~~&space;=&space;(1&space;-&space;\alpha_{n})[(1&space;-&space;\alpha_{n-1})Q_{n-1}&space;&plus;&space;\alpha_{n-1}R_{n-1}]&space;&plus;&space;\alpha_{n}R_{n}\newline~~~~~~~~~~~~&space;=&space;(1&space;-&space\alpha_{n})(1&space;-&space;\alpha_{n-1})Q_{n-1}&space;&plus;&space;\alpha_{n-1}R_{n-1}(1&space;-&space;\alpha_{n})&space;&plus;&space;\alpha_{n}R_{n}\newline~~~~~~~~~~~~&space;=&space;Q_{1}\prod_{i&space;=&space;1}^{n}(1&space;-&space;\alpha_{i})&space;&plus;&space;\sum_{j&space;=&space;1}^{n}\alpha_{j}R_{j}\prod_{j&space;=&space;i}^{n&space;-&space;1}(1&space;-&space;\alpha_{j&space;&plus;&space;1}))

If we take constant α for each time step, the expression becomes:

![equation](https://latex.codecogs.com/png.latex?Q_{n&plus;1}&space;=&space;Q_{1}(1&space;-&space;\alpha)^{n}&space;&plus;&space;\alpha\sum_{i&space;=&space;1}^{n}(1&space;-&space;\alpha)^{n&space;-&space;1}R_{i})



**Exercise 2.6: Mysterious Spikes**

*The results shown in Figure 2.3 should be quite reliable because they are averages over 2000 individual, randomly chosen 10-armed bandit tasks. Why, then, are there oscillations and spikes in the early part of the curve for the optimistic method? In other words, what might make this method perform particularly better or worse, on average, on particular early steps?*

In the optimistic method all the unique actions end up getting explored fast since our initial estimation of the rewards are too high. When an action is selected, the update to our estimation for the selected action gets reduced significantly. Therefore, even greedy methods end up exploring. Once all the actions have been explored, the action with the here true reward gets selected more frequently. Since, most of the 2000 agents finish exploring around the same time, they start selecting the same action with the highest true reward. This causes a spike in the early part of the curve. As this action gets selected multiple times, our reward estimation of it reduces. This results in another action having a higher reward estimate and that action now gets selected albeit, by lesser number of the 2000 agents (greater chance that a different action is selected by different agents due to the sampling step in getting the true reward). This continues for multiple actions and causes the oscillations and spikes in the early part of the curve.



**Exercise 2.8: UCB Spikes**

*In Figure 2.4 the UCB algorithm shows a distinct spike in performance on the 11th step. Why is this? Note that for your answer to be fully satisfactory it must explain both why the reward increases on the 11th step and why it decreases on the subsequent steps. Hint: if c = 1, then the spike is less prominent.*

The UCB algorithm forces the agents to explore all the actions at least once. In the first 10 steps, the agents try out all the different unique 10 actions. This results in having created new reward estimations for the actions in which one action clearly has the higher reward estimate on average (the action with the highest true reward mean). Since the count of each action having occurred is the same i.e. 1, and the time step is also the same i.e. 10, only the first term in the UCB algorithm is used to decide which action to be chosen. Since this is just the reward estimations for the action, the action with the highest reward is chosen. All 2000 agents, on average, also pick the same action, causing the spike in performance on the 11th step.



**Exercise 2.9**

*Show that in the case of two actions, the soft-max distribution is the same as that given by the logistic, or sigmoid, function often used in statistics and artificial neural networks.*

For two actions ***a*** and ***b***. The probability of the action ***a*** occurring using the soft-max distribution is:

![equation](https://latex.codecogs.com/png.latex?P%28A_%7Bt%7D%3Da%29%20%3D%20%5Cfrac%7Be%5E%7BH_%7Bt%7D%28a%29%7D%7D%7Be%5E%7BH_%7Bt%7D%28a%29%7D%20&plus;%20e%5E%7BH_%7Bt%7D%28b%29%7D%7D)

Now, using the property of soft-max, which allows us to subtract some quantity from the parameters and the predictions still remains the same. Hence, subtracting ![equation](https://latex.codecogs.com/png.latex?H_%7Bt%7D%28b%29) from both the parameters, we get:

![equation](https://latex.codecogs.com/png.latex?P%28A_%7Bt%7D%3Da%29%20%3D%20%5Cfrac%7Be%5E%7BH_%7Bt%7D%28a%29%20-%20H_%7Bt%7D%28b%29%7D%7D%7Be%5E%7BH_%7Bt%7D%28a%29%20-%20H_%7Bt%7D%28b%29%7D%20&plus;%20e%5E%7BH_%7Bt%7D%28b%29%20-%20H_%7Bt%7D%28b%29%7D%7D%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%3D%20%5Cfrac%7Be%5E%7B%28H_%7Bt%7D%28a%29%20-%20H_%7Bt%7D%28b%29%29%7D%7D%7Be%5E%7B%28H_%7Bt%7D%28a%29%20-%20H_%7Bt%7D%28b%29%29%7D%20&plus;%20e%5E%7B0%7D%7D%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%3D%20%5Cfrac%7Be%5E%7B%5Ctheta_%7Bt%7D%20%7D%7D%7Be%5E%7B%5Ctheta_%7Bt%7D%7D%20&plus;%201%7D%7E%7E%7E%7E%3B%20%5Ctheta_%7Bt%7D%20%3D%20e%5E%7B%28H_%7Bt%7D%28a%29%20-%20H_%7Bt%7D%28b%29%29%7D%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%7E%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-%5Ctheta_%7Bt%7D%7D%7D)

Hence, we obtain the sigmoid function.



**Exercise 2.10**

*Suppose you face a 2-armed bandit task whose true action values change randomly from time step to time step. Specifically, suppose that, for any time step, the true values of actions 1 and 2 are respectively 0.1 and 0.2 with probability 0.5 (case A), and 0.9 and 0.8 with probability 0.5 (case B). If you are not able to tell which case you face at any step, what is the best expectation of success you can achieve and how should you behave to achieve it? Now suppose that on each step you are told whether you are facing case A or case B (although you still don’t know the true action values). This is an associative search task. What is the best expectation of success you can achieve in this task, and how should you behave to achieve it?*

If we are not able to tell which case we face:

Expected reward on selecting Action 1 always: ![equation](https://latex.codecogs.com/png.latex?0.5\times0.1&space;&plus;&space;0.5\times0.9&space;=&space;0.5)

Expected reward on Selecting Action 2 always: ![equation](https://latex.codecogs.com/png.latex?0.5\times0.2&space;&plus;&space;0.5\times0.8&space;=&space;0.5)

Since both actions have the same value estimates, any action can be selected and on average we would get the same reward ultimately.

If we are told which case we are facing, it makes sense to always chose the action that gives the highest reward for each case. For Case A that would be Action 2 and for Case B that would be Action 1.

The expected reward would then be: ![equation](https://latex.codecogs.com/png.latex?0.5\times0.2&space;&plus;&space;0.5\times0.9&space;=&space;0.55)