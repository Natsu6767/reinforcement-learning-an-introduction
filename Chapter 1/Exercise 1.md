**Exercise 1.1: Self-Play**

*Suppose, instead of playing against a random opponent, the reinforcement learning algorithm described above played against itself, with both sides learning. What do you think would happen in this case? Would it learn a different policy for selecting moves?*

It would learn a different policy for selecting moves. Since, the opponent is not static and also learning (the rl algorithm itself), I think it should be able to reach the optimal policy for playing the game: a nash equilibrium? Also, I think another problem could arise where the policy never converges to an optimal but keeps bouncing back and forth between two similar policies.



**Exercise 1.2: Symmetries**

*Many tic-tac-toe positions appear different but are really the same because of symmetries. How might we amend the learning process described above to take advantage of this? In what ways would this change improve the learning process? Now think again. Suppose the opponent did not take advantage of symmetries. In that case, should we? Is it true, then, that symmetrically equivalent positions should necessarily have the same value?**

To cater to the symmetries we could set the value of all such positions to be the same. This should help improve the learning process because no there are a lesser number of unique states which the algorithm needs to evaluate. If the opponent did not consider symmetries, we should also do the same. This is because there might be state which leads the opponent to perform bad moves, even though this state is symmetric to another state. Therefore, not using symmetry, would allow us to take advantage leading to a possible win. As a result, this state should have a higher value compared to the other symmetric states.



**Exercise 1.3: Greedy Play**

*Suppose the reinforcement learning player was greedy, that is, it always played the move that brought it to the position that it rated the best. Might it learn to play better, or worse, than a nongreedy player? What problems might occur?*

A greedy policy will make the reinforcement learning player learn an equivalent or worse policy than a normal reinforcement player which allows for exploratory moves. Since a greedy player only aims to maximize it's immediate reward, this doesn't necessarily mean that the overall reward for such a player would be high and hence does not guarantee a win.



**Exercise 1.4: Learning from Exploration**

*Suppose learning updates occurred after all moves, including exploratory moves. If the step-size parameter is appropriately reduced over time (but not the tendency to explore), then the state values would converge to a set of probabilities. What are the two sets of probabilities computed when we do, and when we do not, learn from exploratory moves? Assuming that we do continue to make exploratory moves, which set of probabilities might be better to learn? Which would result in more wins?*

Updating the value even after performing exploratory moves would only make sense if our current policy (greedy action) was sub-optimal. Otherwise, this type of update would lead us to underestimate the value of the state. Hence, the value function obtained from updates even after exploratory moves would be a sort of lower bound on the optimal value function. If we continue to make exploratory moves it would be best that we do not update when we perform an exploratory move. If the exploratory move was better than the greedy action defined by our policy, the update to the value of the new state could make this exploratory action  the best (greedy) action for the starting state. This would result in more wins.

