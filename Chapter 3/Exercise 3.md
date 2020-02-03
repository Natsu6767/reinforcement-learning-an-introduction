## Exercise 3.1

*Devise three example tasks of your own that fit into the MDP framework, identifying for each its states, actions, and rewards. Make the three examples as **different** from each other as possible. The framework is abstract and flexible and can be applied in many different ways. Stretch its limits in some way in at least one of your examples.*

Example 1: Playing tag.

- States: Current position and distance and direction of other players around you.
- Actions: Speed at which to run (walk, sprint, etc), direction of running, raising your arms out to make someone "it".
- Rewards: High positive reward (say +100) on tagging another player. A constant negative reward (say -1) for each second that passes.

Example 2:  Playing Blackjack.

- States: Cards in hand and the one card shown by the dealer. (Optional, counting cards, but that would make our bot get banned from casinos.)
- Actions: Stand (not taking any more cards), or Hit (take another card).
- Rewards: The money obtained/lost from winning/going bust.

Example 3:

## Exercise 3.2 

*Is the MDP framework adequate to usefully represent **all** goal-directed learning tasks? Can you think of any clear exceptions?*

The MDP framework is not completely adequate since an important assumption of an MDP is that the current state depends only on the immediately preceding state. This can however be fixed if the we can somehow accommodate all the information of the early states in the state value, however, this may not always be easily possible. Additionally, an MDP requires a single numerical value for a reward. In certain scenarios where the feedback is qualitative, such as a verbal praise "Your singing is wonderful." or "You sing well.". Although both of the previous praises are positive it is difficult to assign a numerical value to them ranking their "rewardiness". Just assigning the same reward for all feedbacks in a category is limiting the amount of information that we can make use of.

## Exercise 3.3

*Consider the problem of driving. You could define the actions in terms of the accelerator, steering wheel, and brake, that is, where your body meets the machine. Or you could define them farther out - say, where the rubber meets the road, considering your actions to be tire torques. Or you could define them farther in - say, where your brain meets your body, the actions being muscle twitches to control your limbs. Or you could go to a really high level and say that your actions are your choices of where to drive. What is the right level, the right place to draw the line between agent and environment? On what basis is one location of the line to be preferred over another? Is there any fundamental reason for preferring one location over another, or is it a free choice?*

The limit should be such that whenever the agent tries to do a particular action, that action should occur in the way every time. Hence, defining it where the body meets the machine is apposite.

## Exercise 3.4

*Give a table analogous to that in Example 3.3, but for p(s',r|s, a). It should have columns for s, a, s', r, and p(s', r|s, a), and a row for every 4-tuple for which p(s',r|s, a) > 0.*

Since there is a single reward defined for each triplet (s, a, s'), the table is the same as the one given in Example 3.3. Just remove the rows with the p(s' | s, a) = 0.

## Exercise 3.5

*The equations in Section 3.1 are for the continuing case and need to be modified (very slightly) to apply to episodic tasks. Show that you know the modifications needed by giving the modified version of (3.3).*

For episodic tasks, only those terminal states should be included. Hence, the modified equation becomes:

![equation](https://latex.codecogs.com/gif.latex?%5Csum_%7Bs%27%20%5Cin%20S%7D%20%5Csum_%7Br%20%5Cin%20R%7Dp%28s%27%2C%20r%20%7C%20s%2C%20a%29%20%3D%201%2C%20%7E%7E%20%5Cforall%20%5C%20s%20%5Cin%20S%5E%7B&plus;%7D%2C%20a%20%5Cin%20A%28s%29.)

## Exercise 3.6

*Suppose you treated pole-balancing as an episodic task but also used discounting, with all rewards zero except for -1 upon failure. What then would the return be at each time? How does this return differ from that in the discounted, continuing formulation of this task?*

The reward at each time step should be 0? I couldn't really understand what the question is talking about. If what the question is asking how much reward we get after each episode, then shouldn't that be either 0 or -1 depending on if the pole was balanced or not?

## Exercise 3.7

*Imagine that you are designing a robot to run a maze. You decide to give it a reward of +1 for escaping from the maze and a reward of zero at all other times. The task seems to break down naturally into episodes - the successive runs through the maze - so you decide to treat it as an episodic task, where the goal is to maximize expected total reward (3.1). After running the learning agent for a while, you find that it is showing no improvement in escaping from the maze. What is going wrong? Have you effectively communicated to the agent what you want it to achieve?*

Since the only feedback (+1 reward) that the robot gets is only when it escapes from the maze, the robot assumes that there is only zero reward everywhere. Unless an episode occurs where the robot escapes, the robot would learn nothing. One way to solve this would be to give a negative reward (say, -1) for each time step. Another alternative is to give a high negative reward near the start location of the robot and progressively reduce the magnitude of this negative reward as the robot gets farther away from the start. Ultimately set a high positive reward (say, +10) on successfully escaping the maze.

## Exercise 3.8

*Suppose ![equation](https://latex.codecogs.com/png.latex?%5Cgamma) = 0.5 and the following sequence of rewards is received ![equation](https://latex.codecogs.com/png.latex?R_%7B1%7D) = -1, ![equation](https://latex.codecogs.com/gif.latex?R_%7B2%7D) = 2, ![equation](https://latex.codecogs.com/gif.latex?R_%7B3%7D) = 6, ![equation](https://latex.codecogs.com/gif.latex?R_%7B4%7D) = 3 and ![equation](https://latex.codecogs.com/gif.latex?R_%7B5%7D) = 2, with T = 5. What are ![equation](https://latex.codecogs.com/gif.latex?G_%7B0%7D), ![equation](https://latex.codecogs.com/gif.latex?G_%7B1%7D), ..., ![equation](https://latex.codecogs.com/gif.latex?G_%7B5%7D)? Hint: Work backwards*.

![equation](https://latex.codecogs.com/png.latex?G_%7B5%7D%20%3D%200%5C%5C%20%7E%7E%7E%7E%7EG_%7B4%7D%20%3D%20R_%7B5%7D%20&plus;%20%5Cgamma%20G_%7B5%7D%20%3D%202%20&plus;%200.5%20%5Ctimes%200%20%3D%202%5C%5C%20%7E%7E%7E%7E%7EG_%7B3%7D%20%3D%20R_%7B4%7D%20&plus;%20%5Cgamma%20G_%7B4%7D%20%3D%203%20&plus;%200.5%20%5Ctimes%202%20%3D%204%5C%5C%20%7E%7E%7E%7E%7EG_%7B2%7D%20%3D%20R_%7B3%7D%20&plus;%20%5Cgamma%20G_%7B3%7D%20%3D%206%20&plus;%200.5%20%5Ctimes%204%20%3D%208%5C%5C%20%7E%7E%7E%7E%7EG_%7B1%7D%20%3D%20R_%7B2%7D%20&plus;%20%5Cgamma%20G_%7B2%7D%20%3D%202%20&plus;%200.5%20%5Ctimes%208%20%3D%206%5C%5C%20%7E%7E%7E%7E%7EG_%7B0%7D%20%3D%20R_%7B1%7D%20&plus;%20%5Cgamma%20G_%7B1%7D%20%3D%20-1%20&plus;%200.5%20%5Ctimes%206%20%3D%202)

## Exercise 3.9

Suppose ![equation](https://latex.codecogs.com/png.latex?%5Cgamma)=0.9 and the reward sequence is ![equation](https://latex.codecogs.com/png.latex?R_%7B1%7D) = 2 followed by an infinite sequence of 7s. What are ![equation](https://latex.codecogs.com/png.latex?G_%7B1%7D) and ![equation](https://latex.codecogs.com/png.latex?G_%7B0%7D)?

The rewards are 2, 7, 7, 7, 7, 7, 7, . . .

Thus,

![equation](https://latex.codecogs.com/png.latex?G_%7B1%7D%20%3D%20R_%7B2%7D%20&plus;%20%5Csum_%7Bk%20%3D%201%7D%5E%7B%5Cinfty%7D%5Cgamma%5E%7Bk%7DR_%7Bk%20&plus;%202%7D%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%3D%20R%20&plus;%20R%20%5Csum_%7Bk%20%3D%201%7D%5E%7B%5Cinfty%7D%5Cgamma%5E%7Bk%7D%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%3D%20R%281%20&plus;%20%5Csum_%7Bk%20%3D%201%7D%5E%7B%5Cinfty%7D%5Cgamma%5E%7Bk%7D%29%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%3D%20R%28%5Cgamma%5E%7B0%7D%20&plus;%20%5Csum_%7Bk%20%3D%201%7D%5E%7B%5Cinfty%7D%5Cgamma%5E%7Bk%7D%29%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%3D%20R%5Csum_%7Bk%20%3D%200%7D%5E%7B%5Cinfty%7D%5Cgamma%5E%7Bk%7D%5C%5C%20%7E%7E%7E%7E%7E%7E%7E%7E%7E%3D%20R%5Ctfrac%7B1%7D%7B1-%5Cgamma%7D%20%3D%207%5Ctfrac%7B1%7D%7B1%20-%200.9%7D%20%3D%207%20%5Ctimes%2010%20%3D%2070)

and,

![equation](https://latex.codecogs.com/png.latex?G_%7B0%7D%20%3D%20R_%7B1%7D%20&plus;%20%5Cgamma%20G_%7B1%7D%20%3D%202%20&plus;%200.9%20%5Ctimes%2070%20%3D%2065)

## Exercise 3.10

Prove the second equality in (3.10).

![equation](https://latex.codecogs.com/png.latex?G_%7Bt%7D%20%3D%20%5Csum_%7Bk%20%3D%200%7D%5E%7B%5Cinfty%7D%5Cgamma%5E%7Bk%7D%5C%5C%20%5CRightarrow%20%7E%7EG_%7Bt%7D%20%3D%201%20&plus;%20%5Cgamma%20&plus;%20%5Cgamma%5E%7B2%7D&plus;%20%5Cgamma%5E%7B3%7D&plus;%20%5Cgamma%5E%7B4%7D&plus;%20%5Cgamma%5E%7B5%7D%5Ccdots%7E%7E%7E%7E%7E%281%29%5C%5C%20%5CRightarrow%20%5Cgamma%20G_%7Bt%7D%20%3D%7E%7E%7E%7E%7E%5Cgamma%20&plus;%20%5Cgamma%5E%7B2%7D&plus;%20%5Cgamma%5E%7B3%7D&plus;%20%5Cgamma%5E%7B4%7D&plus;%20%5Cgamma%5E%7B5%7D%5Ccdots%7E%7E%7E%7E%7E%7E%282%29%5C%5C%20Now%2C%7E%281%29%20-%20%282%29%20%7E%20gives%3A%5C%5C%20G_%7Bt%7D%20-%20%5Cgamma%20G_%7Bt%7D%20%3D%201%20&plus;%200%20&plus;%200%20&plus;%200%20&plus;%200%20&plus;%200%5Ccdots%5C%5C%20%5CRightarrow%20G_%7Bt%7D%281%20-%20%5Cgamma%29%20%3D%201%5C%5C%20%5Ctherefore%20G_%7Bt%7D%20%3D%20%5Cfrac%7B1%7D%7B1%20-%20%5Cgamma%7D%20%7E%7E%7E%7E%7E%7E%7E%7E%20%5Cmathit%7BQ.E.D.%7D)

## Exercise 3.11

If the current state is ![equation](https://latex.codecogs.com/png.latex?S_%7Bt%7D), and actions are selected according to stochastic policy ![equation](https://latex.codecogs.com/png.latex?%5Cpi), then what is the expectation of ![equation](https://latex.codecogs.com/png.latex?R_%7Bt&plus;1%7D) in terms of ![equation](https://latex.codecogs.com/png.latex?%5Cpi) and the four-argument function ![equation](https://latex.codecogs.com/png.latex?p)(3.2)?

The four-argument function, ![equation](https://latex.codecogs.com/png.latex?p):

![equation](https://latex.codecogs.com/png.latex?p%28s%27%2C%20r%20%7C%20s%2C%20a%29%20%3D%20Pr%5C%7BS_%7Bt%7D%20%3D%20s%27%2C%20R_%7Bt%7D%20%3D%20r%20%7C%20S_%7Bt%20-%201%7D%20%3D%20s%2C%20A_%7Bt%20-%201%7D%20%3D%20a%5C%7D)

From this we can calculate the expected reward for state-action pairs as:

![equation](https://latex.codecogs.com/png.latex?r%28s%2C%20a%29%20%3D%20%5Cmathbb%7BE%7D%5BR_%7Bt%7D%20%7C%20S_%7Bt%20-%201%7D%20%3D%20s%2C%20A_%7Bt%20-%201%7D%20%3D%20a%5D%20%3D%20%5Csum_%7Br%20%5Cin%20R%7Dr%20%5Csum_%7Bs%27%20%5Cin%20S%7Dp%28s%27%2C%20r%20%7Cs%2C%20a%29%2C)

Hence the expectation of ![equation](https://latex.codecogs.com/png.latex?R_%7Bt&plus;1%7D) is given by:

![equation](https://latex.codecogs.com/png.latex?%5Cmathbb%7BE%7D%5BR_%7Bt&plus;1%7D%20%7C%20S_%7Bt%7D%20%3D%20s%2C%20%5Cpi%5D%20%3D%20%5Csum_%7Bs%20%5Cin%20S%7D%5Cpi%28a%20%7C%20s%29r%28s%2C%20a%29)

## Exercise 3.12

Give an equation for ![equation](https://latex.codecogs.com/png.latex?v_%7B%5Cpi%7D) in terms of ![equation](https://latex.codecogs.com/png.latex?q_%7B%5Cpi%7D) and ![equation](https://latex.codecogs.com/png.latex?%5Cpi).

![equation](https://latex.codecogs.com/png.latex?v_%7B%5Cpi%7D%20%3D%20%5Csum_%7Ba%20%5Cin%20A%28s%29%7D%5Cpi%28a%20%7C%20s%29%20q_%7B%5Cpi%7D%28a%2C%20s%29)

