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



