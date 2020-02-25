## Exercise 4.7 (programming)

*Write a program for policy iteration and re-solve Jack’s car rental problem with the following changes. One of Jack’s employees at the first location rides a bus home each night and lives near the second location. She is happy to shuttle one car to the second location for free. Each additional car still costs $2, as do all cars moved in the other direction. In addition, Jack has limited parking space at each location. If more than 10 cars are kept overnight at a location (after any moving of cars), then an additional cost of $4 must be incurred to use a second parking lot (independent of how many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often occur in real problems and cannot easily be handled by optimization methods other than dynamic programming. To check your program, first replicate the results given for the original problem.*

![Policies](./figures/car_rental_policies_4.7.gif)

![Optimal Value Function](./figures/Optimal_Value_Function_4.9.png)

## Exercise 4.9 (programming)

*Implement value iteration for the gambler’s problem and solve it for ![equation](https://latex.codecogs.com/png.latex?p_%7Bh%7D) = 0.25 and ![equation](https://latex.codecogs.com/png.latex?p_%7Bh%7D) = 0.55. In programming, you may find it convenient to introduce two dummy states corresponding to termination with capital of 0 and 100, giving them values of 0 and 1 respectively. Show your results graphically, as in Figure 4.3. Are your results stable as ![equation](https://latex.codecogs.com/png.latex?%5Ctheta%20%5Crightarrow%200).*

![Gambler_0.25](./figures/Gambler_025_4.9.png)

![Gambler_0.55](./figures/Gambler_055_4.9.png)