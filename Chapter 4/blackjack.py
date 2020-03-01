import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

np.random.seed(69420)

HIT = 1
STAND = 0

class BlackJack:
    def __init__(self):
        self.base = 12
        self.dealer_actions = np.zeros(22, dtype=np.int)
        for i in range(0, 18):
            self.dealer_actions[i] = HIT
        for i in range(18, 22):
            self.dealer_actions[i] = STAND

    def deal_card(self):
        card = min(np.random.randint(1, 14), 10)
        return card

    def card_value(self, card):
        if card == 1:
            return 11
        else:
            return card

    def play(self, player_policy, initial=None):
        player_sum = 0
        player_trajectory = list()
        initial_action = None

        if initial is None:
            player_usable_ace = False

            while player_sum < 12:
                new_card = self.deal_card()
                player_sum += self.card_value(new_card)

                if player_sum > 21:
                    # Can only happen if two aces.
                    assert player_sum == 22
                    player_sum -= 10

                player_usable_ace = (new_card == 1)

            dealer_card_1 = self.deal_card()
        else:
            player_usable_ace, player_sum, dealer_card_1, initial_action = \
                initial

        dealer_card_2 = self.deal_card()
        dealer_usable_ace = 1 in (dealer_card_1, dealer_card_2)
        dealer_sum = self.card_value(dealer_card_1) + self.card_value(dealer_card_2)

        if dealer_sum > 21:
            # Can only happen if two aces.
            assert dealer_sum == 22
            dealer_sum -= 10

        assert player_sum <= 21 and dealer_sum <= 21
        
        # Let the Games Begin!
        # Player Plays First

        while True:
            state = [player_usable_ace, player_sum, dealer_card_1]
            if initial_action is None:
                action = player_policy(state)
            else:
                action = initial_action
                initial_action = None

            player_trajectory.append([state, action])

            if action == STAND:
                break

            new_card = self.deal_card()
            player_sum += self.card_value(new_card)

            ace_count = int(player_usable_ace)
            if new_card == 1:
                ace_count += 1

            while player_sum > 21 and (ace_count > 0):
                player_sum -= 10
                ace_count -= 1

            player_usable_ace = (ace_count > 0)

            if player_sum > 21:
                return -1, player_trajectory

        # Dealer Plays
        while True:
            action = self.dealer_actions[dealer_sum]

            if action == STAND:
                break

            new_card = self.deal_card()
            dealer_sum += self.card_value(new_card)

            ace_count = int(dealer_usable_ace)
            if new_card == 1:
                ace_count += 1

            while dealer_sum > 21 and (ace_count > 0):
                dealer_sum -= 10
                ace_count -= 1

            dealer_usable_ace = (ace_count > 0)

            if dealer_sum > 21:
                return 1, player_trajectory

        # Both Player and Dealer chose to STAND and
        # neither has gone bust.

        assert player_sum <= 21 and dealer_sum <= 21
        
        if player_sum > dealer_sum:
            return 1, player_trajectory
        elif player_sum == dealer_sum:
            return 0, player_trajectory
        else:
            return -1, player_trajectory

"""
Player Policies
"""
NAIVE_PLAYER_ACTIONS = np.zeros(10)
for i in range(12, 20):
    NAIVE_PLAYER_ACTIONS[i-12] = HIT
NAIVE_PLAYER_ACTIONS[20-12] = STAND
NAIVE_PLAYER_ACTIONS[21-12] = STAND

def naive_player_policy(state):
    _, player_sum, _ = state
    player_sum -= 12

    return NAIVE_PLAYER_ACTIONS[player_sum]

# For off-policy monte-carlo
def behaviour_policy(state):
    return np.random.choice([STAND, HIT])


"""
Different Monte-Carlo Methods.
"""
def monte_carlo_on_policy_eval(blackjack, episodes):
    state_values = np.zeros((2, 10, 10))
    state_values_counts = np.ones((2, 10, 10), dtype=np.int)

    for eps in tqdm.tqdm(range(episodes)):
        reward, player_trajectory = blackjack.play(player_policy=naive_player_policy)

        for (usable_ace, player_sum, dealer_card), _ in player_trajectory:
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1
            state_values[usable_ace, player_sum, dealer_card] += reward
            state_values_counts[usable_ace, player_sum, dealer_card] += 1

    return state_values/state_values_counts

def monte_carlo_exploring_control(blackjack, episodes):
    state_action_values = np.zeros((2, 10, 10, 2))
    state_action_counts = np.ones((2, 10, 10, 2), dtype=np.int)

    def greedy_player_policy(state):
        usable_ace, player_sum, dealer_card = state
        usable_ace = int(usable_ace)
        player_sum -= 12
        dealer_card -= 1

        values = state_action_values[usable_ace, player_sum, dealer_card, :]/\
            state_action_counts[usable_ace, player_sum, dealer_card, :]

        max_value = np.max(values)
        actions = [action for action, value in enumerate(values) if value == max_value]

        return np.random.choice(actions)

    for eps in tqdm.tqdm(range(episodes)):
        i_usable_ace = np.random.choice([True, False])
        i_player_sum = np.random.choice(np.arange(12, 22))
        i_dealer_card = np.random.choice(np.arange(1, 11))
        i_action = np.random.choice([HIT, STAND])
        initial = [i_usable_ace, i_player_sum, i_dealer_card, i_action]

        reward, player_trajectory = blackjack.play(player_policy=greedy_player_policy, initial=initial)

        for (usable_ace, player_sum, dealer_card), action in player_trajectory:
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1

            state_action_values[usable_ace, player_sum, dealer_card, action] += reward
            state_action_counts[usable_ace, player_sum, dealer_card, action] += 1

    return state_action_values/state_action_counts

def monte_carlo_off_policy_eval(blackjack, test_state, episodes):
    test_state = test_state
    state_value = np.zeros(episodes, dtype=np.float)
    rhos = np.zeros(episodes, dtype=np.float)

    for eps in range(episodes):
        w = 1
        reward, player_trajectory = blackjack.play(player_policy=behaviour_policy, initial=test_state)            
        for state, action in player_trajectory:
            if action == naive_player_policy(state):
                w *= 1 / 0.5
            else:
                w = 0
                break

        state_value[eps] = w * reward
        rhos[eps] = w
    
    accu_state_values = np.add.accumulate(state_value)
    
    ordinary_importance = accu_state_values / np.arange(1, episodes+1)

    accu_rho = np.add.accumulate(rhos)

    with np.errstate(divide='ignore', invalid='ignore'):
        weighted_importance = np.where(accu_rho != 0, accu_state_values/accu_rho, 0)

    return ordinary_importance, weighted_importance

def monte_carlo_off_policy_control(blackjack, episodes):
    state_action_values = np.zeros((2, 10, 10, 2), dtype=np.float)
    rho_sums = np.zeros((2, 10, 10, 2), dtype=np.float)
    greedy_policy = np.argmax(state_action_values, axis=-1)

    for eps in tqdm.tqdm(range(episodes)):
        reward, player_trajectory = blackjack.play(player_policy=behaviour_policy)
        
        w = 1
        for (usable_ace, player_sum, dealer_card), action in reversed(player_trajectory):
            
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1
            
            rho_sums[usable_ace, player_sum, dealer_card, action] += w

            state_action_values[usable_ace, player_sum, dealer_card, action] += \
                (w / rho_sums[usable_ace, player_sum, dealer_card, action]) * \
                (reward - state_action_values[usable_ace, player_sum, dealer_card, action])

            greedy_policy = np.argmax(state_action_values, axis=-1)
            if action != greedy_policy[usable_ace, player_sum, dealer_card]:
                break

            if action == greedy_policy[usable_ace, player_sum, dealer_card]:
                w *= 1/0.5
            else:
                w *= 0.0
                break


    return state_action_values

"""
Functions for the corresponding figures in the book.
"""

def figure_5_1():
    table1 = BlackJack()
    t1_sv = monte_carlo_on_policy_eval(table1, int(1e3))
    table2 = BlackJack()
    t2_sv = monte_carlo_on_policy_eval(table2, int(5e5))

    plots = [t1_sv[1, :, :], t2_sv[1, :, :], t1_sv[0, :, :], t2_sv[0, :, :]]

    X = np.arange(0, 10)
    Y = np.arange(0, 10)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(40, 30))

    titles = ['Optimal Value with Usable Ace. 10,000 Episodes',
              'Optimal Value with Usable Ace. 500,000 Episodes',
              'Optimal Value with no Usable Ace. 10,000 Episodes',
              'Optimal Value with no Usable Ace. 500,000 Episodes']

    for i, (plot, title) in enumerate(zip(plots, titles)):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.plot_wireframe(Y, X, Z=np.transpose(plot), cstride=1, rstride=1)
        ax.set_title(title, fontsize=30)
        
        ax.set_xlabel("Dealer Showing", fontsize=30)
        ax.set_xticks(np.arange(0, 10))
        xticks = list(np.arange(1, 11))
        xticks[0] = 'A'
        ax.set_xticklabels(xticks)

        ax.set_ylabel("Player Sum", fontsize=30)
        ax.set_yticks(np.arange(0, 10))
        ax.set_yticklabels(np.arange(12, 22))

        ax.set_zlim(-1, 1)

    
    plt.savefig("bj_mc_on_eval.png")
    plt.close()

def figure_5_2():
    table = BlackJack()
    sa = monte_carlo_exploring_control(table, int(5e6))
    
    state_values = np.max(sa, axis=-1)
    optimal_policy = np.argmax(sa, axis=-1)

    plots = [optimal_policy[1, :, :], state_values[1, :, :],\
             optimal_policy[0, :, :], state_values[0, :, :]]

    titles = ['Optimal Policy with Usable Ace.',
              'Optimal Value with Usable Ace.',
              'Optimal Policy with no Usable Ace.',
              'Optimal Value with no Usable Ace.']
    X = np.arange(0, 10)
    Y = np.arange(0, 10)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(40, 30))
    
    
    for i, (plot, title) in enumerate(zip(plots, titles)):
        if not i%2:
            ax = fig.add_subplot(2, 2, i+1)
            #ax = ax.flatten()
            fig_ = sns.heatmap(np.flipud(plot), cmap="YlGnBu", ax=ax, xticklabels=range(1, 11),
                              yticklabels=list(reversed(range(12, 22))))

        else:
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            ax.plot_wireframe(Y, X, Z=np.transpose(plot), cstride=1, rstride=1)
            ax.set_yticks(np.arange(0, 10))
            ax.set_yticklabels(np.arange(12, 22))

        ax.set_title(title, fontsize=30)

        ax.set_xlabel("Dealer Showing", fontsize=30)
        ax.set_xticks(np.arange(0, 10))
        xticks = list(np.arange(1, 11))
        xticks[0] = 'A'
        ax.set_xticklabels(xticks)

        ax.set_ylabel("Player Sum", fontsize=30)
        #ax.set_yticks(np.arange(0, 10))
        #ax.set_yticklabels(np.arange(12, 22))
    
    plt.savefig('bj_mc_exploration_control.png')
    plt.close()


def figure_5_3():
    true_value = -0.27726
    episodes = int(1e4)
    runs = int(1e2)
    test_state = [True, 13, 2, None]
    
    error_ordinary = np.zeros(episodes, dtype=np.float)
    error_weighted = np.zeros(episodes, dtype=np.float)

    for r in tqdm.tqdm(range(1, runs+1)):
        table = BlackJack()
        ordinary, weighted = monte_carlo_off_policy_eval(table, test_state, episodes)
            
        #import pdb; pdb.set_trace();
        error_ordinary += (1 / r) * (np.power(ordinary - true_value, 2) - error_ordinary)
        error_weighted += (1 / r) * (np.power(weighted - true_value, 2) - error_weighted)

    plt.plot(error_ordinary, label="Ordinary Importance Sampling")
    plt.plot(error_weighted, label="Weighted Importance Sampling")
    plt.xlabel("Episodes (log scale)")
    plt.ylabel("Mean Square Error")
    plt.xscale('log')
    plt.legend()

    plt.savefig('bj_mc_off_eval.png')
    plt.close()

def mc_off_policy_plots():
    table = BlackJack()
    sa = monte_carlo_off_policy_control(table, int(5e6))
    
    state_values = np.max(sa, axis=-1)
    optimal_policy = np.argmax(sa, axis=-1)

    plots = [optimal_policy[1, :, :], state_values[1, :, :],\
             optimal_policy[0, :, :], state_values[0, :, :]]

    titles = ['Optimal Policy with Usable Ace.',
              'Optimal Value with Usable Ace.',
              'Optimal Policy with no Usable Ace.',
              'Optimal Value with no Usable Ace.']
    X = np.arange(0, 10)
    Y = np.arange(0, 10)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(40, 30))
    
    
    for i, (plot, title) in enumerate(zip(plots, titles)):
        if not i%2:
            ax = fig.add_subplot(2, 2, i+1)
            #ax = ax.flatten()
            fig_ = sns.heatmap(np.flipud(plot), cmap="YlGnBu", ax=ax, xticklabels=range(1, 11),
                              yticklabels=list(reversed(range(12, 22))))

        else:
            ax = fig.add_subplot(2, 2, i+1, projection='3d')
            ax.plot_wireframe(Y, X, Z=np.transpose(plot), cstride=1, rstride=1)
            ax.set_yticks(np.arange(0, 10))
            ax.set_yticklabels(np.arange(12, 22))

        ax.set_title(title, fontsize=30)

        ax.set_xlabel("Dealer Showing", fontsize=30)
        ax.set_xticks(np.arange(0, 10))
        xticks = list(np.arange(1, 11))
        xticks[0] = 'A'
        ax.set_xticklabels(xticks)

        ax.set_ylabel("Player Sum", fontsize=30)

    plt.savefig("bj_mc_off_policy_control.png")
    plt.close()


if __name__ == "__main__":
    
    print("Plotting Figure 5.1")
    print('*'*50, '\n')
    figure_5_1()
    
    print("Plotting Figure 5.2")
    print('*'*50, '\n') 
    figure_5_2()
    
    print("Plotting Figure 5.3")
    print('*'*50, '\n') 
    figure_5_3()
    
    print("Off Policy Monte Carlo Control")
    print('+'*50, '\n')
    mc_off_policy_plots()
