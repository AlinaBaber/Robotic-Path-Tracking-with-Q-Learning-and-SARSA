import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import final_states

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, exploration_prob=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.exploration_prob = exploration_prob
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.exploration_prob:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, state, action, reward, next_state):
        self.check_state_exist(next_state)
        q_predict = self.q_table.loc[state, action]
        if next_state != 'goal' or next_state != 'obstacle':
            q_target = reward + self.reward_decay * self.q_table.loc[next_state, :].max()
        else:
            q_target = reward
        self.q_table.loc[state, action] += self.learning_rate * (q_target - q_predict)
        return self.q_table.loc[state, action]

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table.loc[state] = [0] * len(self.actions)

    def print_q_table(self):
        final_route_coordinates = final_states()

        for i in range(len(final_route_coordinates)):
            state = str(final_route_coordinates[i])
            for j in range(len(self.q_table.index)):
                if self.q_table.index[j] == state:
                    self.q_table_final.loc[state, :] = self.q_table.loc[state, :]

        print()
        print('Length of final Q-table =', len(self.q_table_final.index))
        print('Final Q-table with values from the final route:')
        print(self.q_table_final)

        print()
        print('Length of full Q-table =', len(self.q_table.index))
        print('Full Q-table:')
        print(self.q_table)

    def plot_results(self, steps, cost):
        # Create a new figure with multiple subplots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

        # Plot 1: Episode via steps
        ax1 = axes[0, 0]
        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via Steps')

        # Plot 2: Episode via cost
        ax2 = axes[0, 1]
        ax2.plot(np.arange(len(cost)), cost, 'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Episode via Cost')

        # Calculate and plot rolling average of steps
        rolling_window = 10
        rolling_average = pd.Series(steps).rolling(rolling_window).mean()
        ax3 = axes[1, 0]
        ax3.plot(np.arange(len(rolling_average)), rolling_average, 'g')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Rolling Average Steps')
        ax3.set_title('Rolling Average Steps (Window={})'.format(rolling_window))

        # Calculate and plot rolling average of cost
        rolling_average_cost = pd.Series(cost).rolling(rolling_window).mean()
        ax4 = axes[1, 1]
        ax4.plot(np.arange(len(rolling_average_cost)), rolling_average_cost, 'm')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Rolling Average Cost')
        ax4.set_title('Rolling Average Cost (Window={})'.format(rolling_window))

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()
