from env import Environment
from agent_brain import QLearningTable

def run_episodes():
    episode_steps = []  # List to store the number of steps for each episode
    episode_costs = []  # List to store the cost for each episode

    for episode in range(1000):
        # Reset the environment to the initial observation
        observation = env.reset()

        # Initialize episode-specific variables
        num_steps = 0
        episode_cost = 0

        while True:
            # Render the environment
            env.render()

            # Agent chooses an action based on the current observation
            action = agent.choose_action(str(observation))

            # Agent takes an action and receives the next observation and reward
            next_observation, reward, done = env.step(action)

            # Agent learns from this transition and calculates the cost
            episode_cost += agent.learn(str(observation), action, reward, str(next_observation))

            # Update the current observation
            observation = next_observation

            # Increment the step count
            num_steps += 1

            # Break the loop when the episode ends (agent reaches the goal or an obstacle)
            if done:
                episode_steps.append(num_steps)
                episode_costs.append(episode_cost)
                break

    # Show the final route
    env.final()

    # Show the Q-table with values for each action
    agent.print_q_table()

    # Plot the results
    agent.plot_results(episode_steps, episode_costs)

# Entry point
if __name__ == "__main__":
    # Create the environment
    env = Environment()

    # Create the Q-learning agent
    agent = QLearningTable(actions=list(range(env.n_actions)))

    # Start running episodes
    env.after(100, run_episodes)
    env.mainloop()
