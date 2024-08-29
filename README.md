# Robotic Path Tracking with Q-Learning and SARSA

This project focuses on robotic path tracking using reinforcement learning algorithms, specifically Q-Learning and SARSA. The goal is to enable an autonomous agent to navigate a grid-based environment, avoid obstacles, and reach a specified goal.

## Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
  - [Environment](#environment)
  - [Reinforcement Learning Algorithms](#reinforcement-learning-algorithms)
  - [Simulation and Visualization](#simulation-and-visualization)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In the field of robotics and artificial intelligence, pathfinding and navigation are critical challenges. This project, titled "Robotic Path Tracking with Q-Learning and SARSA," explores the use of reinforcement learning algorithms to train an agent to find optimal paths in a grid-based environment. The agent learns to avoid obstacles and reach its goal by leveraging Q-Learning and SARSA, two prominent reinforcement learning techniques.

## Methodology

### Environment

The project simulates a 2D grid-based environment where:
- **Grid Size**: The grid world is configurable in size, allowing users to set the number of rows and columns.
- **Obstacles**: Various obstacles are placed within the grid, blocking the agent's path. The agent must learn to navigate around them.
- **Goal**: A goal location is defined, marked by a flag, which the agent must reach.

### Reinforcement Learning Algorithms

#### Q-Learning
- **Q-Table**: Stores Q-values for state-action pairs, helping the agent learn the best actions to take in each state.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation, controlled by the epsilon parameter.
- **Learning Rate & Discount Factor**: Influence how Q-values are updated, affecting the agent's learning process.

#### SARSA (State-Action-Reward-State-Action)
- **On-Policy Learning**: Unlike Q-Learning, SARSA updates Q-values based on the policy the agent is currently following.
- **SARSA Update Rule**: Updates Q-values using the actual actions taken, which can lead to more cautious decision-making compared to Q-Learning.

### Simulation and Visualization

The project uses Python's Tkinter library to create a graphical user interface (GUI) that:
- **Real-Time Rendering**: Displays the grid world, obstacles, agent, and goal in real-time.
- **Visualization**: Allows users to observe the agent's actions and learning process as it navigates the environment.

## Results

The project compares the performance of Q-Learning and SARSA based on metrics like:
- **Average Episode Length**: The average number of steps taken to reach the goal.
- **Success Rate**: The percentage of episodes where the agent successfully reached the goal.
- **Exploration Rate**: How often the agent explores new paths versus exploiting known paths.
- **Shortest and Longest Routes**: The efficiency of the learned paths, with comparisons between Q-Learning and SARSA.

## Demo

https://github.com/user-attachments/assets/9875384e-fabc-4988-81f8-3efc18a852a5

## Requirements

- Python 3.x
- Tkinter (for GUI)
- Numpy, Pandas (for data manipulation)
- Matplotlib (for plotting)
- PIL (Pillow) (for image processing)

## Usage

To run the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/alinababer/robotic-path-tracking-with-q-learning-and-sarsa.git
2. Navigate to the project directory:
cd robotic-path-tracking-with-q-learning-and-sarsa
3. Run the Q-Learning algorithm:
python run_agent.py --algorithm q-learning
4. Run the SARSA algorithm:
python run_agent.py --algorithm sarsa
