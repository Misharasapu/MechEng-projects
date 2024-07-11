import pandas as pd
import pandapipes as ppipe
import numpy as np
import random


class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.9):
        self.q_table = np.zeros((num_states, num_actions + 1))
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.99  # Decaying epsilon value
        self.min_epsilon = 0.01  # Minimum value of epsilon
        self.rewards_per_episode = []  # Track rewards per episode
        self.q_max_history = []  # Track max Q-value history

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(1, self.num_actions + 1)
        else:
            return np.argmax(self.q_table[state][1:]) + 1

    def update_q_table(self, state, action, reward, next_state):
        best_future_value = np.max(self.q_table[next_state][1:])
        td_target = reward + self.gamma * best_future_value
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_delta

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update_rewards(self, reward):
        self.rewards_per_episode.append(reward)
        self.q_max_history.append(np.max(self.q_table))



def run_q_learning_episode(sim, q_agent):
    sim.reset()  # Reset the simulation to the initial state
    state = sim.get_state()  # Get the initial state of the simulation
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action = q_agent.choose_action(state)  # Decide on an action based on the current state
        next_state, reward = sim.step(action)  # Execute the action in the simulation

        # Update the Q-table with the result of the action and get the updated Q-value for the current state-action pair
        updated_q_value = q_agent.update_q_table(state, action, reward, next_state)

        # Logging for visibility
        print(f"Step {step_count}:")
        print(f"  Current State: {state}")
        print(f"  Action Taken: {action} (Add Strategic Source at location {action})")  # Corrected the action description
        print(f"  Reward: {reward}")
        print(f"  Next State: {next_state}")
        print(f"  Updated Q-Value for State {state}, Action {action}: {updated_q_value}")
        print("-" * 50)

        state = next_state  # Move to the next state
        total_reward += reward  # Accumulate the total reward
        done = sim.check_if_done(state)  # Check if the simulation should end
        step_count += 1  # Increment step count

    return total_reward

num_states = 251  # Depending on your discretization
num_actions = 3  # Since you have three strategic source locations

q_agent = QLearningAgent(num_states, num_actions)
