import pandas as pd
import pandapipes as ppipe
import numpy as np
import random




class QLearningAgent:
    def __init__(self, num_states, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros(num_states)  # One-dimensional Q-table for single action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        # Always return the action to add a strategic source, with some randomness for exploration
        return 1 if np.random.rand() > self.epsilon else 0

    def update_q_table(self, state, action, reward, next_state):
        # Assuming a single action, so the action parameter might be ignored
        best_future_value = self.q_table[next_state]
        td_target = reward + self.gamma * best_future_value
        td_delta = td_target - self.q_table[state]
        self.q_table[state] += self.alpha * td_delta
        return self.q_table[state]  # Return the updated Q-value for the current state


def run_q_learning_episode(sim, q_agent):
    sim.reset()
    state = sim.get_state()
    done = False
    total_reward = 0

    while not done:
        action = q_agent.choose_action(state)
        next_state, reward = sim.step(action)
        q_agent.update_q_table(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        done = sim.check_if_done(state)

    return total_reward


# Adjusted parameters for Q-learning
num_states = 251  # Or the appropriate number based on your state discretization

# Initialize Q-learning agent
q_agent = QLearningAgent(num_states)

# Run a single Q-learning episode with the simulation
reward = run_q_learning_episode(sim, q_agent)
print(f"Total reward from the episode: {reward}")
