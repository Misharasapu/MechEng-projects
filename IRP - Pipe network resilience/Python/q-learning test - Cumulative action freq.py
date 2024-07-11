import pandas as pd
import pandapipes as ppipe
import numpy as np
import random


class GasPipeNetworkSimulation:
    def __init__(self, nodes_file_path, gas_file_path, source_file_path, pipes_file_path):
        # Store file paths
        self.nodes_file_path = nodes_file_path
        self.gas_file_path = gas_file_path
        self.source_file_path = source_file_path
        self.pipes_file_path = pipes_file_path

        # Load data
        self.load_data()

        # Setup constants for demand and reduction factors (These could be arguments too)
        self.mean_demand_increase = 2.5
        self.std_dev_demand_increase = 0
        self.mean_reduction_factor = 0.8
        self.std_dev_reduction_factor = 0
        self.conversion_factor_m3_per_s = (28316.8466 / 730) / 3600  # Converts MMCF/mo to m³/s

        # Initialize the strategic source locations in a dictionary
        self.strategic_source_locations = {
            1: (30.08, -97.353),  # The coordinates for the first strategic source
            2: (30.2150, -97.6050),
            3: (29.95, -97.729)# The coordinates for the second strategic source
        }

        # Initial setup
        self.create_network()

    def load_data(self):
        self.nodes_df = pd.read_csv(self.nodes_file_path)
        self.gas_demand_df = pd.read_csv(self.gas_file_path)
        self.source_df = pd.read_csv(self.source_file_path)
        self.pipes_df = pd.read_csv(self.pipes_file_path)
        self.pipes_df.rename(columns={'Sarting Node': 'Starting Node', 'Ending Ndoe': 'Ending Node'}, inplace=True)

    def create_network(self):
        self.net = ppipe.create_empty_network()
        ppipe.create_fluid_from_lib(self.net, 'methane', overwrite=True)
        self.create_junctions()
        self.add_sources_and_sinks()
        self.add_pipes()

    def create_junctions(self):
        self.junction_name_to_index = {}  # Initialize the mapping dictionary
        for _, row in self.nodes_df.iterrows():
            junction_name = str(int(row['Node#']))
            # Create each junction
            ppipe.create_junction(self.net, pn_bar=row['P (kPa)'] / 100, tfluid_k=15, name=junction_name,
                                  geodata=(row['Lat'], row['Lon']))

        # After all junctions are created, populate the name-to-index mapping
        for idx, name in enumerate(self.net.junction.name.values):
            self.junction_name_to_index[name] = idx

    def add_sources_and_sinks(self):
        for _, row in self.gas_demand_df.iterrows():
            node_str = str(int(row['Node#']))
            junction_index = self.junction_name_to_index[node_str]  # Reference junction by name-to-index mapping

            demand_increase_factor = np.random.normal(self.mean_demand_increase, self.std_dev_demand_increase)
            demand_increase_factor = max(demand_increase_factor, 1.0)  # Ensure factor is at least 1.0

            consumption_m3_per_s = row[
                                       'Consumption (MMCF/mo)'] * self.conversion_factor_m3_per_s * demand_increase_factor

            # Print the demand increase factor for each sink
            print(
                f"Demand at Node {node_str}: {consumption_m3_per_s:.3f} m³/s, Increase Factor: {demand_increase_factor:.2f}")

            ppipe.create_sink(self.net, junction=junction_index, mdot_kg_per_s=consumption_m3_per_s,
                              name="Sink at Node " + node_str)

        source_row = self.source_df.iloc[0]
        junction_index = self.junction_name_to_index[
            str(int(source_row['Node']))]  # Again, use the mapping for the source
        ppipe.create_ext_grid(self.net, junction=junction_index, p_bar=source_row['Pressure (kPa)'] / 100,
                              name="External Grid at Node 24")

    def add_pipes(self):
        for _, row in self.pipes_df.iterrows():
            start_node, end_node = str(int(row['Starting Node'])), str(int(row['Ending Node']))
            # Use the mapping to get indices
            start_junction_index = self.junction_name_to_index[start_node]
            end_junction_index = self.junction_name_to_index[end_node]

            reduction_factor = np.random.normal(self.mean_reduction_factor, self.std_dev_reduction_factor)
            reduction_factor = min(reduction_factor, 1.0)  # Ensure the reduction factor is not above 1.0

            adjusted_diameter = row[
                                    'Diameter (mm)'] * reduction_factor / 1000  # Convert mm to meters and apply reduction factor

            # Print the adjusted diameter and reduction factor for each pipe
            print(
                f"Adjusted Diameter for Pipe from {start_node} to {end_node}: {adjusted_diameter:.3f} m, Factor: {reduction_factor:.2f}")

            ppipe.create_pipe_from_parameters(self.net, from_junction=start_junction_index,
                                              to_junction=end_junction_index,
                                              length_km=row['Length (km)'], diameter_m=adjusted_diameter, k_mm=row['k'],
                                              name=f"Pipe from {start_node} to {end_node}")

    def get_state(self):
        """
        Returns the current state of the environment as the minimum pressure across all junctions.
        """
        self.run_simulation()  # Ensure the latest state is reflected

        # Get continuous pressure values
        continuous_pressures = self.net.res_junction.p_bar.values

        # Find the minimum pressure value
        min_pressure = np.min(continuous_pressures)

        # Define the pressure range and bin size for discretization
        bin_size = 0.1  # in bar
        min_pressure_range = 0  # minimum expected pressure in the network
        max_pressure_range = 25  # maximum expected pressure in the network
        bins = np.arange(min_pressure_range, max_pressure_range + bin_size, bin_size)

        # Discretize the minimum pressure: np.digitize returns the bin index
        state = np.digitize(min_pressure, bins) - 1  # -1 to start bin index at 0

        print(f"Current state (minimum pressure bin): {state}")

        return state

    def add_strategic_source(self, location_id):
        # Get the new source's coordinates from the dictionary using the location_id
        print(f"Adding strategic source at location {location_id}")
        new_source_lat, new_source_lon = self.strategic_source_locations[location_id]
        new_junction_id = len(self.net.junction) + 1  # Unique ID for the new junction

        # Add the new junction for the source
        junction_idx = ppipe.create_junction(self.net, pn_bar=5, tfluid_k=15 + 273.15,
                                             name=f"Junction {new_junction_id}",
                                             geodata=(new_source_lat, new_source_lon))

        # Add a new source at this junction with specified mass flow rate
        source_idx = ppipe.create_source(self.net, junction=junction_idx, mdot_kg_per_s=2,
                                         name=f"Source at Junction {new_junction_id}")

        # Choose the load to connect to based on the location_id
        if location_id == 1:
            # Connect to Load #12 for Source 1
            load_id = '12'
        elif location_id == 2:
            # Connect to Load #3 for Source 2
            load_id = '3'
        elif location_id == 3:
            load_id = '10'
        else:
            raise ValueError(f"Location id {location_id} is not recognized.")

        # Get junction index for the selected load
        load_junction_index = self.junction_name_to_index[load_id]

        # Create a pipe connecting the new source to the selected load
        ppipe.create_pipe_from_parameters(self.net, from_junction=junction_idx, to_junction=load_junction_index,
                                          length_km=0.1, diameter_m=0.1, k_mm=1,
                                          name=f"Pipe from Source {new_junction_id} to Load {load_id}")

        print(f"Strategic Source {location_id} added at Junction {new_junction_id} and connected to Load {load_id}.")
        # self.run_simulation()  # Update the simulation to reflect changes

        return junction_idx

    # Optionally return the new source's junction index for feedback

    def adjust_source_flow_rate(self, source_idx, adjustment):
        if source_idx in self.net.source.index:
            self.net.source.loc[source_idx, 'mdot_kg_per_s'] += adjustment
            # self.run_simulation()  # Reflect this adjustment in the simulation state
            return True  # Indicate successful adjustment
        else:
            print(f"Source at index {source_idx} not found.")
            return False  # Indicate failure to adjust

    def calculate_reward(self, state_before, state_after):
        desired_pressure_bin = 250  # Assuming 25 bar is desired and bin size is 0.1 bar
        tolerance_bins = 5  # This corresponds to a tolerance of 0.5 bar
        max_possible_deviation = 250  # Assuming 250 bins cover all possible states

        reward = 0
        if desired_pressure_bin - tolerance_bins <= state_after <= desired_pressure_bin + tolerance_bins:
            reward += 10  # Maximum reward for being within the desired range
        else:
            deviation_after = abs(state_after - desired_pressure_bin)
            deviation_before = abs(state_before - desired_pressure_bin)

            # Normalize deviations based on the total possible range of states
            normalized_deviation_after = deviation_after / max_possible_deviation
            normalized_deviation_before = deviation_before / max_possible_deviation

            # Calculate improvement or regression
            improvement = normalized_deviation_before - normalized_deviation_after

            if improvement > 0:
                # Reward improvements more generously, especially closer to target
                reward += improvement * 300  # Scale this factor to adjust sensitivity
            else:
                # Apply a smaller, capped penalty for regressions
                penalty = -improvement * 5  # Less severe penalty factor
                reward -= min(penalty, 2)  # Cap the penalty to prevent large negative values

        return reward

    def check_if_done(self, state):
        # The reference pressure is 25 bar, and each bin represents 0.1 bar
        desired_pressure_bin = 250  # 25 bar
        tolerance_bins = 25  # This corresponds to a tolerance of 2.5 bar (10% of 25 bar)

        # Check if the minimum pressure is within the acceptable range
        return desired_pressure_bin - tolerance_bins <= state <= desired_pressure_bin + tolerance_bins

    def step(self, action):
        """
        Executes a given action in the simulation environment and analyzes the impact.

        Parameters:
        - action (int): The action to execute, indicating which strategic source to add.
                        1, 2, or 3 are valid actions corresponding to predefined source locations.

        Returns:
        - state_after (int): The new state of the environment after executing the action.
        - reward (float): The reward obtained from taking the action.
        """
        # Initialize reward
        reward = 0

        # Get the state before the action for comparison
        state_before = self.get_state()

        # Validate action and execute the adding of a strategic source at the specified location
        if action in [1, 2, 3]:  # Ensuring action is valid
            self.add_strategic_source(action)
        else:
            raise ValueError(f"Invalid action specified: {action}. Valid actions are 1, 2, or 3.")

        # Run the simulation to reflect changes
        self.run_simulation()

        # Get the new state after the action has been executed
        state_after = self.get_state()

        # Calculate the reward based on the change in the state
        reward = self.calculate_reward(state_before, state_after)

        return state_after, reward

    def visualize_network(self):
        import matplotlib.pyplot as plt
        import pandapipes.plotting as ppplot

        ppplot.simple_plot(self.net, plot_sinks=True, plot_sources=True, plot_pipe_labels=True)
        plt.show()

    def run_simulation(self):
        print("Starting simulation...")
        try:
            ppipe.pipeflow(self.net,
                           friction_model="nikuradse",  # Example: specifying friction model
                           mode="hydraulics",  # Running hydraulic simulation only
                           stop_condition="tol",  # Stopping based on tolerance
                           iter=1000,  # Maximum number of iterations
                           tol_p=0.01,  # Tolerance for pressure
                           tol_v=0.01)  # Tolerance for velocity
            print("Simulation ran successfully.")
            # Optionally, print the pressures at all junctions after the simulation
            print(self.net.res_junction.p_bar)
        except Exception as e:
            if "PipeflowNotConverged" in str(e):
                print("Simulation failed to converge due to nodes being out of service or not supplied.")
            else:
                print(f"An unexpected error occurred: {e}")

    def reset(self):
        # Reset the network to its initial state
        self.create_network()


import random
def run_episode(sim):
    done = False
    steps = 0
    max_steps = 300
    sim.reset()

    while not done and steps < max_steps:
        action = random.choice([1, 2, 3])  # Randomly choose among the available actions
        state_after, _ = sim.step(action)
        done = sim.check_if_done(state_after)
        steps += 1
        print(f"After {steps} steps, minimum pressure state is {state_after}")
        if done:
            print(f"Goal reached in {steps} steps.")
            break

    if not done:
        print("Maximum steps reached without fulfilling goal.")
    return steps













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
        self.action_frequency = np.zeros(num_actions + 1)  # Initialize the action frequency tracker

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(1, self.num_actions + 1)
        else:
            action = np.argmax(self.q_table[state][1:]) + 1
        self.action_frequency[action] += 1  # Update the action frequency tracker
        return action

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

    def reset_action_frequencies(self):
        self.action_frequency = np.zeros(self.num_actions + 1)  # Reset the action frequency tracker

# Note: You may want to call reset_action_frequencies() method before each new simulation run if you are looking for per-run statistics.




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






# Parameters for Q-learning
num_states = 251  # Adjusted to match your fixed-length state vector in get_state()
num_actions = 3  # Assuming there are three different actions for the three locations

q_agent = QLearningAgent(num_states, num_actions)



# Instantiate the simulation class
sim = GasPipeNetworkSimulation(
    nodes_file_path='C:\\Users\\misha\\OneDrive - University of Bristol\\Year 3\\IRP\\Python\\Copy of Travis150_Gas_Data_nodes.csv',
    gas_file_path='C:\\Users\\misha\\OneDrive - University of Bristol\\Year 3\\IRP\\Python\\Copy of Travis150_Gas_Data_GasDemand.csv',
    source_file_path='C:\\Users\\misha\\OneDrive - University of Bristol\\Year 3\\IRP\\Python\\Copy of Travis150_Gas_Data_Source.csv',
    pipes_file_path='C:\\Users\\misha\\OneDrive - University of Bristol\\Year 3\\IRP\\Python\\Copy of Travis150_Gas_Data_Pipes.csv'
)

import matplotlib.pyplot as plt


def run_multiple_episodes(sim, q_agent, num_episodes):
    total_rewards_per_episode = []  # List to track the total reward for each episode
    max_q_values_per_episode = []  # List to track the maximum Q-value for each episode

    for episode in range(num_episodes):
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

        # After the episode, append the total reward and the max Q-value in the table to the lists
        total_rewards_per_episode.append(total_reward)
        max_q_values_per_episode.append(np.max(q_agent.q_table))  # Track maximum Q-value from Q-table
        q_agent.decay_epsilon()  # Decay epsilon after each episode

        print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {q_agent.epsilon:.4f}")

    # Plot the total reward evolution and max Q-values over episodes
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(total_rewards_per_episode, label='Total Reward per Episode')
    plt.title("Total Reward per Episode Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(max_q_values_per_episode, label='Max Q-value per Episode')
    plt.title('Max Q-value Per Episode')
    plt.xlabel("Episode")
    plt.ylabel("Q-value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return total_rewards_per_episode


# Define the number of episodes you want to run
num_episodes = 5  # for example, run for 100 episodes

# Call the function with the simulation instance, the Q-learning agent, and the number of episodes
q_value_tracking = run_multiple_episodes(sim, q_agent, num_episodes)


def run_single_detailed_episode(sim, q_agent):
    sim.reset()
    state = sim.get_state()
    total_reward = 0
    steps = 0

    while not sim.check_if_done(state) and steps < 200:  # Limit steps to prevent infinite loop
        action = q_agent.choose_action(state)
        next_state, reward = sim.step(action)
        q_agent.update_q_table(state, action, reward, next_state)
        print(f"Step: {steps}, State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
        state = next_state
        total_reward += reward
        steps += 1

    print(f"Total reward: {total_reward}")
    return q_agent.q_table

# Run this to see detailed outputs for each step
#run_single_detailed_episode(sim, q_agent)

def plot_q_values(q_table):
    plt.figure(figsize=(10, 8))
    for action in range(1, q_table.shape[1]):  # Assuming action indices start from 1
        plt.plot(q_table[:, action], label=f'Action {action}')
    plt.title('Q-value Convergence Over Episodes')
    plt.xlabel('State')
    plt.ylabel('Q-value')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_q_values(q_agent.q_table)  # Where `q_agent.q_table` is your Q-table after learning

def plot_action_frequencies(q_agent):
    actions = range(1, len(q_agent.action_frequency))  # Assuming actions start from 1
    frequencies = q_agent.action_frequency[1:]  # Ignore the 0th index as it is unused

    plt.figure(figsize=(8, 5))
    plt.bar(actions, frequencies, tick_label=actions)
    plt.title('Action Frequencies Over All Episodes')
    plt.xlabel('Actions')
    plt.ylabel('Frequency')
    plt.show()

plot_action_frequencies(q_agent)

