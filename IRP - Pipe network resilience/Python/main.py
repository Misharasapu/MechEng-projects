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

        # Assign different fixed flow rates to each source to differentiate impact
        if location_id == 1:
            flow_rate = 1  # Flow rate for source 1
        elif location_id == 2:
            flow_rate = 3  # Flow rate for source 2
        elif location_id == 3:
            flow_rate = 5  # Flow rate for source 3
        else:
            raise ValueError(f"Location id {location_id} is not recognized.")

        # Add a new source at this junction with the specified mass flow rate
        source_idx = ppipe.create_source(self.net, junction=junction_idx, mdot_kg_per_s=flow_rate,
                                         name=f"Source at Junction {new_junction_id}")

        # Choose the load to connect to based on the location_id
        if location_id == 1:
            load_id = '12'  # Load for Source 1
        elif location_id == 2:
            load_id = '3'  # Load for Source 2
        elif location_id == 3:
            load_id = '10'  # Load for Source 3

        # Get junction index for the selected load
        load_junction_index = self.junction_name_to_index[load_id]

        # Create a pipe connecting the new source to the selected load
        ppipe.create_pipe_from_parameters(self.net, from_junction=junction_idx, to_junction=load_junction_index,
                                          length_km=0.1, diameter_m=0.1, k_mm=1,
                                          name=f"Pipe from Source {new_junction_id} to Load {load_id}")

        print(f"Strategic Source {location_id} added at Junction {new_junction_id} and connected to Load {load_id}.")
        # Optionally, update the simulation here to reflect changes

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
        # Define the desired pressure range in bins
        desired_pressure_bin = 250  # Assuming 25 bar is desired and bin size is 0.1 bar
        tolerance_bins = 5  # This corresponds to a tolerance of 0.5 bar

        reward = 0
        if desired_pressure_bin - tolerance_bins <= state_after <= desired_pressure_bin + tolerance_bins:
            reward += 5  # Increase reward for being within the desired range
        else:
            deviation_after = abs(state_after - desired_pressure_bin)
            deviation_before = abs(state_before - desired_pressure_bin)

            # Normalize deviations based on the range of pressures
            max_deviation = desired_pressure_bin - tolerance_bins
            normalized_deviation_after = deviation_after / max_deviation

            # Check if current state is closer to the desired range than before
            if deviation_after < deviation_before:
                reward += 2  # Reward improvements towards the desired range
            else:
                # Apply a scaled, capped penalty based on how far from the desired range
                reward -= min(normalized_deviation_after * 5, 10)  # Cap the penalty to prevent excessively large values

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

    def get_pressures(self):
        # Run the simulation
        self.run_simulation()

        # Retrieve pressures at all junctions
        pressures = self.net.res_junction.p_bar.values
        return pressures

    def plot_pressures(self, pressures_before, pressures_after):
        import matplotlib.pyplot as plt

        # Trim pressures_after to match the length of pressures_before
        pressures_after = pressures_after[:len(pressures_before)]

        # Create a list of junction indices based on the length of the pressures lists
        junction_indices = list(range(len(pressures_before)))

        # Create subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

        # Plot initial pressures
        axs[0].plot(junction_indices, pressures_before, 'o-', label='Before Q-learning')
        axs[0].set_title('Pressure Distribution Before Q-learning')
        axs[0].set_ylabel('Pressure (bar)')
        axs[0].legend()
        axs[0].axhline(y=22.5, color='r', linestyle='--', label='Threshold Pressure')

        # Plot pressures after Q-learning
        axs[1].plot(junction_indices, pressures_after, 'o-', label='After Q-learning')
        axs[1].set_title('Pressure Distribution After Q-learning')
        axs[1].set_xlabel('Junction')
        axs[1].set_ylabel('Pressure (bar)')
        axs[1].legend()
        axs[1].axhline(y=22.5, color='r', linestyle='--', label='Threshold Pressure')

        plt.tight_layout()
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
        self.epsilon_decay = 0.8  # Decaying epsilon value
        self.min_epsilon = 0.01  # Minimum value of epsilon
        self.rewards_per_episode = []  # Track rewards per episode
        self.q_max_history = []  # Track max Q-value history
        self.action_frequency = np.zeros(num_actions + 1)  # Initialize the action frequency tracker
        self.episode_actions = []  # List to store action frequencies per episode

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
        self.epsilon_decay = 0.5  # Adjust this value to make it decay faster
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update_rewards(self, reward):
        self.rewards_per_episode.append(reward)
        self.q_max_history.append(np.max(self.q_table))


    def reset_action_frequencies(self):
        self.action_frequency = np.zeros(self.num_actions + 1)  # Reset the action frequency tracker

    def end_of_episode(self):
        # Call this method at the end of each episode to store the action frequencies
        self.episode_actions.append(self.action_frequency.copy())
        self.reset_action_frequencies()

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
    action_frequencies_over_time = []
    q_tables_over_time = []

    for episode in range(num_episodes):
        sim.reset()
        state = sim.get_state()
        done = False
        total_reward = 0
        q_agent.reset_action_frequencies()

        while not done:
            action = q_agent.choose_action(state)
            next_state, reward = sim.step(action)
            q_agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            done = sim.check_if_done(state)

        total_rewards_per_episode.append(total_reward)
        q_tables_over_time.append(q_agent.q_table.copy())
        action_frequencies_over_time.append(q_agent.action_frequency.copy())
        q_agent.decay_epsilon()
        q_agent.end_of_episode()

        print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {q_agent.epsilon:.4f}")

    return total_rewards_per_episode, q_tables_over_time, action_frequencies_over_time

def run_random_action_episodes(sim, num_episodes):
    total_rewards_per_episode = []
    for _ in range(num_episodes):
        sim.reset()
        done = False
        total_reward = 0
        while not done:
            action = random.choice([1, 2, 3])  # Choose a random action
            _, reward = sim.step(action)
            total_reward += reward
            done = sim.check_if_done(sim.get_state())
        total_rewards_per_episode.append(total_reward)
    return total_rewards_per_episode


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


def plot_selected_state_q_values(q_tables_over_time, start_state, end_state, num_episodes):
    plt.figure(figsize=(12, 6))
    # Loop through the range of states and plot their Q-values over episodes
    for state in range(start_state, end_state + 1):
        state_q_values = [q_table[state, :].max() for q_table in q_tables_over_time]
        plt.plot(range(num_episodes), state_q_values, label=f'State {state}')

    plt.xlabel('Episode')
    plt.ylabel('Max Q-value')
    plt.title('Q-value Convergence for Selected States Over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_learning_curve(total_rewards_per_episode):
    rolling_avg_reward = pd.Series(total_rewards_per_episode).rolling(window=50).mean()  # Adjust window size as needed
    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards_per_episode, label='Total Reward per Episode', alpha=0.5)
    plt.plot(rolling_avg_reward, label='Rolling Average of Total Rewards', linewidth=2, color='red')
    plt.title("Learning Curve Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_action_selection_frequency_over_time(action_frequencies_over_time, window=10):
    import pandas as pd
    import matplotlib.pyplot as plt

    # Convert the list of numpy arrays into a DataFrame for easier manipulation
    # Adjust the DataFrame creation to skip the first column if it's not used
    action_df = pd.DataFrame(action_frequencies_over_time, columns=['Action 0', 'Action 1', 'Action 2', 'Action 3'])

    # Drop the unused 'Action 0' column if your actions are labeled from 1 to num_actions
    action_df = action_df.drop('Action 0', axis=1)

    # Calculate cumulative sum to show cumulative frequencies
    cumulative_action_frequencies = action_df.cumsum()

    # Apply a moving average to smooth the data
    smoothed_actions = cumulative_action_frequencies.rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(smoothed_actions['Action 1'], label='Action 1')
    plt.plot(smoothed_actions['Action 2'], label='Action 2')
    plt.plot(smoothed_actions['Action 3'], label='Action 3')
    plt.title('Cumulative Action Selection Frequency Over Time (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Frequency of Action Selection')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rewards_comparison(q_rewards, random_rewards, num_episodes):
    episodes = list(range(num_episodes))
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, q_rewards, label='Q-Learning Rewards')
    plt.plot(episodes, random_rewards, label='Random Action Rewards', linestyle='--')
    plt.title('Comparison of Total Rewards Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.show()


# After defining your QLearningAgent and GasPipeNetworkSimulation classes and the run_multiple_episodes function...
# Before Q-learning
pressures_before = sim.get_pressures()
# Run the simulation for a specified number of episodes
num_episodes = 50  # or however many episodes you wish to run
# Adjusting the function calls
q_learning_results = run_multiple_episodes(sim, q_agent, num_episodes)
q_learning_rewards = q_learning_results[0]  # Only taking the first element which is total_rewards_per_episode
random_action_rewards = run_random_action_episodes(sim, num_episodes)

# Plotting the comparison
plot_rewards_comparison(q_learning_rewards, random_action_rewards, num_episodes)




