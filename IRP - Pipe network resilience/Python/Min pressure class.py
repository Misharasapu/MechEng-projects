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
        self.std_dev_demand_increase = 0.3
        self.mean_reduction_factor = 0.8
        self.std_dev_reduction_factor = 0.1
        self.conversion_factor_m3_per_s = (28316.8466 / 730) / 3600  # Converts MMCF/mo to m³/s

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


    def add_strategic_source(self):
        # New source coordinates slightly adjusted from Load #3's location for demonstration
        new_source_lat = 30.2150  # Slightly adjusted latitude
        new_source_lon = -97.6050  # Slightly adjusted longitude
        new_junction_id = len(self.net.junction) + 1  # Unique ID for the new junction

        # Add the new junction for the source
        junction_idx = ppipe.create_junction(self.net, pn_bar=5, tfluid_k=15 + 273.15,
                                             name=f"Junction {new_junction_id}",
                                             geodata=(new_source_lat, new_source_lon))

        # Add a new source at this junction with specified mass flow rate
        source_idx = ppipe.create_source(self.net, junction=junction_idx, mdot_kg_per_s=1,
                                         name=f"Source at Junction {new_junction_id}")

        # Connecting this new source to "Load #3"
        load_3_id = self.junction_name_to_index['3']  # Get junction index for Load #3
        ppipe.create_pipe_from_parameters(self.net, from_junction=junction_idx, to_junction=load_3_id, length_km=0.1,
                                          diameter_m=0.1, k_mm=1, name=f"Pipe from Source {new_junction_id} to Load #3")

        print(f"New source added at Junction {new_junction_id} and connected to Load #3.")
        #self.run_simulation()  # Update the simulation to reflect changes

        return junction_idx  # Optionally return the new source's junction index for feedback

    def adjust_source_flow_rate(self, source_idx, adjustment):
        if source_idx in self.net.source.index:
            self.net.source.loc[source_idx, 'mdot_kg_per_s'] += adjustment
            #self.run_simulation()  # Reflect this adjustment in the simulation state
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
        - action (int): The action to execute. For simplicity, 1 indicates adding a strategic source.

        Returns:
        - state_after (int): The new state of the environment after executing the action.
        - reward (float): The reward obtained from taking the action.
        """
        # Initialize reward
        reward = 0

        # Get the state before the action for comparison
        state_before = self.get_state()

        if action == 1:
            # Execute the action of adding a strategic source
            self.add_strategic_source()

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


def run_episode(sim):
    done = False
    steps = 0
    max_steps = 300  # Define a maximum number of steps to avoid infinite loops

    sim.reset()

    while not done and steps < max_steps:
        # Use the action to add a strategic source
        action = 1
        state_after, _ = sim.step(action)  # Assuming step returns a tuple (state_after, reward)

        # Check if the episode should end based on the current state
        done = sim.check_if_done(state_after)

        steps += 1

        print(f"After {steps} steps, minimum pressure state is {state_after}")

        if done:
            print(f"Goal reached in {steps} steps.")
            break

    if not done:
        print("Maximum steps reached without fulfilling goal.")

    return steps




# Instantiate the simulation class
sim = GasPipeNetworkSimulation(
    nodes_file_path='C:\\Users\\misha\\OneDrive - University of Bristol\\Year 3\\IRP\\Python\\Copy of Travis150_Gas_Data_nodes.csv',
    gas_file_path='C:\\Users\\misha\\OneDrive - University of Bristol\\Year 3\\IRP\\Python\\Copy of Travis150_Gas_Data_GasDemand.csv',
    source_file_path='C:\\Users\\misha\\OneDrive - University of Bristol\\Year 3\\IRP\\Python\\Copy of Travis150_Gas_Data_Source.csv',
    pipes_file_path='C:\\Users\\misha\\OneDrive - University of Bristol\\Year 3\\IRP\\Python\\Copy of Travis150_Gas_Data_Pipes.csv'
)

# Run a single episode to test
steps_taken = run_episode(sim)
print(f"Steps taken to reach the goal: {steps_taken}")

# Now visualize the network
sim.visualize_network()



