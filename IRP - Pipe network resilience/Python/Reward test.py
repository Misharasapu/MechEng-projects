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
        self.mean_demand_increase = 2
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
        Returns the current state of the environment.
        This method is designed to capture and return the state of the environment that the agent will observe.
        """
        # Run a simulation to ensure the latest state is reflected
        self.run_simulation()

        # Capture the pressures at all junctions as the state
        state = self.net.res_junction.p_bar.copy()

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
        # Define the reference pressure and the acceptable tolerance (10%)
        reference_pressure = 25  # bar
        tolerance = 0.1  # 10%

        lower_bound = reference_pressure * (1 - tolerance)
        upper_bound = reference_pressure * (1 + tolerance)

        # Initialize the reward
        reward = 0

        # Penalty or reward for each junction based on its pressure staying within the desired range
        for pressure in state_after:
            if lower_bound <= pressure <= upper_bound:
                # Add a small reward for each junction within the acceptable range
                reward += 1
            else:
                # Apply a penalty for each junction outside the acceptable range
                # The penalty is proportional to the deviation from the closest bound
                deviation = min(abs(pressure - lower_bound), abs(pressure - upper_bound))
                reward -= deviation * 10  # Adjust the penalty factor as needed

        return reward

    def test_reward_function(sim):
        # Step 1: Capture the initial state and pressures
        initial_state = sim.get_state()
        initial_pressures = sim.net.res_junction.p_bar.copy()

        # Print initial pressures for verification
        print("Initial Pressures:", initial_pressures)

        # Step 2: Perform an action (e.g., add a strategic source)
        sim.add_strategic_source()

        # Step 3: Capture the new state and pressures after action
        new_state = sim.get_state()
        new_pressures = sim.net.res_junction.p_bar.copy()

        # Print new pressures for verification
        print("New Pressures:", new_pressures)

        # Step 4: Calculate the reward
        reward = sim.calculate_reward(initial_state, new_state)

        # Print the reward for verification
        print("Reward for adding a strategic source:", reward)

    def check_if_done(self, state):
        """
        Check if the current state satisfies the goal, i.e., all junction pressures are within
        10% of the reference pressure.

        Parameters:
        - state (pd.Series): The current state, represented as pressures at all junctions.

        Returns:
        - done (bool): True if the episode is done (all pressures within the desired range), False otherwise.
        """
        reference_pressure = 25  # bar
        tolerance = 0.1  # 10%

        # Calculate the acceptable pressure range
        lower_bound = reference_pressure * (1 - tolerance)
        upper_bound = reference_pressure * (1 + tolerance)

        # Check if all pressures are within the acceptable range
        all_within_range = state.apply(lambda p: lower_bound <= p <= upper_bound).all()

        return all_within_range

    def step(self, action):
        """
        Executes a given action in the simulation environment and analyzes the impact.

        Parameters:
        - action (int): The action to execute. For simplicity, 1 indicates adding a strategic source.

        Returns:
        - state_after (DataFrame): The new state of the environment after executing the action.
        - pressure_diff (Series): The difference in pressure at each junction due to the action.
        """
        # Initialize pressure_diff to ensure scope availability
        pressure_diff = None

        if action == 1:
            # Get the state before the action for comparison
            state_before = self.get_state().copy()

            # Execute the action of adding a strategic source
            self.add_strategic_source()

            # Run the simulation to reflect changes
            self.run_simulation()

            # Get the new state after the action has been executed
            state_after = self.get_state()

            # Calculate the difference in pressures to analyze the impact
            pressure_diff = state_after - state_before
        else:
            print("No action recognized or other actions not implemented yet.")
            state_after = self.get_state()  # Fallback to current state if no valid action is provided

        return state_after, pressure_diff

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


    # Implement the get_state, calculate_reward, and check_if_done methods based on your needs


# Example usage
sim = GasPipeNetworkSimulation('C:\\Users\\misha\\OneDrive - University of Bristol\\Year 3\\IRP\\Python\\Copy of Travis150_Gas_Data_nodes.csv', 'C:\\Users\\misha\\OneDrive - University of Bristol\\Year 3\\IRP\\Python\\Copy of Travis150_Gas_Data_GasDemand.csv', 'C:\\Users\\misha\\OneDrive - University of Bristol\\Year 3\\IRP\\Python\\Copy of Travis150_Gas_Data_Source.csv', 'C:\\Users\\misha\\OneDrive - University of Bristol\\Year 3\\IRP\\Python\\Copy of Travis150_Gas_Data_Pipes.csv')

# Before adding a source
state_before = sim.get_state()
print("State before adding a strategic source:", state_before)

# Execute the action of adding a strategic source
state_after, pressure_diff = sim.step(1)

# Check if 'state_after' and 'pressure_diff' are not None before printing
if state_after is not None and pressure_diff is not None:
    print("State after adding a strategic source:", state_after)
    print("Pressure difference:", pressure_diff)

# Run the test
sim.test_reward_function()