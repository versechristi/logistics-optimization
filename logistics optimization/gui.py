# gui/main_window.py
# -*- coding: utf-8 -*-
"""
Main window class for the Logistics Optimization GUI application,
adapted for Multi-Depot, Two-Echelon VRP with Drones and Split Deliveries.

This file orchestrates the user interface, parameter input, data generation,
algorithm selection, optimization execution, and results visualization.
It interacts with the core modules (`core.route_optimizer`, `core.problem_utils`,
`core.distance_calculator`, `core.cost_function`), algorithm implementations,
and visualization/utility modules.

The structure maintains the existing GUI layout, focusing on updating
the backend logic to handle multi-depot scenarios and process the
structured results returned by the refactored `route_optimizer`.
Optimization execution is performed in a separate thread to keep the GUI responsive.
"""

# --- Standard Library Imports ---
import tkinter as tk
import traceback
from tkinter import ttk, filedialog, messagebox, font as tkFont, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import random
import configparser
import os
import copy
import webbrowser
import sys
import math # For checking inf/nan
import threading # Essential for running optimization in a separate thread

# --- Path Setup ---
# Ensure the project root directory (containing 'core', 'algorithm', etc.) is in the Python path
# This makes imports more reliable, especially when running scripts directly.
try:
    # Assuming this script is in a 'gui' subdirectory of the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Appended project root to sys.path: {project_root}")
    _PATH_SETUP_SUCCESS = True
except Exception as e:
    print(f"CRITICAL ERROR during path setup: {e}")
    traceback.print_exc()
    _PATH_SETUP_SUCCESS = False # Indicate critical failure


# --- Project Module Imports ---
# Attempt to import core components and algorithm run functions.
# Implement robust try-except blocks to handle potential import failures gracefully,
# preventing the application from crashing if a module is missing or has errors.
# Define dummy functions/classes if imports fail, allowing the GUI to open
# and display an error message instead of crashing.

try:
    # Import the main optimization orchestrator
    from core.route_optimizer import run_optimization as core_run_optimization
    # Import core problem data structures and utilities used by the GUI (e.g., SolutionCandidate for type hinting if needed)
    from core.problem_utils import SolutionCandidate # For type hinting and understanding structure
    # Import data generation functions
    from data.data_generator import generate_locations, generate_demand
    # Import visualization components
    from visualization.map_generator import generate_folium_map, open_map_in_browser
    from visualization.plot_generator import PlotGenerator
    # Import report generator
    from utils.report_generator import generate_delivery_report # If report generation is triggered/displayed here

    _CORE_FUNCTIONS_AVAILABLE = True

except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import core/data/visualization modules: {e}")
    traceback.print_exc()
    _CORE_FUNCTIONS_AVAILABLE = False
    # Define dummy functions for critical components if imports fail
    def core_run_optimization(*args, **kwargs):
        messagebox.showerror("Optimization Error", "Core optimization modules failed to load.")
        return {'overall_status': 'Error', 'error_message': 'Core modules failed to load.', 'results_by_algorithm': {}}
    class SolutionCandidate: pass # Dummy class
    def generate_locations(*args, **kwargs):
        messagebox.showerror("Data Error", "Data generation module failed to load.")
        return None # Dummy return
    def generate_demand(*args, **kwargs): return [] # Dummy return
    def generate_folium_map(*args, **kwargs): warnings.warn("Map generator not available."); return None
    def open_map_in_browser(*args, **kwargs): warnings.warn("Map browser opener not available.")
    class PlotGenerator:
        def __init__(self): warnings.warn("PlotGenerator not available.")
        def generate_cost_history_plot(self, *args, **kwargs): warnings.warn("Plot method not available.")
        def generate_comparison_plots(self, *args, **kwargs): warnings.warn("Plot method not available.")
    def generate_delivery_report(*args, **kwargs): warnings.warn("Report generator not available."); return "Report generation module not available."


# Import GUI utility helpers if you have them (e.g., gui/utils.py)
# try:
#     from gui.utils import create_label, create_entry, create_button, create_checkbox, create_separator
#     _GUI_UTILS_AVAILABLE = True
# except ImportError:
#     warnings.warn("GUI utility functions not available. Using direct widget creation.")
#     _GUI_UTILS_AVAILABLE = False
#     # Define simple placeholder lambda functions or methods if you rely heavily on these utilities


# --- MainWindow Class Definition ---
class MainWindow(tk.Tk): # Inherit from tk.Tk
    """
    The main application window for the Logistics Optimization GUI.
    Manages the UI layout, user input, triggers optimization runs,
    and displays results and visualizations.
    """
    def __init__(self):
        """Initializes the main window and its components."""
        super().__init__() # Initialize the Tkinter window

        self.title("Logistics Optimization System (MD-2E-VRPSD)")
        self.geometry("1200x800") # Initial window size

        # --- Class Attributes ---
        self.config = configparser.ConfigParser() # Config parser for loading/saving settings
        self.config_file = 'config/default_config.ini' # Default config file path

        self.current_problem_data = None # Store the currently generated/loaded problem data (locations, demands)
        self.current_optimization_results = None # Store the results from the last optimization run

        # --- GUI Layout Setup ---
        # This structure follows the user's existing satisfactory layout
        self.create_widgets()

        # --- Load Configuration and Populate UI ---
        self.load_configuration()

        # --- Initial Map Display (Optional: show empty map or initial points if loaded from config) ---
        # self.display_initial_map() # Method to call to show map area initially

    # --- Widget Creation Methods ---
    # Break down widget creation into separate methods for clarity, following the original structure.
    # These methods should create frames, notebooks, labels, entries, buttons, etc.

    def create_widgets(self):
        """Creates all the main widgets for the window layout."""
        # Example: Create main frames (left panel for params, right panel for results)
        self.params_frame = ttk.Frame(self, padding="10")
        self.results_frame = ttk.Frame(self, padding="10")

        self.params_frame.grid(row=0, column=0, sticky="nsew")
        self.results_frame.grid(row=0, column=1, sticky="nsew")

        # Configure grid weights to make frames resizable
        self.grid_columnconfigure(0, weight=1) # Left panel takes 1 part of width
        self.grid_columnconfigure(1, weight=2) # Right panel takes 2 parts of width
        self.grid_rowconfigure(0, weight=1) # Main row takes full height

        # Create parameter input widgets within params_frame
        self.create_parameter_widgets(self.params_frame)

        # Create results display widgets within results_frame
        self.create_results_widgets(self.results_frame)


    def create_parameter_widgets(self, parent_frame):
        """Creates parameter input widgets in the left panel."""
        # Use a Notebook for different parameter categories (Data, Vehicle, Drone, Objective, Algorithms)
        self.param_notebook = ttk.Notebook(parent_frame)
        self.param_notebook.pack(fill="both", expand=True)

        # Data Generation Tab
        self.data_gen_frame = ttk.Frame(self.param_notebook, padding="10")
        self.param_notebook.add(self.data_gen_frame, text="Data Generation")
        self.create_data_generation_widgets(self.data_gen_frame) # Call method to fill this tab

        # Vehicle Params Tab
        self.vehicle_params_frame = ttk.Frame(self.param_notebook, padding="10")
        self.param_notebook.add(self.vehicle_params_frame, text="Vehicle Params")
        self.create_vehicle_params_widgets(self.vehicle_params_frame) # Call method to fill this tab

        # Drone Params Tab
        self.drone_params_frame = ttk.Frame(self.param_notebook, padding="10")
        self.param_notebook.add(self.drone_params_frame, text="Drone Params")
        self.create_drone_params_widgets(self.drone_params_frame) # Call method to fill this tab


        # Objective Function Weights/Penalty Tab
        self.objective_func_frame = ttk.Frame(self.param_notebook, padding="10")
        self.param_notebook.add(self.objective_func_frame, text="Objective Func")
        self.create_objective_func_widgets(self.objective_func_frame) # Call method to fill this tab


        # Algorithm Parameters Tab (Notebook for different algorithms)
        self.algo_params_notebook = ttk.Notebook(self.param_notebook) # Nested notebook
        self.param_notebook.add(self.algo_params_notebook, text="Algorithm Params")
        self.create_algorithm_params_widgets(self.algo_params_notebook) # Call method to fill this tab


        # Data Generation/Optimization Action Buttons Area (below the notebook)
        self.action_buttons_frame = ttk.Frame(parent_frame, padding="10")
        self.action_buttons_frame.pack(fill="x")
        self.create_action_buttons_widgets(self.action_buttons_frame) # Call method to fill this area


    def create_data_generation_widgets(self, parent_frame):
        """Widgets for the Data Generation tab."""
        # Add input for Number of Logistics Centers
        # Outline labels, entry/spinbox widgets, and their associated tk.Var
        # e.g., self.num_logistics_centers_var = tk.IntVar(value=1)
        #      ttk.Label(parent_frame, text="Num Logistics Centers:").grid(...)
        #      ttk.Entry(parent_frame, textvariable=self.num_logistics_centers_var).grid(...)

        # Add inputs for Number of Sales Outlets, Number of Customers, Center Latitude, Center Longitude, Radius
        # Outline their labels, entry widgets, and tk.Var variables (e.g., tk.IntVar, tk.DoubleVar)

        # Add checkbox for Solomon-like distribution if supported
        # Outline checkbox widget and tk.BooleanVar

        # Add button to trigger data generation
        # Outline button widget and link command to self._generate_data
        pass # Placeholder for implementation


    def create_vehicle_params_widgets(self, parent_frame):
        """Widgets for the Vehicle Parameters tab."""
        # Add inputs for Vehicle Payload, Cost per km, Speed (kmph)
        # Outline labels, entry widgets, and tk.Var variables
        pass # Placeholder for implementation

    def create_drone_params_widgets(self, parent_frame):
        """Widgets for the Drone Parameters tab."""
        # Add inputs for Drone Payload, Max Flight Distance (km), Cost per km, Speed (kmph)
        # Outline labels, entry widgets, and tk.Var variables
        pass # Placeholder for implementation

    def create_objective_func_widgets(self, parent_frame):
        """Widgets for Objective Function Weights and Penalty tab."""
        # Add inputs for Cost Weight, Time Weight, Unmet Demand Penalty
        # Outline labels, entry widgets, and tk.Var variables
        pass # Placeholder for implementation

    def create_algorithm_params_widgets(self, parent_notebook):
        """Widgets for the Algorithm Parameters notebook (nested tabs)."""
        # Create a tab for EACH implemented algorithm (GA, SA, PSO, etc.)
        # Use ALGORITHM_REGISTRY keys to dynamically create tabs if possible or hardcode based on expected algorithms.

        # Example for GA tab:
        self.ga_params_frame = ttk.Frame(parent_notebook, padding="10")
        parent_notebook.add(self.ga_params_frame, text="Genetic Algorithm")
        # Add inputs for GA specific parameters (e.g., population size, generations, mutation rate, crossover rate, elite count, tournament size)
        # Outline labels, entry widgets, and tk.Var variables for GA parameters
        # e.g., self.ga_pop_size_var = tk.IntVar(value=100)

        # Example for SA tab:
        self.sa_params_frame = ttk.Frame(parent_notebook, padding="10")
        parent_notebook.add(self.sa_params_frame, text="Simulated Annealing")
        # Add inputs for SA specific parameters (e.g., initial temperature, cooling rate, max iterations)
        # Outline labels, entry widgets, and tk.Var variables for SA parameters

        # Example for PSO tab:
        self.pso_params_frame = ttk.Frame(parent_notebook, padding="10")
        parent_notebook.add(self.pso_params_frame, text="Particle Swarm Optimization")
        # Add inputs for PSO specific parameters (e.g., num particles, max iterations, inertia, cognitive, social weights)
        # Outline labels, entry widgets, and tk.Var variables for PSO parameters

        # Example for Greedy tab:
        self.greedy_params_frame = ttk.Frame(parent_notebook, padding="10")
        parent_notebook.add(self.greedy_params_frame, text="Greedy Heuristic")
        # Greedy usually has no specific parameters to input, maybe just a label indicating this.

        pass # Placeholder for implementation

    def create_action_buttons_widgets(self, parent_frame):
        """Widgets for Data Generation and Optimization execution buttons."""
        # Add "Generate Data Points" button
        # Outline button widget and link command to self._generate_data

        # Add Algorithm Selection Checkboxes
        # Create a frame or group for algorithm checkboxes
        # Outline checkboxes for GA, SA, PSO, Greedy
        # Use tk.BooleanVar for each checkbox state
        # Link checkboxes to enable/disable corresponding algorithm parameter tabs if desired

        # Add "Run Optimization" button
        # Outline button widget and link command to self._run_optimization
        # This button should be disabled until data is generated

        # Add Status Label
        # Outline label to display current status (e.g., "Ready", "Generating Data...", "Running GA...", "Optimization Complete")
        pass # Placeholder for implementation


    def create_results_widgets(self, parent_frame):
        """Creates results display and visualization widgets in the right panel."""
        self.results_notebook = ttk.Notebook(parent_frame)
        self.results_notebook.pack(fill="both", expand=True)

        # Route Map (Plot) Tab
        self.map_plot_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(self.map_plot_frame, text="Route Map (Plot)")
        self.create_map_plot_widgets(self.map_plot_frame) # Call method to fill this tab

        # Iteration Curve Tab
        self.iter_curve_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(self.iter_curve_frame, text="Iteration Curve")
        self.create_iteration_curve_widgets(self.iter_curve_frame) # Call method to fill this tab

        # Results Comparison Tab
        self.results_comparison_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(self.results_comparison_frame, text="Results Comparison")
        self.create_results_comparison_widgets(self.results_comparison_frame) # Call method to fill this tab

        # Report Tab (Optional: if displaying reports within GUI)
        self.report_frame = ttk.Frame(self.results_notebook, padding="10")
        self.results_notebook.add(self.report_frame, text="Report")
        self.create_report_widgets(self.report_frame) # Call method to fill this tab


    def create_map_plot_widgets(self, parent_frame):
        """Widgets for the Route Map (Plot) tab."""
        # Add a Frame to hold the Matplotlib figure canvas OR a placeholder for a map viewer
        # Matplotlib approach:
        # self.map_figure, self.map_ax = plt.subplots(figsize=(8, 6)) # Create figure and axes
        # self.map_canvas = FigureCanvasTkAgg(self.map_figure, master=parent_frame) # Create canvas
        # self.map_canvas.get_tk_widget().pack(fill="both", expand=True)
        # self.map_toolbar = NavigationToolbar2Tk(self.map_canvas, parent_frame) # Optional toolbar

        # Need a way to select which algorithm's map to display if multiple were run
        # Outline a dropdown (ttk.Combobox) and a variable (tk.StringVar) to hold selection
        # Outline a method linked to the dropdown selection to update the map display

        # Placeholder logic for initial empty plot
        # self.map_ax.set_title("Generated Locations / Route Map")
        # self.map_ax.set_xlabel("Longitude")
        # self.map_ax.set_ylabel("Latitude")
        # self.map_ax.grid(True)
        # self.map_canvas.draw() # Draw initial empty plot

        # If using Folium maps (saved to HTML):
        # Need a button to open the map HTML file in a web browser
        # Need a label or textbox to display the path to the generated map file

        pass # Placeholder for implementation


    def create_iteration_curve_widgets(self, parent_frame):
        """Widgets for the Iteration Curve tab."""
        # Add a Frame to hold the Matplotlib figure canvas for cost history plots
        # self.history_figure, self.history_ax = plt.subplots(figsize=(8, 6)) # Create figure and axes
        # self.history_canvas = FigureCanvasTkAgg(self.history_figure, master=parent_frame) # Create canvas
        # self.history_canvas.get_tk_widget().pack(fill="both", expand=True)
        # self.history_toolbar = NavigationToolbar2Tk(self.history_canvas, parent_frame) # Optional toolbar

        # Placeholder logic for initial empty plot
        # self.history_ax.set_title("Optimization Cost History")
        # self.history_ax.set_xlabel("Iteration/Generation")
        # self.history_ax.set_ylabel("Weighted Cost")
        # self.history_ax.grid(True)
        # self.history_canvas.draw() # Draw initial empty plot
        pass # Placeholder for implementation

    def create_results_comparison_widgets(self, parent_frame):
        """Widgets for the Results Comparison tab."""
        # Use a Treeview widget or a set of Labels to display comparison table/list
        # Outline a ttk.Treeview or a layout of ttk.Labels
        # Define columns for Treeview (Algorithm, Weighted Cost, Raw Cost, Time, Unmet Demand, Feasible, Computation Time)
        pass # Placeholder for implementation

    def create_report_widgets(self, parent_frame):
        """Widgets for the Report tab (if displaying reports internally)."""
        # Use a ScrolledText widget to display the content of generated text reports
        # Outline a scrolledtext.ScrolledText widget
        # Outline a way to load and display the content of a report file (e.g., from a dropdown selection or button)
        pass # Placeholder for implementation


    # --- Configuration Methods ---

    def load_configuration(self):
        """Loads parameters from the configuration file and populates the GUI."""
        # Use self.config.read(self.config_file)
        # Check if sections and options exist before reading
        # Populate the tk.Var variables linked to input widgets with values from config
        # Handle potential errors during file reading or parsing
        print(f"Loading configuration from {self.config_file}...")
        if os.path.exists(self.config_file):
            try:
                self.config.read(self.config_file)
                # Outline reading values for data generation params, vehicle, drone, objective weights
                # Read number of logistics centers: e.g., self.num_logistics_centers_var.set(self.config.getint('DataGeneration', 'num_logistics_centers', fallback=1))
                # Outline reading algorithm-specific parameters (nested structure)
                # e.g., if 'AlgorithmParams' in self.config:
                #          if 'ga_params' in self.config['AlgorithmParams']:
                #              ga_params_str = self.config['AlgorithmParams']['ga_params']
                #              try:
                #                  ga_params_dict = json.loads(ga_params_str) # Assuming params are stored as JSON string
                #                  # Populate GA input fields using ga_params_dict
                #              except json.JSONDecodeError:
                #                  warnings.warn("Failed to parse GA parameters from config.")
                # Do similar for SA, PSO, etc.

                print("Configuration loaded successfully.")
            except Exception as e:
                warnings.warn(f"Error loading configuration file {self.config_file}: {e}")
                traceback.print_exc()
                messagebox.showwarning("Config Load Error", f"Failed to load configuration:\n{e}\nUsing default values.")
        else:
            warnings.warn(f"Configuration file not found: {self.config_file}. Using default values.")
            # No file to load, default values of tk.Var will be used

    def save_configuration(self):
        """Saves current GUI parameters to the configuration file."""
        # Create or update sections in self.config
        # Get values from all tk.Var variables linked to input widgets
        # Set values in the config object
        # Handle saving algorithm-specific parameters (potentially as JSON strings)
        # e.g., if 'AlgorithmParams' not in self.config: self.config['AlgorithmParams'] = {}
        #      ga_params_to_save = { 'population_size': self.ga_pop_size_var.get(), ... }
        #      self.config['AlgorithmParams']['ga_params'] = json.dumps(ga_params_to_save)
        # Ensure the config directory exists
        # Write the config object to self.config_file
        print(f"Saving configuration to {self.config_file}...")
        try:
            # Outline setting values for data generation, vehicle, drone, objective weights
            # Set number of logistics centers: e.g., self.config['DataGeneration']['num_logistics_centers'] = str(self.num_logistics_centers_var.get())
            # Outline collecting and setting algorithm-specific parameters as JSON strings

            output_dir = os.path.dirname(self.config_file)
            if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir, exist_ok=True) # Ensure config directory exists

            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)
            print("Configuration saved successfully.")
        except Exception as e:
            warnings.warn(f"Error saving configuration file {self.config_file}: {e}")
            traceback.print_exc()
            messagebox.showwarning("Config Save Error", f"Failed to save configuration:\n{e}")


    # --- Data Generation & Optimization Execution ---

    def _get_all_parameters(self):
        """
        Collects all parameter values from GUI widgets and returns them in
        structured dictionaries as expected by core.route_optimizer.run_optimization.
        Includes basic validation.
        """
        # Outline retrieving values from all tk.Var variables associated with input widgets
        # Create and populate problem_data_params dict (num_centers, num_outlets, etc.)
        # Create and populate vehicle_params dict
        # Create and populate drone_params dict
        # Create and populate objective_weights dict
        # Create and populate optimization_params dict (including unmet_demand_penalty and nested algorithm params)
        # Get selected algorithm keys based on checkbox states

        # Include validation checks for numeric inputs (e.g., using try-except for float/int conversion)
        # Return the collected parameters as a tuple or dictionary
        print("Collecting parameters from GUI...")
        # Example validation for a numeric input:
        # try:
        #     num_centers = self.num_logistics_centers_var.get()
        #     if not isinstance(num_centers, int) or num_centers < 0:
        #         raise ValueError("Number of logistics centers must be a non-negative integer.")
        # except ValueError as e:
        #      messagebox.showerror("Parameter Error", f"Invalid input for Logistics Centers: {e}")
        #      return None # Indicate validation failure

        # Outline collecting all parameters similarly

        # Outline getting selected algorithm keys from checkboxes

        # Return collected parameters if validation passes, else None
        pass # Placeholder for implementation


    def _generate_data(self):
        """Triggers data generation based on GUI parameters."""
        print("Generating problem data...")
        # Disable relevant GUI elements (e.g., Generate button, Run button)
        # Update status label

        # Get data generation parameters from GUI (part of _get_all_parameters or a subset)
        params = self._get_all_parameters() # Reuse or get specific data params
        if params is None:
             # Validation failed in _get_all_parameters, message already shown
             # Re-enable GUI elements and return
             self._enable_gui_elements()
             return

        # Extract specific data generation parameters from params
        try:
             data_params = params.get('problem_data_params', {})
             num_logistics_centers = data_params.get('num_logistics_centers', 1) # Default
             num_sales_outlets = data_params.get('num_sales_outlets', 10) # Default
             num_customers = data_params.get('num_customers', 100) # Default
             center_latitude = data_params.get('center_latitude', 0.0) # Default
             center_longitude = data_params.get('center_longitude', 0.0) # Default
             radius_km = data_params.get('radius_km', 10.0) # Default
             use_solomon_distribution = data_params.get('use_solomon_like_distribution', False) # Default
             # Add other data params if needed (e.g., demand range)

             # Call data generator functions
             generated_locations = generate_locations(
                 num_logistics_centers=num_logistics_centers,
                 num_sales_outlets=num_sales_outlets,
                 num_customers=num_customers,
                 center_latitude=center_latitude,
                 center_longitude=center_longitude,
                 radius_km=radius_km,
                 use_solomon_like_distribution=use_solomon_distribution
             )

             if generated_locations is None:
                 raise RuntimeError("Location generation failed.")

             generated_demands = generate_demand(
                 num_customers=num_customers,
                 # Add demand generation specific params here
             )

             if generated_demands is None:
                 raise RuntimeError("Demand generation failed.")


             # Store the generated data
             self.current_problem_data = {
                 'locations': generated_locations,
                 'demands': generated_demands
             }
             print("Problem data generated and stored.")

             # Update status label
             # Re-enable Run Optimization button
             # Trigger initial map display

             # Display generated locations on the map tab
             self._display_generated_data_map()

             messagebox.showinfo("Data Generation", "Data generation complete.")

        except Exception as e:
            print(f"Error during data generation: {e}")
            traceback.print_exc()
            messagebox.showerror("Data Generation Error", f"Failed to generate data:\n{e}")
        finally:
            # Ensure GUI elements are re-enabled even if an error occurs
            self._enable_gui_elements()
            # Update status label to "Ready" or error message

    def _display_generated_data_map(self):
         """Displays the generated locations on the map tab."""
         if self.current_problem_data and _CORE_FUNCTIONS_AVAILABLE:
             try:
                  print("Displaying generated data points on map...")
                  # Use map_generator to create a map of just the points
                  # This might require a slightly different call to generate_folium_map
                  # or the generator needs to handle input without routes.
                  # Let's assume generate_folium_map can plot points if solution_structure is None or contains only locations.
                  # It's better to generate a SolutionCandidate object first to pass to map_generator for consistency.
                  dummy_initial_solution = SolutionCandidate(
                       problem_data=self.current_problem_data,
                       vehicle_params={}, # Dummy params needed by Candidate init
                       drone_params={}, # Dummy params needed by Candidate init
                       unmet_demand_penalty=0.0 # Dummy needed by Candidate init
                       # Other initial_... args will be default empty {} or None
                  )
                  # Need to pass dummy initial_solution_candidate.all_locations to map_generator

                  # Temporarily pass required data structure for point plotting
                  temp_solution_structure_for_map = {
                       'all_locations': self.current_problem_data.get('locations', {}),
                       'stage1_routes': {}, # No routes yet
                       'stage2_trips': {}, # No trips yet
                       'served_customer_details': {}, # No details yet
                  }


                  # Need to save the map HTML to a temporary file to display
                  map_output_path = os.path.join("output", "temp_generated_data_map.html")
                  os.makedirs(os.path.dirname(map_output_path), exist_ok=True) # Ensure output dir exists

                  generated_map_path = generate_folium_map(
                      problem_data=self.current_problem_data, # Pass full problem data
                      solution_structure=temp_solution_structure_for_map, # Pass the structure for point plotting
                      vehicle_params={}, # No vehicle params needed for just points
                      drone_params={}, # No drone params needed for just points
                      output_path=map_output_path,
                      map_title="Generated Data Points"
                  )

                  if generated_map_path and os.path.exists(generated_map_path):
                      print(f"Generated data map saved to {generated_map_path}. Opening in browser...")
                      # Open the generated HTML map in the default web browser
                      open_map_in_browser(generated_map_path)
                      # Optionally update a label in the GUI with the map file path
                  else:
                       warnings.warn("Failed to generate or save generated data map.")


             except Exception as e:
                 warnings.warn(f"Error displaying generated data map: {e}")
                 traceback.print_exc()


    def _run_optimization(self):
        """Triggers the optimization process in a separate thread."""
        print("Initiating optimization run...")
        # Check if data is generated first
        if not self.current_problem_data:
            messagebox.showwarning("Run Optimization", "Please generate data points first.")
            return

        # Get all parameters from GUI
        all_params = self._get_all_parameters()
        if all_params is None:
             # Validation failed in _get_all_parameters, message already shown
             return

        # Disable relevant GUI elements (e.g., all input fields, Generate button, Run button)
        self._disable_gui_elements()
        # Update status label to "Running Optimization..."

        # Extract parameters for the optimization function call
        problem_data_for_optimizer = self.current_problem_data # Use the stored generated data
        vehicle_params = all_params.get('vehicle_params', {})
        drone_params = all_params.get('drone_params', {})
        optimization_params = all_params.get('optimization_params_gui', {}) # Contains unmet_penalty and algo_params
        selected_algorithm_keys = all_params.get('selected_algorithm_keys', [])
        objective_weights = all_params.get('objective_weights', {})

        # Check if any algorithms are selected
        if not selected_algorithm_keys:
             messagebox.showwarning("Run Optimization", "Please select at least one algorithm to run.")
             self._enable_gui_elements()
             # Update status label
             return

        # --- Start Optimization Thread ---
        # Pass the collected parameters to the thread function
        self.optimization_thread = threading.Thread(
            target=self.run_optimization_thread,
            args=(problem_data_for_optimizer, vehicle_params, drone_params,
                  optimization_params, selected_algorithm_keys, objective_weights)
        )
        self.optimization_thread.start()


    def run_optimization_thread(self, problem_data, vehicle_params, drone_params,
                                optimization_params, selected_algorithm_keys, objective_weights):
        """
        Executes the optimization (calling core.route_optimizer) in a separate thread.
        Designed to be run by a threading.Thread.
        """
        print("Optimization thread started.")
        optimization_results = None # Variable to store results from run_optimization
        try:
            # Call the main optimization orchestration function
            if _CORE_FUNCTIONS_AVAILABLE:
                 optimization_results = core_run_optimization(
                     problem_data=problem_data,
                     vehicle_params=vehicle_params,
                     drone_params=drone_params,
                     optimization_params=optimization_params, # This includes unmet_penalty and nested algo params
                     selected_algorithm_keys=selected_algorithm_keys,
                     objective_weights=objective_weights
                 )
            else:
                 # core_run_optimization dummy function was called and showed an error
                 # Return a dummy error result
                 optimization_results = {'overall_status': 'Error', 'error_message': 'Core optimization modules not available.', 'results_by_algorithm': {}}


        except Exception as e:
            print(f"An unexpected error occurred in the optimization thread: {e}")
            traceback.print_exc()
            optimization_results = {'overall_status': 'Error', 'error_message': f"Optimization failed unexpectedly: {e}", 'results_by_algorithm': {}}

        finally:
            # Schedule the results display and GUI re-enabling back on the main Tkinter thread
            # Use self.after(delay_ms, callback, *args)
            print("Optimization thread finished. Scheduling results display on main thread.")
            self.after(0, self._display_results, optimization_results)


    def _disable_gui_elements(self):
         """Disables relevant GUI elements during data generation or optimization."""
         # Outline disabling entry fields, spinboxes, buttons, checkboxes that shouldn't be changed during a run
         pass # Placeholder for implementation

    def _enable_gui_elements(self):
         """Enables relevant GUI elements after data generation or optimization is complete."""
         # Outline enabling the elements disabled in _disable_gui_elements
         pass # Placeholder for implementation

    # --- Results Display ---

    def _display_results(self, optimization_results):
        """
        Receives results from the optimization thread and updates the GUI display.
        This method runs on the main Tkinter thread.
        """
        print("Displaying optimization results...")
        self.current_optimization_results = optimization_results

        if optimization_results is None:
            messagebox.showerror("Optimization Failed", "Optimization returned no results.")
            # Update status label
            self._enable_gui_elements()
            return

        # Check overall status and display error message if any
        overall_status = optimization_results.get('overall_status', 'Unknown')
        error_message = optimization_results.get('error_message')
        print(f"Optimization Overall Status: {overall_status}")
        if error_message:
            print(f"Optimization Error Message: {error_message}")
            messagebox.showerror("Optimization Status", f"Optimization finished with status: {overall_status}\nError: {error_message}")
            # Update status label with error
        else:
             # Update status label to "Optimization Complete"
             print("Optimization completed successfully (no overall error message).")


        # Get results for each algorithm
        results_by_algorithm = optimization_results.get('results_by_algorithm', {})

        # Update Results Comparison tab
        self._update_results_comparison(results_by_algorithm)

        # Update Iteration Curve tab
        self._update_iteration_curve(results_by_algorithm)

        # Update Route Map (Plot) tab
        self._update_route_map(optimization_results) # Need overall results to find best solution's key

        # Update Report tab (if implemented)
        self._update_report_display(results_by_algorithm) # Need results to find report paths

        # Re-enable GUI elements
        self._enable_gui_elements()
        # Final status update


    def _update_results_comparison(self, results_by_algorithm):
        """Updates the Results Comparison tab with metrics from each algorithm."""
        # Outline clearing the existing Treeview or Labels
        # Outline iterating through results_by_algorithm dictionary
        # For each algorithm result (if valid and no run error):
        #    Extract relevant metrics (weighted_cost, evaluated_cost, evaluated_time, evaluated_unmet, is_feasible, total_computation_time)
        #    Insert a row into the Treeview or update Labels
        # Handle cases where an algorithm had a run_error (display error message instead of metrics)
        print("Updating results comparison display...")
        # Example Treeview insertion:
        # self.results_treeview.insert('', 'end', values=(
        #     algo_name,
        #     format_float(weighted_cost, 4),
        #     format_float(raw_cost, 2),
        #     format_float(time_makespan, 2),
        #     format_float(unmet_demand, 2),
        #     "Yes" if is_feasible else "No",
        #     format_float(comp_time, 2)
        # ))
        pass # Placeholder for implementation

    def _update_iteration_curve(self, results_by_algorithm):
        """Updates the Iteration Curve tab with cost history plots."""
        # Outline clearing the existing Matplotlib figure
        # Check if PlotGenerator is available (_CORE_FUNCTIONS_AVAILABLE)
        # Instantiate PlotGenerator: self.plot_generator = PlotGenerator()
        # Call plot_generator.generate_cost_history_plot(results_by_algorithm)
        # Draw the canvas: self.history_canvas.draw()
        # Handle cases where an algorithm doesn't have cost history (e.g., Greedy) - the plotter should handle this
        print("Updating iteration curve plot...")
        if _CORE_FUNCTIONS_AVAILABLE:
             try:
                 # Clear previous plot
                 self.history_ax.clear()
                 # Generate plot using PlotGenerator
                 # The generate_cost_history_plot method should handle multiple algorithms
                 # and plot their histories on the same axes.
                 self.plot_generator.generate_cost_history_plot(
                     self.history_ax, results_by_algorithm
                 )
                 # Redraw the canvas
                 self.history_canvas.draw()
             except Exception as e:
                  warnings.warn(f"Error generating iteration curve plot: {e}")
                  traceback.print_exc()
                  # Display error on plot area or via message box
                  self.history_ax.clear()
                  self.history_ax.text(0.5, 0.5, f"Error generating plot:\n{e}", horizontalalignment='center', verticalalignment='center', color='red')
                  self.history_canvas.draw()
        else:
             warnings.warn("Cannot update iteration curve plot: PlotGenerator not available.")
             # Display placeholder on plot area


        pass # Placeholder for implementation

    def _update_route_map(self, optimization_results):
        """Updates the Route Map tab with the best solution's map or allows selection."""
        # Outline finding the best solution's result data from optimization_results
        # (use best_algorithm_key or fully_served_best_key)
        # If a map_path exists for the best solution:
        #    Outline opening the HTML file in a web browser OR embedding if possible.
        # If multiple algorithms ran:
        #    Populate the map selection dropdown with algorithm names
        #    Link dropdown selection change to a method that generates/loads and displays the map for the selected algorithm.
        print("Updating route map display...")
        if _CORE_FUNCTIONS_AVAILABLE and optimization_results:
             try:
                  # Find the key of the best algorithm result
                  best_algo_key = optimization_results.get('fully_served_best_key') or optimization_results.get('best_algorithm_key')

                  if best_algo_key:
                       best_algo_result_summary = optimization_results.get('results_by_algorithm', {}).get(best_algo_key)
                       if best_algo_result_summary and best_algo_result_summary.get('map_path'):
                            map_file_path = best_algo_result_summary['map_path']
                            if os.path.exists(map_file_path):
                                print(f"Opening map for best solution ({best_algo_key}) from: {map_file_path}")
                                open_map_in_browser(map_file_path)
                                # Optionally update a label with the path
                            else:
                                 warnings.warn(f"Map file not found for best solution ({best_algo_key}): {map_file_path}")
                                 messagebox.showwarning("Map Display Error", f"Map file not found for the best solution:\n{map_file_path}")

                  # Populate the map selection dropdown (if you have one)
                  # self._populate_map_selection_dropdown(optimization_results.get('results_by_algorithm', {}))


             except Exception as e:
                 warnings.warn(f"Error updating route map display: {e}")
                 traceback.print_exc()
                 messagebox.showwarning("Map Display Error", f"Error displaying map:\n{e}")

        else:
            warnings.warn("Cannot update route map: Core functions or results not available.")


        pass # Placeholder for implementation

    def _update_report_display(self, results_by_algorithm):
        """Updates the Report tab with selected report content."""
        # Outline populating a report selection dropdown with algorithm names
        # Link dropdown selection change to a method that reads the report file
        # and displays its content in the ScrolledText widget.
        # Handle cases where an algorithm had a report_generation_error or no report_path
        print("Updating report display...")
        if _CORE_FUNCTIONS_AVAILABLE and _REPORTING_AVAILABLE:
            # Populate report selection dropdown (if you have one)
            # e.g., self._populate_report_selection_dropdown(results_by_algorithm)
            pass # Placeholder for implementation
        else:
             warnings.warn("Cannot update report display: Core or ReportGenerator not available.")


        pass # Placeholder for implementation


# --- Helper for formatting float values (Copied from problem_utils/report_generator) ---
# Define locally for robustness if core modules fail to import,
# or rely on the imported version if successful. Using a placeholder here.
def format_float(value, precision=4):
    """Safely formats a numerical value to a specified precision string."""
    # Placeholder: Implement or ensure the imported format_float is available
    # Example implementation:
    if isinstance(value, (int, float)):
        if math.isnan(value): return "NaN"
        if value == float('inf'): return "Inf"
        if value == float('-inf'): return "-Inf"
        try: return f"{value:.{precision}f}"
        except: return str(value)
    elif value is None: return "N/A"
    else: return str(value)


# --- Main Application Entry Point ---
# This should remain largely the same, initializing the Tk root and running the MainWindow.
# It includes handling basic theme setup and launching the main application instance.

def main():
    """Main entry function to set up and run the GUI application."""
    # Code to set up TTK theme (as in original main_window.py __main__ block)
    # Create a temporary root window, set style, apply theme, then destroy
    # This should be done before creating the main MainWindow instance

    # Example (from original __main__):
    # root = tk.Tk()
    # root.withdraw() # Hide the temporary root window
    # try:
    #     style = ttk.Style()
    #     # ... theme application logic ...
    # except Exception as e:
    #     print(f"Warning: Could not configure ttk theme: {e}")
    # finally:
    #     root.destroy() # Destroy the temporary root window


    print("Creating and running the application...")
    # Create the main window instance which inherits from Tk
    app = MainWindow()
    # Start the Tkinter event loop
    app.mainloop()
    print("Application finished.")


# Standard Python entry point check
if __name__ == "__main__":
    main()