# visualization/plot_generator.py
# -*- coding: utf-8 -*-
"""
Generates Matplotlib plots for visualizing optimization algorithm results,
designed for embedding within a Tkinter GUI canvas or saving standalone.

Provides methods to plot:
- Cost convergence curves over iterations/generations.
- Bar charts comparing final weighted cost and computation time across algorithms.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
from typing import Dict, Any, List, Tuple, Optional

# Configure logging for this module
import logging
logger = logging.getLogger(__name__)

# Define consistent styling for algorithms
# Ensure these keys match the algorithm keys used in route_optimizer.py
DEFAULT_ALGO_STYLES = {
    'genetic_algorithm': {'color': 'blue', 'marker': 'o', 'name': 'Genetic Algorithm'},
    'greedy_heuristic': {'color': 'grey', 'marker': 's', 'name': 'Greedy Heuristic'},
    'simulated_annealing': {'color': 'orange', 'marker': '^', 'name': 'Simulated Annealing'},
    'pso_optimizer': {'color': 'purple', 'marker': 'p', 'name': 'PSO'},
    # Add styles for other algorithms here
    'default': {'color': 'red', 'marker': 'x', 'name': 'Unknown Algorithm'}
}

# Define a small tolerance for floating-point comparisons
FLOAT_TOLERANCE_PLOT = 1e-9


class PlotGenerator:
    """
    Handles the generation and customization of Matplotlib plots for visualizing
    logistics optimization algorithm performance and results.
    """

    def __init__(self, algorithm_styles: Optional[Dict[str, Dict]] = None):
        """
        Initializes the PlotGenerator.

        Args:
            algorithm_styles: An optional dictionary defining custom styles
                              (color, marker, name) for algorithm keys.
                              If None, uses DEFAULT_ALGO_STYLES.
        """
        self.algo_styles = algorithm_styles if algorithm_styles else DEFAULT_ALGO_STYLES
        logger.debug("PlotGenerator initialized.")

    def _get_algo_style(self, algo_key: str) -> Tuple[str, str, str]:
        """Safely retrieves color, marker, and display name for a given algorithm key."""
        style = self.algo_styles.get(algo_key, self.algo_styles['default'])
        color = style.get('color', self.algo_styles['default']['color'])
        marker = style.get('marker', self.algo_styles['default']['marker'])
        name = style.get('name', self.algo_styles['default']['name'])
        return color, marker, name

    def _plot_no_data_message(self, ax: plt.Axes, message: str):
        """Helper function to display a message on an Axes object when no data is available."""
        if ax:
            ax.clear()
            ax.text(0.5, 0.5, message,
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='grey', fontsize='large')
            try:
                 # Redraw the canvas if possible (GUI context)
                 ax.figure.canvas.draw_idle()
            except AttributeError:
                 # Ignore if not in a canvas environment (e.g., saving directly)
                 pass

    def plot_iteration_curves(self, results_by_algorithm: Dict[str, Dict], ax: plt.Axes):
        """
        Plots the cost convergence history for multiple algorithms on a given Axes object.

        Args:
            results_by_algorithm: The results dictionary from route_optimizer, where keys
                                  are algorithm identifiers and values are dictionaries
                                  containing algorithm run details, including a 'result_data'
                                  sub-dictionary which should have a 'cost_history' list.
                                  Example: {'ga': {'result_data': {'cost_history': [...]}, ...}, ...}
            ax: The matplotlib.axes.Axes object to plot on.
        """
        if not ax:
            logger.error("Plotting Error: No Matplotlib Axes provided for iteration curves.")
            return

        logger.info("Generating iteration curve plot...")
        ax.clear()
        ax.set_title('Algorithm Cost Convergence')
        ax.set_xlabel('Iteration / Generation (Scaled %)')
        ax.set_ylabel('Best Cost Found') # Initial label, may change to log scale
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        max_len = 0
        plot_data_available = False
        plotted_lines = [] # Keep track of lines plotted for the legend

        # --- First Pass: Check data and find max length for scaling ---
        for algo_key, result_summary in results_by_algorithm.items():
            result_data = result_summary.get('result_data')
            if not result_data or result_summary.get('run_error'):
                 logger.debug(f"Skipping iteration plot for '{algo_key}' (no result data or run error).")
                 continue

            cost_history = result_data.get('cost_history')
            if not cost_history or not isinstance(cost_history, list):
                 logger.debug(f"Skipping iteration plot for '{algo_key}' (missing or invalid 'cost_history').")
                 continue

            # Filter out potential leading invalid values (inf, nan)
            valid_history = [c for c in cost_history if c is not None and math.isfinite(c)]
            if not valid_history:
                 logger.debug(f"Skipping iteration plot for '{algo_key}' (cost history contains only invalid values).")
                 continue

            plot_data_available = True
            # Use length of original history (minus initial invalid ones if any) for scaling
            first_valid_idx = next((i for i, c in enumerate(cost_history) if c is not None and math.isfinite(c)), len(cost_history))
            current_len = len(cost_history) - first_valid_idx
            max_len = max(max_len, current_len)


        if not plot_data_available:
            logger.warning("No valid iteration data found for any algorithm.")
            self._plot_no_data_message(ax, "No valid iteration data available.")
            return

        # Ensure max_len is at least 1 if there's data (handles single-point histories)
        max_len = max(1, max_len)

        # --- Second Pass: Plot the data ---
        can_use_log_scale = True # Assume log scale possible initially
        min_positive_cost = float('inf')

        for algo_key, result_summary in results_by_algorithm.items():
            result_data = result_summary.get('result_data')
            # Skip if no valid data (already checked, but double-check)
            if not result_data or result_summary.get('run_error') or not result_data.get('cost_history'):
                 continue

            cost_history = result_data.get('cost_history', [])
            # Find first valid starting point again
            first_valid_idx = next((i for i, c in enumerate(cost_history) if c is not None and math.isfinite(c)), -1)
            if first_valid_idx == -1: continue # Skip if still no valid data

            valid_history = cost_history[first_valid_idx:]
            if not valid_history: continue

            color, marker, algo_name = self._get_algo_style(algo_key)

            # Scale x-axis based on max_len for better comparison
            num_points = len(valid_history)
            if num_points > 1:
                # Scale x-axis from 0 to 100% using the original length relative to max_len
                original_len = len(cost_history) - first_valid_idx
                # Linspace from 0 to 100 * (original_len / max_len) for correct relative scaling
                x_values = np.linspace(0, 100 * (original_len / max_len), num_points)
                label = f"{algo_name}"
            elif num_points == 1:
                x_values = [0] # Plot single point at the beginning (0%)
                label = f"{algo_name} (Final Value)"
                # Plot single points with marker only, no line
                marker = marker
                linestyle = 'None' # Plot marker only for single point
            else: # Should not happen
                 continue

            # Check if log scale is viable for this algorithm's data
            if any(c <= FLOAT_TOLERANCE_PLOT for c in valid_history):
                can_use_log_scale = False
                logger.warning(f"Algorithm '{algo_key}' has non-positive costs. Disabling log scale for y-axis.")
            else:
                min_positive_cost = min(min_positive_cost, min(valid_history))


            # Plot the line/markers
            line, = ax.plot(x_values, valid_history, marker=marker, linestyle='-' if num_points > 1 else 'None',
                            color=color, label=label, markersize=4, alpha=0.8)
            plotted_lines.append(line)

        # --- Final Plot Adjustments ---
        if can_use_log_scale and min_positive_cost != float('inf'):
            try:
                ax.set_yscale('log')
                ax.set_ylabel('Best Cost Found (Log Scale)')
                # Optional: Add minor grid lines for log scale
                ax.grid(True, which='minor', linestyle=':', linewidth=0.5)
            except ValueError as e:
                 logger.warning(f"Could not set log scale despite initial check (e.g., identical values near zero). Using linear scale. Error: {e}")
                 ax.set_yscale('linear')
                 ax.set_ylabel('Best Cost Found (Linear Scale)')
        else:
             ax.set_yscale('linear')
             ax.set_ylabel('Best Cost Found (Linear Scale)')
             # Ensure y starts at or slightly below the minimum value if linear
             min_val = min((min(d.get_ydata()) for d in plotted_lines if d.get_ydata()), default=0)
             max_val = max((max(d.get_ydata()) for d in plotted_lines if d.get_ydata()), default=1)
             padding = (max_val - min_val) * 0.05
             ax.set_ylim(bottom=max(0, min_val - padding), top=max_val + padding) # Avoid negative ylim if min_val is near 0


        # Add legend if lines were plotted
        if plotted_lines:
            ax.legend(handles=plotted_lines, fontsize='small', loc='best')
        else:
            # This case should be caught earlier, but safeguard
            self._plot_no_data_message(ax, "No data lines to plot.")

        # Redraw canvas
        try:
            ax.figure.tight_layout(pad=1.5)
            ax.figure.canvas.draw_idle()
            logger.info("Iteration curve plot generated successfully.")
        except AttributeError:
            logger.debug("Iteration plot generated (no canvas to draw).")
        except Exception as e:
             logger.error(f"Error during final iteration plot adjustments or drawing: {e}", exc_info=True)


    def plot_comparison_bars(self, results_by_algorithm: Dict[str, Dict], ax_cost: plt.Axes, ax_time: plt.Axes):
        """
        Plots bar charts comparing final weighted cost and computation time
        for algorithms on two separate Axes objects.

        Args:
            results_by_algorithm: The results dictionary from route_optimizer.
                                  Expected structure: {'algo_key': {'result_data': {...}, 'computation_time': float}, ...}
            ax_cost: The matplotlib.axes.Axes object for the cost comparison plot.
            ax_time: The matplotlib.axes.Axes object for the time comparison plot.
        """
        if not ax_cost or not ax_time:
            logger.error("Plotting Error: Axes for comparison plots not provided.")
            return

        logger.info("Generating comparison bar charts...")
        ax_cost.clear()
        ax_time.clear()

        plot_data = []
        algo_keys_with_data = []

        # --- Extract and Validate Data for Plotting ---
        for algo_key, result_summary in results_by_algorithm.items():
            result_data = result_summary.get('result_data')
            run_error = result_summary.get('run_error')
            comp_time = result_summary.get('computation_time') # Get computation time from summary

            if run_error:
                 logger.debug(f"Skipping comparison plot for '{algo_key}' due to run error: {run_error}")
                 # Optionally, represent failed runs differently later
                 continue
            if not result_data:
                 logger.debug(f"Skipping comparison plot for '{algo_key}' (no result data).")
                 continue

            # Extract metrics safely
            w_cost = result_data.get('weighted_cost')
            is_feasible = result_data.get('is_feasible') # Check feasibility
            comp_time = comp_time if comp_time is not None else result_data.get('total_computation_time') # Fallback if needed

            # Validate extracted metrics
            valid_cost = w_cost is not None and math.isfinite(w_cost)
            valid_time = comp_time is not None and math.isfinite(comp_time)

            if not valid_cost:
                logger.warning(f"Invalid or missing weighted_cost for '{algo_key}'. Plotting as INF.")
                w_cost = float('inf') # Use inf for sorting/comparison, plot visually marked
            if not valid_time:
                logger.warning(f"Invalid or missing computation_time for '{algo_key}'. Plotting as 0.")
                comp_time = 0 # Plot time as 0 if invalid/missing

            plot_data.append({
                'key': algo_key,
                'cost': w_cost,
                'time': comp_time,
                'feasible': is_feasible if isinstance(is_feasible, bool) else False # Default to False if missing/invalid
            })
            algo_keys_with_data.append(algo_key)


        if not plot_data:
            logger.warning("No valid result data found for comparison plots.")
            self._plot_no_data_message(ax_cost, "No results to compare (Cost).")
            self._plot_no_data_message(ax_time, "No results to compare (Time).")
            return

        # --- Prepare Data for Bar Charts ---
        # Sort algorithms for consistent plotting order (e.g., by name)
        plot_data.sort(key=lambda x: self._get_algo_style(x['key'])[2]) # Sort by display name
        sorted_keys = [item['key'] for item in plot_data]
        algo_names = [self._get_algo_style(key)[2] for key in sorted_keys]
        costs = [item['cost'] for item in plot_data]
        times = [item['time'] for item in plot_data]
        feasibility_flags = [item['feasible'] for item in plot_data]
        colors = [self._get_algo_style(key)[0] for key in sorted_keys]

        x_pos = np.arange(len(algo_names))
        bar_width = 0.6

        # Determine max values for axis scaling, ignoring infinities for cost
        max_cost_plot = max((c for c in costs if math.isfinite(c)), default=0)
        max_time_plot = max(times, default=0)

        # --- Plot Costs ---
        ax_cost.set_title('Algorithm Final Cost Comparison')
        ax_cost.set_ylabel('Weighted Cost')
        ax_cost.grid(True, axis='y', linestyle='--', linewidth=0.5, zorder=0)

        bars_cost = ax_cost.bar(x_pos, [c if math.isfinite(c) else max_cost_plot * 1.1 for c in costs], # Plot INF slightly above max valid
                                width=bar_width, color=colors, alpha=0.75, zorder=3)

        # Add annotations and styling for feasibility/infinity
        for i, bar in enumerate(bars_cost):
             height = costs[i]
             is_inf = not math.isfinite(height)
             is_feasible = feasibility_flags[i]
             bar_alpha = 0.75 if is_feasible else 0.4 # Dim infeasible bars
             bar_hatch = '' if is_feasible else '//' # Hatch infeasible bars

             bar.set_alpha(bar_alpha)
             if bar_hatch: bar.set_hatch(bar_hatch)

             # Annotation text
             cost_text = 'INF' if is_inf else f'{height:.2f}'
             feasibility_text = '(Feasible)' if is_feasible else '(Infeasible)'
             label_text = f"{cost_text}\n{feasibility_text}"

             # Position annotation
             plot_height = bar.get_height()
             # Adjust y offset based on plot scale
             y_offset = max_cost_plot * 0.02 if max_cost_plot > 0 else 0.1
             text_y_pos = plot_height + y_offset

             ax_cost.text(bar.get_x() + bar.get_width() / 2., text_y_pos, label_text,
                          ha='center', va='bottom', fontsize='x-small', zorder=5,
                          # Optional: highlight INF or infeasible differently
                          bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5, ec='none') if is_inf else None)

        # Set x-axis ticks and labels for cost plot
        ax_cost.set_xticks(x_pos)
        ax_cost.set_xticklabels(algo_names, rotation=30, ha='right', fontsize='small')
        ax_cost.tick_params(axis='x', which='major', length=0) # Hide x tick marks if desired

        # Set y-limit dynamically based on valid data
        if max_cost_plot > 0:
            cost_padding = max_cost_plot * 0.15 # Increased padding for annotations
            ax_cost.set_ylim([0, max_cost_plot + cost_padding])
        else: # All costs were INF or 0
            ax_cost.set_ylim([0, 1]) # Default small range


        # --- Plot Times ---
        ax_time.set_title('Algorithm Computation Time Comparison')
        ax_time.set_ylabel('Computation Time (seconds)')
        ax_time.set_xlabel('Algorithm')
        ax_time.grid(True, axis='y', linestyle='--', linewidth=0.5, zorder=0)

        bars_time = ax_time.bar(x_pos, times, width=bar_width, color=colors, alpha=0.75, zorder=3)

        # Add time annotations
        for i, bar in enumerate(bars_time):
            height = bar.get_height()
            # Adjust y offset based on plot scale
            y_offset = max_time_plot * 0.02 if max_time_plot > 0 else 0.1
            text_y_pos = height + y_offset
            ax_time.text(bar.get_x() + bar.get_width() / 2., text_y_pos, f'{height:.2f}s',
                         ha='center', va='bottom', fontsize='x-small', zorder=5)

        # Set x-axis ticks and labels for time plot
        ax_time.set_xticks(x_pos)
        ax_time.set_xticklabels(algo_names, rotation=30, ha='right', fontsize='small')

        # Set y-limit dynamically based on valid data
        if max_time_plot > 0:
            time_padding = max_time_plot * 0.15 # Increased padding for annotations
            ax_time.set_ylim([0, max_time_plot + time_padding])
        else:
            ax_time.set_ylim([0, 1]) # Default small range

        # --- Final Redraw ---
        try:
            # Adjust layout to prevent labels overlapping titles etc.
            ax_cost.figure.tight_layout(rect=[0, 0.05, 1, 0.95]) # Rect = [left, bottom, right, top]
            ax_cost.figure.canvas.draw_idle()
            logger.info("Comparison plots generated successfully.")
        except AttributeError:
            logger.debug("Comparison plots generated (no canvas to draw).")
        except Exception as e:
             logger.error(f"Error during final comparison plot adjustments or drawing: {e}", exc_info=True)


# --- Standalone Testing Block (Optional) ---
if __name__ == '__main__':
    """
    Example usage for testing the PlotGenerator class independently.
    """
    logger.info("Running plot_generator.py in standalone test mode.")

    # --- Create Dummy Results Data ---
    # Structure mimicking output from the updated route_optimizer
    dummy_results = {
        'genetic_algorithm': {
            'result_data': {
                'weighted_cost': 1500.5, 'time': 2.5, 'is_feasible': True,
                'cost_history': [3000, 2500, 2000, 1800, 1600, 1550, 1500.5]
            },
            'computation_time': 10.2
        },
        'simulated_annealing': {
             'result_data': {
                 'weighted_cost': 1850.0, 'time': 2.8, 'is_feasible': False, # Example infeasible
                 'cost_history': [3000, 2800, 2400, 2200, 1900, 1850, 1850]
             },
             'computation_time': 5.5
        },
        'pso_optimizer': {
             'result_data': {
                 'weighted_cost': 1600.8, 'time': 2.2, 'is_feasible': True,
                 'cost_history': [float('inf'), 2900, 2200, 1900, 1700, 1650, 1600.8] # Example starting inf
             },
             'computation_time': 8.1
        },
        'greedy_heuristic': {
             'result_data': {
                 'weighted_cost': 2500.0, 'time': 1.8, 'is_feasible': True,
                 'cost_history': [2500.0] # Single point history
             },
             'computation_time': 0.1
        },
        'failed_algorithm': {
             'run_error': 'Something went wrong',
             'result_data': None,
             'computation_time': 1.0
        }
    }

    # --- Create Plots ---
    plot_gen = PlotGenerator()

    # 1. Iteration Curve Plot
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_gen.plot_iteration_curves(dummy_results, ax1)
    fig1.suptitle("Test Iteration Curve Plot")
    # Save the plot
    try:
        iter_filename = "output/charts/test_iteration_curves.png"
        os.makedirs(os.path.dirname(iter_filename), exist_ok=True)
        fig1.savefig(iter_filename, dpi=150, bbox_inches='tight')
        logger.info(f"Test iteration plot saved to {iter_filename}")
    except Exception as e:
        logger.error(f"Failed to save test iteration plot: {e}")


    # 2. Comparison Plot
    # Need two axes for comparison (cost and time)
    fig2, (ax_cost_test, ax_time_test) = plt.subplots(2, 1, figsize=(8, 8), sharex=False) # Don't share x for clarity
    plot_gen.plot_comparison_bars(dummy_results, ax_cost_test, ax_time_test)
    fig2.suptitle("Test Comparison Plot")
    # Save the plot
    try:
        comp_filename = "output/charts/test_comparison_bars.png"
        os.makedirs(os.path.dirname(comp_filename), exist_ok=True)
        fig2.savefig(comp_filename, dpi=150, bbox_inches='tight')
        logger.info(f"Test comparison plot saved to {comp_filename}")
    except Exception as e:
        logger.error(f"Failed to save test comparison plot: {e}")


    # Optional: Show plots if run interactively
    # plt.show()

    logger.info("Standalone plot generation test finished.")