import matplotlib.pyplot as plt
import numpy as np

# Apply Seaborn style (this style might set its own default fonts)
plt.style.use('seaborn-v0_8-whitegrid')
# We no longer need to explicitly set Chinese fonts if all text is English.
# Matplotlib will use its default English-compatible fonts.
plt.rcParams['axes.unicode_minus'] = False # Ensure minus sign displays correctly

# Data
# Original Chinese labels: labels = ['单独车辆配送', '单独无人机配送', '协同配送']
# New order: 'Collaborative Delivery' first
labels = ['Collaborative Delivery', 'Solo Vehicle Delivery', 'Solo Drone Delivery']
operation_times = [2.06, 17.17, 168.06] # Operation Time (Hours) - Reordered
total_costs = [1006.70, 1373.83, 10083.84] # Total Cost (Yuan) - Reordered

x = np.arange(len(labels))
bar_width = 0.5

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- Subplot 1: Operation Time Comparison ---
ax1 = axes[0]
color_time_bar = '#87CEEB'  # SkyBlue
color_time_line = '#4682B4' # SteelBlue
edge_color_time = '#4A90E2' # Slightly darker blue border

# Operation Time Bar Chart
rects_time = ax1.bar(x, operation_times, bar_width,
                    label='Operation Time (Hours)', color=color_time_bar,
                    edgecolor=edge_color_time, linewidth=0.5, alpha=0.85)
# Operation Time Line Chart
line_time = ax1.plot(x, operation_times, color=color_time_line,
                     marker='o', markersize=7, linestyle='-', linewidth=2,
                     label='Operation Time Trend')

ax1.set_xlabel('Delivery Method', fontsize=12, labelpad=10)
ax1.set_ylabel('Operation Time (Hours)', fontsize=12, labelpad=10)
ax1.set_title('Operation Time Comparison', fontsize=15, pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10, rotation=15, ha="right") # Rotate for better fit
ax1.tick_params(axis='y', labelsize=10)
ax1.legend(fontsize=10, loc='upper left') # Changed loc to 'upper left'

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Subplot 2: Total Cost Comparison ---
ax2 = axes[1]
color_cost_bar = '#FFB6C1'  # LightPink
color_cost_line = '#CD5C5C' # IndianRed
edge_color_cost = '#D87093' # Slightly darker pink border

# Total Cost Bar Chart
rects_cost = ax2.bar(x, total_costs, bar_width,
                     label='Total Cost (Yuan)', color=color_cost_bar,
                     edgecolor=edge_color_cost, linewidth=0.5, alpha=0.85)
# Total Cost Line Chart
line_cost = ax2.plot(x, total_costs, color=color_cost_line,
                     marker='s', markersize=7, linestyle='--', linewidth=2,
                     label='Total Cost Trend')

ax2.set_xlabel('Delivery Method', fontsize=12, labelpad=10)
ax2.set_ylabel('Total Cost (Yuan)', fontsize=12, labelpad=10)
ax2.set_title('Total Cost Comparison', fontsize=15, pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10, rotation=15, ha="right") # Rotate for better fit
ax2.tick_params(axis='y', labelsize=10)
ax2.legend(fontsize=10, loc='upper left') # Changed loc to 'upper left'

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Helper function to add data labels to bars
def autolabel_bar(rects, ax, color='dimgray', rotation=0, fontsize=9):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color=color, rotation=rotation, fontsize=fontsize)

# Add data labels to bars in both subplots
autolabel_bar(rects_time, ax1)
autolabel_bar(rects_cost, ax2)

# Main title for the entire figure
# Original Chinese: fig.suptitle('小规模数据下协同配送与单独配送对比', ...)
fig.suptitle('Comparison of Collaborative vs. Solo Delivery (Small-Scale Data)',
             fontsize=18, y=0.99, weight='bold')

# Adjust layout to prevent labels from overlapping and to make space for suptitle
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.show()