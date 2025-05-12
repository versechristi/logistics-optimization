import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider, RadioButtons
import random
import heapq # For A* priority queue

# --- Configuration Parameters ---
MAP_SIZE = np.array([600, 600])  # Increased map size for more spacing
GROUND_Z = 0
MAX_BUILDING_HEIGHT = 120
NUM_BUILDINGS = 25 # Adjusted for better spacing
MIN_BUILDING_SEPARATION = 15 # Minimum distance between buildings
BUILDING_PLACEMENT_ATTEMPTS = 100 # Attempts to place a building without collision

NUM_CUSTOMERS = 6
NUM_DEPOTS = 3
NUM_PATH_SEGMENTS_FOR_STEPPED_PATH = 6 # Segments for fly-over parts
DRONE_MAX_FLYOVER_HEIGHT = 40 # Drones can fly over buildings up to this height
DRONE_TALL_BUILDING_THRESHOLD = 60 # Buildings taller than this are strictly 'tall' for A* detour
ASTAR_GRID_CELL_SIZE = 20 # Size of each cell in the A* grid
ASTAR_DETOUR_ALTITUDE = 25 # Default altitude for A* detour paths (must be < DRONE_MAX_FLYOVER_HEIGHT)


VEHICLE_SIZE = np.array([10, 6, 5])
VEHICLE_COLOR = 'blue'
VEHICLE_PATH_COLOR = 'darkblue'

DRONE_SIZE = 2.5 # Drone's characteristic size
MULTI_DRONE_COLORS = ['red', 'purple', 'deepskyblue', 'magenta', 'saddlebrown', 'lime', 'teal', 'gold']

CUSTOMER_MARKER_DIMS = np.array([5, 5, 5])
CUSTOMER_COLOR = 'green'
DEPOT_SIZE = np.array([25, 25, 8]) # Slightly smaller depots
DEPOT_COLOR = 'darkorange'

BUILDING_COLOR = (0.7, 0.7, 0.7)
BUILDING_ALPHA = 0.5

PATH_LINEWIDTH = 1.5
PATH_ALPHA = 0.8

DRONE_FLIGHT_CLEARANCE = 10
DRONE_MAX_ALTITUDE = 180 # Absolute max Z
MIN_DRONE_CRUISE_ALTITUDE = 20


# --- A* Pathfinding Components ---
class AStarNode:
    def __init__(self, position, parent=None):
        self.position = position  # (x_idx, y_idx) on grid
        self.parent = parent
        self.g = 0  # Cost from start
        self.h = 0  # Heuristic cost to end
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other): # For heapq
        return self.f < other.f

    def __hash__(self): # For visited set
        return hash(self.position)

def heuristic(a, b): # Manhattan distance for grid
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_grid_map_and_conversion(world_map_size, cell_size, buildings_list, tall_building_threshold):
    grid_width = int(world_map_size[0] / cell_size)
    grid_height = int(world_map_size[1] / cell_size)
    grid_map = [[0 for _ in range(grid_height)] for _ in range(grid_width)] # 0 = walkable

    def world_to_grid(world_pos_xy):
        grid_x = int((world_pos_xy[0] + world_map_size[0] / 2) / cell_size)
        grid_y = int((world_pos_xy[1] + world_map_size[1] / 2) / cell_size)
        return (max(0, min(grid_x, grid_width - 1)), max(0, min(grid_y, grid_height - 1)))

    def grid_to_world(grid_pos_ij): # Returns center of grid cell
        world_x = (grid_pos_ij[0] + 0.5) * cell_size - world_map_size[0] / 2
        world_y = (grid_pos_ij[1] + 0.5) * cell_size - world_map_size[1] / 2
        return np.array([world_x, world_y])

    for building in buildings_list:
        if building['height'] > tall_building_threshold:
            b_pos_xy, b_size_lw = building['position_xy'], building['size_lw']
            # Mark all grid cells covered by this tall building as obstacles
            # Iterate over the building's footprint in world coordinates
            # and convert each part to grid cells.
            # Simplified: mark cells around the building's center.
            center_gx, center_gy = world_to_grid(b_pos_xy)
            # Approximate extent in grid cells
            extent_x_cells = int(b_size_lw[0] / cell_size / 2) +1
            extent_y_cells = int(b_size_lw[1] / cell_size / 2) +1

            for i in range(max(0, center_gx - extent_x_cells), min(grid_width, center_gx + extent_x_cells + 1)):
                for j in range(max(0, center_gy - extent_y_cells), min(grid_height, center_gy + extent_y_cells + 1)):
                    grid_map[i][j] = 1 # Mark as obstacle
    return grid_map, world_to_grid, grid_to_world


def a_star_search(grid_map, start_grid_pos, end_grid_pos, allow_diagonal=False):
    start_node = AStarNode(start_grid_pos)
    end_node = AStarNode(end_grid_pos)

    open_list = []
    heapq.heappush(open_list, start_node)
    closed_set = set()

    grid_width = len(grid_map)
    grid_height = len(grid_map[0])

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path

        (x, y) = current_node.position
        neighbors = []
        if allow_diagonal:
            offsets = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        else:
            offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in offsets:
            neighbor_pos = (x + dx, y + dy)
            if 0 <= neighbor_pos[0] < grid_width and \
               0 <= neighbor_pos[1] < grid_height and \
               grid_map[neighbor_pos[0]][neighbor_pos[1]] == 0: # Check walkable
                if neighbor_pos in closed_set:
                    continue

                neighbor_node = AStarNode(neighbor_pos, current_node)
                neighbor_node.g = current_node.g + 1 # Assuming cost of 1 per step
                neighbor_node.h = heuristic(neighbor_node.position, end_node.position)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                # Check if neighbor is in open_list and if this path is better
                if any(open_node for open_node in open_list if open_node == neighbor_node and neighbor_node.g >= open_node.g):
                    continue
                heapq.heappush(open_list, neighbor_node)
    return None # Path not found

# --- Helper function to draw a cuboid ---
def plot_cuboid(ax, base_center_xy, size_lwh, color='gray', alpha=0.8, z_bottom=0):
    l, w, h = size_lwh; x_c, y_c = base_center_xy
    x_min, x_max = x_c - l / 2, x_c + l / 2; y_min, y_max = y_c - w / 2, y_c + w / 2
    z_min, z_max = z_bottom, z_bottom + h
    v = np.array([[x_min,y_min,z_min],[x_max,y_min,z_min],[x_max,y_max,z_min],[x_min,y_max,z_min],
                  [x_min,y_min,z_max],[x_max,y_min,z_max],[x_max,y_max,z_max],[x_min,y_max,z_max]])
    f = [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],[v[0],v[1],v[5],v[4]],
         [v[1],v[2],v[6],v[5]],[v[2],v[3],v[7],v[6]],[v[3],v[0],v[4],v[7]]]
    ax.add_collection3d(Poly3DCollection(f, facecolors=color, linewidths=0.5, edgecolors='k', alpha=alpha))

# --- Entity Data Structures & Plotting ---
def plot_customer_marker(ax, customer_dict):
    pos, dims, cid = customer_dict["target_position_3d"], customer_dict["marker_dims_lwh"], customer_dict['id']
    plot_cuboid(ax, pos[:2], dims, customer_dict["color"], z_bottom=pos[2] - dims[2] / 2, alpha=1.0)
    ax.text(pos[0], pos[1], pos[2] + dims[2], f"C{cid[1:]}", color='darkgreen', fontsize=6, ha='center')

# --- City Generation with Non-Overlapping Buildings ---
def generate_city_elements():
    buildings_list = []
    for i in range(NUM_BUILDINGS):
        placed = False
        for attempt in range(BUILDING_PLACEMENT_ATTEMPTS):
            pos_xy = np.random.rand(2) * (MAP_SIZE * 0.85) - (MAP_SIZE * 0.425) # Confine to central area
            size_lw = np.random.rand(2) * 35 + 10  # length, width between 10-45
            height = np.random.rand() * MAX_BUILDING_HEIGHT + 15 # Min height 15
            
            new_building = {'position_xy': pos_xy, 'size_lw': size_lw, 'height': height}
            
            # Check for collision with existing buildings
            collision = False
            for existing_b in buildings_list:
                dx = abs(new_building['position_xy'][0] - existing_b['position_xy'][0])
                dy = abs(new_building['position_xy'][1] - existing_b['position_xy'][1])
                
                min_dist_x = (new_building['size_lw'][0] + existing_b['size_lw'][0]) / 2 + MIN_BUILDING_SEPARATION
                min_dist_y = (new_building['size_lw'][1] + existing_b['size_lw'][1]) / 2 + MIN_BUILDING_SEPARATION
                
                if dx < min_dist_x and dy < min_dist_y:
                    collision = True
                    break
            
            if not collision:
                buildings_list.append({
                    'id': f"b{i}", 'type': "building", 'position_xy': pos_xy, 'size_lw': size_lw, 'height': height,
                    'color': BUILDING_COLOR, 'alpha': BUILDING_ALPHA
                })
                placed = True
                break
        # if not placed: print(f"Warning: Could not place building {i} without collision after {BUILDING_PLACEMENT_ATTEMPTS} attempts.")

    depots_list = []
    for i in range(NUM_DEPOTS):
        angle = (i / NUM_DEPOTS) * 2 * np.pi + np.pi/4 # Offset start angle
        radius = MAP_SIZE[0] * 0.5 # Place depots on periphery
        pos_xy = [radius * np.cos(angle), radius * np.sin(angle)]
        depot_h = DEPOT_SIZE[2]
        depots_list.append({
            'id': f"d{i}", 'type': "depot", 'position_xy': np.array(pos_xy), 'size_lw': DEPOT_SIZE[:2], 'height': depot_h,
            'color': DEPOT_COLOR, 'alpha': 0.9, 'launch_z': GROUND_Z + depot_h + DRONE_SIZE / 2
        })

    customers_list = []
    if buildings_list or NUM_CUSTOMERS > 0: # Allow ground customers if no buildings
        for i in range(NUM_CUSTOMERS):
            if buildings_list and random.random() > 0.1: # 90% chance on building if buildings exist
                assoc_building = random.choice(buildings_list)
                min_dz = GROUND_Z + CUSTOMER_MARKER_DIMS[2]/2
                max_dz = GROUND_Z + assoc_building['height'] - CUSTOMER_MARKER_DIMS[2]/2
                delivery_z = random.uniform(min_dz, max_dz) if max_dz > min_dz else (min_dz+max_dz)/2
                c_target_xy = assoc_building['position_xy'] + (np.random.rand(2)-0.5)*assoc_building['size_lw']*0.4
            else: # Ground customer or fallback
                assoc_building = None
                delivery_z = GROUND_Z + CUSTOMER_MARKER_DIMS[2]/2
                c_target_xy = np.random.rand(2) * (MAP_SIZE*0.7) - (MAP_SIZE*0.35)

            customers_list.append({
                'id': f"c{i}", 'type': "customer", 'associated_building': assoc_building, 'delivery_z_abs': delivery_z,
                'marker_dims_lwh': CUSTOMER_MARKER_DIMS, 'color': CUSTOMER_COLOR,
                'target_position_3d': np.array([c_target_xy[0], c_target_xy[1], delivery_z])
            })
    return buildings_list, customers_list, depots_list

# --- Path Interpolation ---
def get_point_on_path(path_nodes, t_normalized):
    # (Same as before)
    if not path_nodes: return None
    if len(path_nodes) == 1: return path_nodes[0]
    t_normalized = np.clip(t_normalized, 0, 1)
    segment_lengths = [np.linalg.norm(path_nodes[i+1] - path_nodes[i]) for i in range(len(path_nodes)-1)]
    total_length = sum(segment_lengths)
    if total_length < 1e-6: return path_nodes[0]
    target_dist = t_normalized * total_length
    current_dist = 0
    for i in range(len(segment_lengths)):
        if segment_lengths[i] < 1e-6: # Skip zero-length segments
            if i == len(segment_lengths) -1 : return path_nodes[i+1] # if it's the last one
            continue 
        if current_dist + segment_lengths[i] >= target_dist - 1e-6 : # Add tolerance
            segment_t = (target_dist - current_dist) / segment_lengths[i]
            return path_nodes[i] + segment_t * (path_nodes[i+1] - path_nodes[i])
        current_dist += segment_lengths[i]
    return path_nodes[-1]


# --- Smart Path Generation ---
def is_segment_intersecting_building(p1_xy, p2_xy, building_xy, building_lw):
    # Basic line segment vs AABB intersection (simplified)
    # This should ideally be a more robust Segment vs Polygon test.
    # For now, check if building's AABB is "near" the line segment's AABB.
    b_min_x, b_max_x = building_xy[0] - building_lw[0]/2, building_xy[0] + building_lw[0]/2
    b_min_y, b_max_y = building_xy[1] - building_lw[1]/2, building_xy[1] + building_lw[1]/2

    line_min_x, line_max_x = min(p1_xy[0], p2_xy[0]), max(p1_xy[0], p2_xy[0])
    line_min_y, line_max_y = min(p1_xy[1], p2_xy[1]), max(p1_xy[1], p2_xy[1])
    
    # Check for AABB overlap
    if not (line_max_x < b_min_x - DRONE_SIZE or line_min_x > b_max_x + DRONE_SIZE or \
            line_max_y < b_min_y - DRONE_SIZE or line_min_y > b_max_y + DRONE_SIZE):
        # Further check: very simple - does building center lie close to the line?
        # This is a placeholder for a proper line-rectangle intersection.
        if np.linalg.norm(p2_xy - p1_xy) < 1e-6: return False # Zero length segment
        # Distance from building center to line segment
        # For simplicity, assume if AABBs overlap significantly, it's an intersection.
        return True 
    return False

def get_max_obstacle_height_on_segment(p1_xy, p2_xy, buildings, max_height_cap=DRONE_MAX_ALTITUDE):
    max_h = GROUND_Z
    for b in buildings:
        if is_segment_intersecting_building(p1_xy, p2_xy, b['position_xy'], b['size_lw']):
            if b['height'] <= max_height_cap : # Only consider obstacles we might fly over
                 max_h = max(max_h, b['height'])
    return max_h

def generate_stepped_flyover_path_segment(start_3d, end_target_xy, current_buildings, num_steps, max_flyover_h):
    """Generates a stepped path from start_3d to end_target_xy, flying over low obstacles."""
    path_segment_nodes = []
    current_pos_3d = np.copy(start_3d)

    total_horizontal_vec = end_target_xy - start_3d[:2]
    if np.linalg.norm(total_horizontal_vec) < 1e-3: # No horizontal movement
        # Just ensure Z is fine, or direct connection if already aligned
        path_segment_nodes.append(np.array([end_target_xy[0], end_target_xy[1], start_3d[2]])) # Keep current Z
        return path_segment_nodes

    step_horizontal_vec = total_horizontal_vec / num_steps
    
    for i in range(num_steps):
        seg_start_xy = current_pos_3d[:2]
        seg_end_xy = start_3d[:2] + (i + 1) * step_horizontal_vec
        
        obs_h = get_max_obstacle_height_on_segment(seg_start_xy, seg_end_xy, current_buildings, max_flyover_h)
        target_alt = min(max(obs_h + DRONE_FLIGHT_CLEARANCE, MIN_DRONE_CRUISE_ALTITUDE), max_flyover_h)

        if abs(current_pos_3d[2] - target_alt) > 0.1 : # Vertical adjustment
            current_pos_3d[2] = target_alt
            path_segment_nodes.append(np.copy(current_pos_3d))
        
        current_pos_3d[:2] = seg_end_xy # Horizontal movement
        path_segment_nodes.append(np.copy(current_pos_3d))
        
    return path_segment_nodes


def generate_drone_smart_path(depot, customer, all_buildings, grid_map_data):
    depot_launch_pos = np.array([depot['position_xy'][0], depot['position_xy'][1], depot['launch_z']])
    customer_target_pos = customer['target_position_3d']
    
    grid_map, world_to_grid, grid_to_world = grid_map_data

    path_nodes = [np.copy(depot_launch_pos)]
    current_pos_3d = np.copy(depot_launch_pos)

    # Analyze direct path for tall obstacles
    direct_path_tall_obstacles = []
    for b in all_buildings:
        if b['height'] > DRONE_TALL_BUILDING_THRESHOLD: # Check against stricter tall threshold
            if is_segment_intersecting_building(depot_launch_pos[:2], customer_target_pos[:2], b['position_xy'], b['size_lw']):
                direct_path_tall_obstacles.append(b)
                break # One is enough to trigger A*

    if not direct_path_tall_obstacles:
        # Strategy 1: Direct Stepped Fly-Over (if no strictly tall buildings on direct path)
        # Fly over obstacles up to DRONE_MAX_FLYOVER_HEIGHT
        stepped_nodes = generate_stepped_flyover_path_segment(current_pos_3d, customer_target_pos[:2], all_buildings, NUM_PATH_SEGMENTS_FOR_STEPPED_PATH, DRONE_MAX_FLYOVER_HEIGHT)
        if stepped_nodes: path_nodes.extend(stepped_nodes); current_pos_3d = path_nodes[-1]
    
    else:
        # Strategy 2: A* Detour for Tall Buildings
        start_grid = world_to_grid(depot_launch_pos[:2])
        end_grid = world_to_grid(customer_target_pos[:2])
        
        astar_grid_path = a_star_search(grid_map, start_grid, end_grid, allow_diagonal=True)

        if astar_grid_path:
            # Initial ascent to A* detour altitude
            if abs(current_pos_3d[2] - ASTAR_DETOUR_ALTITUDE) > 0.1:
                current_pos_3d[2] = ASTAR_DETOUR_ALTITUDE
                path_nodes.append(np.copy(current_pos_3d))

            for grid_node_pos in astar_grid_path[1:]: # Skip start node (already there or ascended)
                world_wp_xy = grid_to_world(grid_node_pos)
                
                # Optional: minor fly-over for low obstacles on this A* segment
                # For simplicity, maintain ASTAR_DETOUR_ALTITUDE
                # alt_for_astar_segment = ASTAR_DETOUR_ALTITUDE
                # obs_h_astar = get_max_obstacle_height_on_segment(current_pos_3d[:2], world_wp_xy, all_buildings, ASTAR_DETOUR_ALTITUDE)
                # alt_for_astar_segment = min(max(obs_h_astar + DRONE_FLIGHT_CLEARANCE, MIN_DRONE_CRUISE_ALTITUDE), ASTAR_DETOUR_ALTITUDE + DRONE_FLIGHT_CLEARANCE/2)

                # if abs(current_pos_3d[2] - alt_for_astar_segment) > 0.1:
                #    current_pos_3d[2] = alt_for_astar_segment
                #    path_nodes.append(np.copy(current_pos_3d)) # Vertical adjustment at start of A* leg

                current_pos_3d[:2] = world_wp_xy # Fly to next A* waypoint XY
                path_nodes.append(np.copy(current_pos_3d))
            
            # After A* path, transition to final approach (fly over low buildings to customer XY)
            if np.linalg.norm(current_pos_3d[:2] - customer_target_pos[:2]) > ASTAR_GRID_CELL_SIZE /2 : # If A* didn't end exactly at customer XY
                 stepped_to_customer_xy = generate_stepped_flyover_path_segment(current_pos_3d, customer_target_pos[:2], all_buildings, max(1,NUM_PATH_SEGMENTS_FOR_STEPPED_PATH//2), DRONE_MAX_FLYOVER_HEIGHT)
                 if stepped_to_customer_xy : path_nodes.extend(stepped_to_customer_xy); current_pos_3d = path_nodes[-1]

        else: # A* failed - fallback to high fly-over (less ideal)
            print(f"A* pathfinding failed for C{customer['id'][1:]} from D{depot['id'][-1]}. Defaulting to high fly-over.")
            # High direct fly-over, ignoring DRONE_MAX_FLYOVER_HEIGHT restriction for this fallback
            obs_h = get_max_obstacle_height_on_segment(current_pos_3d[:2], customer_target_pos[:2], all_buildings, DRONE_MAX_ALTITUDE)
            fallback_alt = min(max(obs_h + DRONE_FLIGHT_CLEARANCE, MIN_DRONE_CRUISE_ALTITUDE), DRONE_MAX_ALTITUDE)
            if abs(current_pos_3d[2] - fallback_alt) > 0.1:
                current_pos_3d[2] = fallback_alt; path_nodes.append(np.copy(current_pos_3d))
            current_pos_3d[:2] = customer_target_pos[:2]; path_nodes.append(np.copy(current_pos_3d))


    # Final descent to customer target Z
    # Ensure XY is aligned with customer at current flight Z before final descent
    if not np.allclose(current_pos_3d[:2], customer_target_pos[:2]):
        current_pos_3d[:2] = customer_target_pos[:2]
        path_nodes.append(np.copy(current_pos_3d))
    
    if abs(current_pos_3d[2] - customer_target_pos[2]) > 0.1 : # If not already at customer Z
        current_pos_3d[2] = customer_target_pos[2]
        path_nodes.append(np.copy(current_pos_3d))
    
    if not np.allclose(path_nodes[-1], customer_target_pos): # Ensure last point IS customer target
        path_nodes.append(np.copy(customer_target_pos))

    # Create return path (simple reverse for now)
    path_from_customer = path_nodes[::-1] 
    full_path = path_nodes + path_from_customer[1:]

    unique_path = [] # Filter duplicates
    if full_path:
        unique_path.append(full_path[0])
        for pt_idx in range(1, len(full_path)):
            if not np.allclose(full_path[pt_idx], full_path[pt_idx-1], atol=0.1):
                unique_path.append(full_path[pt_idx])
    return unique_path


# --- Main Simulation Setup ---
fig = plt.figure(figsize=(17, 14))
ax = fig.add_subplot(111, projection='3d')
plt.style.use('seaborn-v0_8-whitegrid')

random.seed(50) 
np.random.seed(50)
generated_buildings, generated_customers, generated_depots = generate_city_elements()

# Prepare grid map data for A* (once)
grid_map_for_astar, world_to_grid_func, grid_to_world_func = get_grid_map_and_conversion(
    MAP_SIZE, ASTAR_GRID_CELL_SIZE, generated_buildings, DRONE_TALL_BUILDING_THRESHOLD
)
GRID_MAP_DATA = (grid_map_for_astar, world_to_grid_func, grid_to_world_func)


# Plot static elements
for b in generated_buildings:
    plot_cuboid(ax, b['position_xy'], [b['size_lw'][0], b['size_lw'][1], b['height']], b['color'], alpha=b['alpha'])
for i, d in enumerate(generated_depots):
    plot_cuboid(ax, d['position_xy'], [d['size_lw'][0], d['size_lw'][1], d['height']], d['color'], alpha=0.95)
    ax.text(d['position_xy'][0], d['position_xy'][1], d['height'] + 7, f"D{i}", color='black', fontsize=7, ha='center', fontweight='bold')
for c in generated_customers:
    plot_customer_marker(ax, c)

# --- SCENARIO PATH DEFINITIONS ---
scenario_vehicle_paths = {} # Not used in this version, can be added back if needed
scenario_drone_paths_lists = {}

# SCENARIO: "Multi-Drone Smart Delivery"
s_multi_smart_paths = []
if generated_depots and generated_customers:
    # Ensure every customer has a supply
    for cust_idx, customer in enumerate(generated_customers):
        depot_for_this_customer = generated_depots[cust_idx % len(generated_depots)] # Round-robin assignment
        
        print(f"Pathing for Customer C{customer['id'][1:]} from Depot D{depot_for_this_customer['id'][-1]}")
        drone_path = generate_drone_smart_path(depot_for_this_customer, customer, generated_buildings, GRID_MAP_DATA)
        
        if drone_path:
            s_multi_smart_paths.append(drone_path)
        else:
            print(f"Warning: No path generated for Customer {customer['id']}")

scenario_drone_paths_lists["Multi-Drone Smart Delivery"] = s_multi_smart_paths
SCENARIOS = list(scenario_drone_paths_lists.keys())
current_scenario_name = SCENARIOS[0] if SCENARIOS else None

# --- Global GFX storage ---
current_drone_gfx_objs = []
current_drone_paths_gfx_objs_list = [] # Stores the actual Line3D objects

def update_animation(frame_t_normalized):
    global current_drone_gfx_objs, current_drone_paths_gfx_objs_list

    # Clear previous dynamic graphics
    for drone_obj in current_drone_gfx_objs: drone_obj.remove()
    current_drone_gfx_objs.clear()
    for line_obj in current_drone_paths_gfx_objs_list: line_obj.remove() # Direct list of Line3D objects
    current_drone_paths_gfx_objs_list.clear()

    if not current_scenario_name or not scenario_drone_paths_lists.get(current_scenario_name):
        ax.set_title("Scenario not ready or no paths."); fig.canvas.draw_idle(); return

    active_drone_paths_list = scenario_drone_paths_lists.get(current_scenario_name, [])
    
    drone_positions_this_frame = []
    for i, drone_path_nodes in enumerate(active_drone_paths_list):
        if not drone_path_nodes: continue # Skip if a path is empty
        drone_pos = get_point_on_path(drone_path_nodes, frame_t_normalized)
        if drone_pos is not None: drone_positions_this_frame.append(drone_pos)
        
        dx, dy, dz = zip(*drone_path_nodes)
        color_idx = i % len(MULTI_DRONE_COLORS)
        line_obj, = ax.plot(dx, dy, dz, color=MULTI_DRONE_COLORS[color_idx], linestyle=':', linewidth=PATH_LINEWIDTH, alpha=PATH_ALPHA, zorder=1) # zorder for paths
        current_drone_paths_gfx_objs_list.append(line_obj)
    
    for i, drone_pos in enumerate(drone_positions_this_frame):
        color_idx = i % len(MULTI_DRONE_COLORS)
        label = f"Drone {i+1}"
        drone_gfx = ax.scatter([drone_pos[0]], [drone_pos[1]], [drone_pos[2]], 
                               color=MULTI_DRONE_COLORS[color_idx], s=60, depthshade=True, 
                               label=label, edgecolors='black', linewidth=0.5, zorder=10) # zorder for drones
        current_drone_gfx_objs.append(drone_gfx)

    ax.set_title(f"Scenario: {current_scenario_name} (Time: {frame_t_normalized:.2f})", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Unique legend entries
    if by_label: ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize='xx-small')
    fig.canvas.draw_idle()

# --- Matplotlib UI Setup ---
plt.subplots_adjust(left=0.03, bottom=0.12, right=0.99, top=0.95)
ax_slider = plt.axes([0.15, 0.04, 0.7, 0.02])
time_slider = Slider(ax_slider, 'Time', 0.0, 1.0, valinit=0.0, valstep=0.002, color='cornflowerblue', track_color='lightgrey')

ax_radio = plt.axes([0.01, 0.85, 0.18, 0.12], frame_on=False)
if SCENARIOS: scenario_radio = RadioButtons(ax_radio, SCENARIOS, active=0, activecolor='royalblue')
else: ax_radio.text(0.5, 0.5, "No Scenarios", ha='center'); scenario_radio = None

def on_slider_update(val): update_animation(time_slider.val)
def on_radio_update(label):
    global current_scenario_name; current_scenario_name = label
    time_slider.reset(); update_animation(0.0)

time_slider.on_changed(on_slider_update)
if scenario_radio: scenario_radio.on_clicked(on_radio_update)

# --- Plotting Configuration ---
ax.set_xlabel("X (m)", fontsize=8); ax.set_ylabel("Y (m)", fontsize=8); ax.set_zlabel("Z (m)", fontsize=8)
ax.tick_params(axis='both', which='major', labelsize=6)
plot_margin = 50
ax.set_xlim([-MAP_SIZE[0]/2 - plot_margin, MAP_SIZE[0]/2 + plot_margin])
ax.set_ylim([-MAP_SIZE[1]/2 - plot_margin, MAP_SIZE[1]/2 + plot_margin])
ax.set_zlim([GROUND_Z, DRONE_MAX_ALTITUDE + 20])
ax.view_init(elev=30, azim=-135) # Viewing angle

if current_scenario_name: update_animation(0.0)
else: ax.set_title("Simulation Environment Ready", fontsize=10)
plt.suptitle("3D Drone Smart Delivery Simulation", fontsize=12, y=0.98)
plt.show()