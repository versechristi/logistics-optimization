import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider, RadioButtons
import random
import heapq # For A* priority queue
import sys # For setattr in lambda for radio button

# --- Configuration Parameters ---
MAP_SIZE = np.array([600, 600])
GROUND_Z = 0
MAX_BUILDING_HEIGHT = 110 # Slightly reduced max height for more fly-around scenarios
NUM_BUILDINGS = 25
MIN_BUILDING_SEPARATION = 20 # Increased separation
BUILDING_PLACEMENT_ATTEMPTS = 100

NUM_CUSTOMERS = 6 # As requested: 5-6 customers
NUM_DEPOTS = 2 # Example: 2 depots

# Drone Pathfinding Parameters
NUM_PATH_SEGMENTS_FOR_STEPPED_PATH = 5 # Fewer segments for possibly shorter drone paths
DRONE_MAX_FLYOVER_HEIGHT = 40
DRONE_TALL_BUILDING_THRESHOLD_FOR_ASTAR = 65 # Buildings taller are A* obstacles for drones
DRONE_ASTAR_DETOUR_ALTITUDE = 30

# Vehicle Pathfinding Parameters
VEHICLE_ASTAR_GRID_CELL_SIZE = 18
VEHICLE_Z_LEVEL = GROUND_Z + 2.0 # Vehicle driving height (center of vehicle)

# General A* Parameters
ASTAR_GRID_CELL_SIZE_DRONE = 22

VEHICLE_SIZE = np.array([7, 4, 4]) # Slightly smaller vehicle
VEHICLE_COLOR = 'darkolivegreen' # Changed vehicle color
VEHICLE_PATH_COLOR = 'olivedrab'

DRONE_SIZE = 2.0
MULTI_DRONE_COLORS = ['orangered', 'darkviolet', 'dodgerblue', 'deeppink', 'sienna', 'chartreuse']

CUSTOMER_MARKER_DIMS = np.array([4, 4, 4]) # Smaller customer markers
CUSTOMER_COLOR = 'mediumblue'
DEPOT_SIZE = np.array([22, 22, 7]) # Smaller depots
DEPOT_COLOR = 'chocolate'

BUILDING_COLOR = (0.85, 0.85, 0.85)
BUILDING_ALPHA = 0.25

PATH_LINEWIDTH = 1.7
PATH_ALPHA = 0.9

DRONE_FLIGHT_CLEARANCE = 12
DRONE_MAX_ALTITUDE = 160
MIN_DRONE_CRUISE_ALTITUDE = 25


# --- A* Pathfinding Components ---
class AStarNode:
    def __init__(self, position, parent=None):
        self.position = position; self.parent = parent
        self.g = 0; self.h = 0; self.f = 0
    def __eq__(self, other): return self.position == other.position
    def __lt__(self, other): return self.f < other.f if self.f != other.f else self.h < other.h # Tie-breaking
    def __hash__(self): return hash(self.position)

def heuristic(a, b): return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) # Euclidean distance

def get_grid_map_and_conversion_functions(world_map_size, cell_size, buildings_list, obstacle_height_threshold):
    # (Implementation largely same as before)
    grid_width = int(world_map_size[0] / cell_size) + 1
    grid_height = int(world_map_size[1] / cell_size) + 1
    grid_map = [[0 for _ in range(grid_height)] for _ in range(grid_width)]

    def world_to_grid(world_pos_xy):
        grid_x = int((world_pos_xy[0] + world_map_size[0] / 2) / cell_size)
        grid_y = int((world_pos_xy[1] + world_map_size[1] / 2) / cell_size)
        return (max(0, min(grid_x, grid_width - 1)), max(0, min(grid_y, grid_height - 1)))

    def grid_to_world(grid_pos_ij):
        world_x = (grid_pos_ij[0] + 0.5) * cell_size - world_map_size[0] / 2
        world_y = (grid_pos_ij[1] + 0.5) * cell_size - world_map_size[1] / 2
        return np.array([world_x, world_y])

    for building in buildings_list:
        if building['height'] > obstacle_height_threshold:
            b_pos_xy, b_size_lw = building['position_xy'], building['size_lw']
            # More robustly mark cells by iterating over building's corners in grid
            corners_world = [
                [b_pos_xy[0] - b_size_lw[0]/2, b_pos_xy[1] - b_size_lw[1]/2],
                [b_pos_xy[0] + b_size_lw[0]/2, b_pos_xy[1] - b_size_lw[1]/2],
                [b_pos_xy[0] - b_size_lw[0]/2, b_pos_xy[1] + b_size_lw[1]/2],
                [b_pos_xy[0] + b_size_lw[0]/2, b_pos_xy[1] + b_size_lw[1]/2],
            ]
            grid_corners_x = [world_to_grid(c)[0] for c in corners_world]
            grid_corners_y = [world_to_grid(c)[1] for c in corners_world]
            min_gx, max_gx = min(grid_corners_x), max(grid_corners_x)
            min_gy, max_gy = min(grid_corners_y), max(grid_corners_y)

            for i in range(min_gx, max_gx + 1):
                for j in range(min_gy, max_gy + 1):
                    if 0 <= i < grid_width and 0 <= j < grid_height:
                        grid_map[i][j] = 1
    return grid_map, world_to_grid, grid_to_world

def a_star_search(grid_map, start_grid_pos, end_grid_pos, allow_diagonal=True): # Allow diagonal with sqrt(2) cost
    # (Implementation largely same as before, ensure diagonal cost is handled)
    start_node = AStarNode(start_grid_pos); end_node = AStarNode(end_grid_pos)
    open_list = []; heapq.heappush(open_list, start_node)
    closed_set = set()
    grid_width, grid_height = len(grid_map), len(grid_map[0])

    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node.position in closed_set: continue # Check if already processed
        closed_set.add(current_node.position)

        if current_node == end_node:
            path = []; temp = current_node
            while temp: path.append(temp.position); temp = temp.parent
            return path[::-1]

        (x, y) = current_node.position
        offsets = [(0,1,1),(0,-1,1),(1,0,1),(-1,0,1)] # dx, dy, cost
        if allow_diagonal:
            offsets.extend([(1,1,np.sqrt(2)),(1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(-1,-1,np.sqrt(2))])
        
        for dx, dy, cost in offsets:
            neighbor_pos = (x + dx, y + dy)
            if not (0 <= neighbor_pos[0] < grid_width and 0 <= neighbor_pos[1] < grid_height and \
                    grid_map[neighbor_pos[0]][neighbor_pos[1]] == 0): # Removed closed_set check here for re-opening
                continue
            
            if neighbor_pos in closed_set : continue # If already processed fully, skip

            neighbor_node = AStarNode(neighbor_pos, current_node)
            neighbor_node.g = current_node.g + cost
            neighbor_node.h = heuristic(neighbor_node.position, end_node.position)
            neighbor_node.f = neighbor_node.g + neighbor_node.h
            
            # If another node with same position is in open_list with lower g, skip this one
            if any(on.position == neighbor_pos and neighbor_node.g >= on.g for on in open_list): continue
            heapq.heappush(open_list, neighbor_node)
    return None

# --- Plotting and City Generation ---
def plot_cuboid(ax, base_center_xy, size_lwh, color='gray', alpha=0.8, z_bottom=0, edgecolor='k', linewidth=0.5):
    # (Same as before)
    l,w,h=size_lwh; x_c,y_c=base_center_xy; x_min,x_max=x_c-l/2,x_c+l/2; y_min,y_max=y_c-w/2,y_c+w/2; z_min,z_max=z_bottom,z_bottom+h
    v=np.array([[x_min,y_min,z_min],[x_max,y_min,z_min],[x_max,y_max,z_min],[x_min,y_max,z_min],[x_min,y_min,z_max],[x_max,y_min,z_max],[x_max,y_max,z_max],[x_min,y_max,z_max]])
    f=[[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],[v[0],v[1],v[5],v[4]],[v[1],v[2],v[6],v[5]],[v[2],v[3],v[7],v[6]],[v[3],v[0],v[4],v[7]]]
    ax.add_collection3d(Poly3DCollection(f,facecolors=color,linewidths=linewidth,edgecolors=edgecolor,alpha=alpha))

def plot_customer_marker(ax, customer_dict, is_vehicle_target=False):
    pos, dims, cid = customer_dict["target_position_3d"], customer_dict["marker_dims_lwh"], customer_dict['id']
    if is_vehicle_target:
        # Vehicle ground drop-off point (at Z=0)
        ground_marker_z_bottom = GROUND_Z
        plot_cuboid(ax, pos[:2], [dims[0]*1.2, dims[1]*1.2, dims[2]*0.5], VEHICLE_COLOR, z_bottom=ground_marker_z_bottom, alpha=0.5, edgecolor='darkgreen', linewidth=0.6)
        ax.text(pos[0], pos[1], ground_marker_z_bottom + dims[2]*0.5 + 1, f"C{cid[1:]}-V", color=VEHICLE_COLOR, fontsize=5, ha='center', fontweight='bold')
    else:
        # Drone delivery point (original Z)
        plot_cuboid(ax, pos[:2], dims, customer_dict["color"], z_bottom=pos[2] - dims[2]/2, alpha=1.0, edgecolor='navy', linewidth=0.7)
        ax.text(pos[0], pos[1], pos[2] + dims[2] + 1, f"C{cid[1:]}-D", color=customer_dict["color"], fontsize=5, ha='center', fontweight='bold')

def generate_city_elements(): # (Same non-overlapping logic)
    buildings_list = []
    for i in range(NUM_BUILDINGS):
        placed = False
        for _ in range(BUILDING_PLACEMENT_ATTEMPTS):
            pos_xy = np.random.rand(2)*(MAP_SIZE*0.8)-(MAP_SIZE*0.4) # More central
            size_lw = np.random.rand(2)*30+15 # Min size 15x15
            height = np.random.rand()*MAX_BUILDING_HEIGHT+20 # Min height 20
            new_b = {'position_xy':pos_xy,'size_lw':size_lw,'height':height}
            collision = any(abs(new_b['position_xy'][0]-eb['position_xy'][0])<(new_b['size_lw'][0]+eb['size_lw'][0])/2+MIN_BUILDING_SEPARATION and \
                            abs(new_b['position_xy'][1]-eb['position_xy'][1])<(new_b['size_lw'][1]+eb['size_lw'][1])/2+MIN_BUILDING_SEPARATION
                            for eb in buildings_list)
            if not collision:
                buildings_list.append({'id':f"b{i}",**new_b,'color':BUILDING_COLOR,'alpha':BUILDING_ALPHA}); placed=True; break
    depots_list = []
    for i in range(NUM_DEPOTS):
        angle = (i/NUM_DEPOTS)*2*np.pi + np.pi/(NUM_DEPOTS*2) # Even spread
        radius = MAP_SIZE[0]*0.48 # Closer to map edge
        pos_xy = [radius*np.cos(angle),radius*np.sin(angle)]
        depot_h = DEPOT_SIZE[2]
        depots_list.append({'id':f"d{i}",'type':"depot",'position_xy':np.array(pos_xy),'size_lw':DEPOT_SIZE[:2],
                            'height':depot_h,'color':DEPOT_COLOR,'alpha':0.98,
                            'launch_z':GROUND_Z+depot_h+DRONE_SIZE/2,
                            'vehicle_launch_z':GROUND_Z+VEHICLE_SIZE[2]/2}) # Vehicle launch Z is vehicle center Z
    customers_list = []
    for i in range(NUM_CUSTOMERS):
        assoc_b = random.choice(buildings_list) if buildings_list and random.random()>0.1 else None
        if assoc_b:
            delivery_z = random.uniform(GROUND_Z+CUSTOMER_MARKER_DIMS[2]/2+5, GROUND_Z+assoc_b['height']-CUSTOMER_MARKER_DIMS[2]/2) # ensure above ground for drone
            delivery_z = max(delivery_z, MIN_DRONE_CRUISE_ALTITUDE - DRONE_FLIGHT_CLEARANCE + 1) # ensure practical drone delivery height
            c_xy = assoc_b['position_xy']+(np.random.rand(2)-0.5)*assoc_b['size_lw']*0.25 # Closer to building center
        else: # Ground customer
            delivery_z = GROUND_Z + CUSTOMER_MARKER_DIMS[2]/2 + 5 # Drone delivery height for ground cust
            c_xy = np.random.rand(2)*(MAP_SIZE*0.5)-(MAP_SIZE*0.25) # More central ground customers
        customers_list.append({'id':f"c{i}",'type':"customer",'associated_building':assoc_b,'delivery_z_abs':delivery_z,
                               'marker_dims_lwh':CUSTOMER_MARKER_DIMS,'color':CUSTOMER_COLOR,
                               'target_position_3d':np.array([c_xy[0],c_xy[1],delivery_z])})
    return buildings_list, customers_list, depots_list

def get_point_on_path(path_nodes, t_normalized): # (Same)
    if not path_nodes: return None
    if len(path_nodes) == 1: return path_nodes[0]
    t_normalized = np.clip(t_normalized,0,1); seg_lens = [np.linalg.norm(path_nodes[i+1]-path_nodes[i]) for i in range(len(path_nodes)-1)]
    total_len = sum(seg_lens)
    if total_len<1e-6: return path_nodes[0]
    target_d = t_normalized*total_len; curr_d=0
    for i in range(len(seg_lens)):
        if seg_lens[i]<1e-6:
            if i==len(seg_lens)-1: return path_nodes[i+1]
            continue
        if curr_d+seg_lens[i] >= target_d-1e-6:
            return path_nodes[i]+((target_d-curr_d)/seg_lens[i])*(path_nodes[i+1]-path_nodes[i])
        curr_d+=seg_lens[i]
    return path_nodes[-1]

def is_segment_intersecting_building(p1_xy, p2_xy, building_xy, building_lw, margin=DRONE_SIZE): # (Same, with margin)
    b_min_x,b_max_x=building_xy[0]-building_lw[0]/2,building_xy[0]+building_lw[0]/2
    b_min_y,b_max_y=building_xy[1]-building_lw[1]/2,building_xy[1]+building_lw[1]/2
    l_min_x,l_max_x=min(p1_xy[0],p2_xy[0]),max(p1_xy[0],p2_xy[0])
    l_min_y,l_max_y=min(p1_xy[1],p2_xy[1]),max(p1_xy[1],p2_xy[1])
    return not (l_max_x<b_min_x-margin or l_min_x>b_max_x+margin or l_max_y<b_min_y-margin or l_min_y>b_max_y+margin)

def get_max_obstacle_height_on_segment(p1_xy, p2_xy, buildings, max_h_cap=DRONE_MAX_ALTITUDE, agent_clearance_margin=DRONE_SIZE): # (Same)
    max_h = GROUND_Z
    for b in buildings:
        if is_segment_intersecting_building(p1_xy,p2_xy,b['position_xy'],b['size_lw'], agent_clearance_margin) and b['height']<=max_h_cap:
            max_h = max(max_h, b['height'])
    return max_h

def generate_stepped_flyover_nodes(start_3d, end_target_xy, buildings, num_steps, max_flyover_h, clearance, min_alt): # (Same)
    nodes=[]; curr=np.copy(start_3d); total_h_vec=end_target_xy-start_3d[:2]
    if np.linalg.norm(total_h_vec)<1e-3: return [np.array([end_target_xy[0],end_target_xy[1],start_3d[2]])]
    step_h_vec=total_h_vec/num_steps
    for i in range(num_steps):
        s_start_xy,s_end_xy=curr[:2],start_3d[:2]+(i+1)*step_h_vec
        obs_h=get_max_obstacle_height_on_segment(s_start_xy,s_end_xy,buildings,max_flyover_h)
        target_alt=min(max(obs_h+clearance,min_alt),max_flyover_h)
        if abs(curr[2]-target_alt)>0.1: curr[2]=target_alt; nodes.append(np.copy(curr))
        curr[:2]=s_end_xy; nodes.append(np.copy(curr))
    return nodes

def clean_path_nodes(path_nodes_list, tolerance=0.1): # (Same)
    if not path_nodes_list: return []
    cleaned=[path_nodes_list[0]]
    for pt in path_nodes_list[1:]:
        if not np.allclose(pt,cleaned[-1],atol=tolerance): cleaned.append(pt)
    return cleaned

def generate_drone_smart_path(depot, customer, all_buildings, grid_map_data_drone): # (Mostly Same)
    launch_pos, target_pos = np.array([*depot['position_xy'],depot['launch_z']]), customer['target_position_3d']
    grid_map, world2grid, grid2world = grid_map_data_drone
    path, curr_pos = [np.copy(launch_pos)], np.copy(launch_pos)
    direct_path_tall_obs = [b for b in all_buildings if b['height']>DRONE_TALL_BUILDING_THRESHOLD_FOR_ASTAR and \
                            is_segment_intersecting_building(launch_pos[:2],target_pos[:2],b['position_xy'],b['size_lw'])]
    if not direct_path_tall_obs:
        stepped_nodes = generate_stepped_flyover_nodes(curr_pos,target_pos[:2],all_buildings,NUM_PATH_SEGMENTS_FOR_STEPPED_PATH,DRONE_MAX_FLYOVER_HEIGHT,DRONE_FLIGHT_CLEARANCE,MIN_DRONE_CRUISE_ALTITUDE)
        if stepped_nodes: path.extend(stepped_nodes); curr_pos=path[-1]
    else:
        start_g,end_g=world2grid(launch_pos[:2]),world2grid(target_pos[:2])
        astar_g_path=a_star_search(grid_map,start_g,end_g,allow_diagonal=True)
        if astar_g_path:
            if abs(curr_pos[2]-DRONE_ASTAR_DETOUR_ALTITUDE)>0.1: curr_pos[2]=DRONE_ASTAR_DETOUR_ALTITUDE; path.append(np.copy(curr_pos))
            for g_node_pos in astar_g_path[1:]:
                world_wp_xy = grid2world(g_node_pos)
                obs_h_astar = get_max_obstacle_height_on_segment(curr_pos[:2],world_wp_xy,all_buildings,DRONE_ASTAR_DETOUR_ALTITUDE+5)
                alt_astar_leg = min(max(obs_h_astar+DRONE_FLIGHT_CLEARANCE,MIN_DRONE_CRUISE_ALTITUDE),DRONE_ASTAR_DETOUR_ALTITUDE+10)
                alt_astar_leg = max(alt_astar_leg, DRONE_ASTAR_DETOUR_ALTITUDE)
                if abs(curr_pos[2]-alt_astar_leg)>0.1: curr_pos[2]=alt_astar_leg; path.append(np.copy(curr_pos))
                curr_pos[:2]=world_wp_xy; path.append(np.copy(curr_pos))
            if np.linalg.norm(curr_pos[:2]-target_pos[:2])>ASTAR_GRID_CELL_SIZE_DRONE*0.7:
                 stepped_to_cust=generate_stepped_flyover_nodes(curr_pos,target_pos[:2],all_buildings,max(1,NUM_PATH_SEGMENTS_FOR_STEPPED_PATH//2),DRONE_MAX_FLYOVER_HEIGHT,DRONE_FLIGHT_CLEARANCE,MIN_DRONE_CRUISE_ALTITUDE)
                 if stepped_to_cust: path.extend(stepped_to_cust); curr_pos=path[-1]
        else:
            print(f"Drone A* fail C{customer['id'][1:]}. High fly-over."); obs_h=get_max_obstacle_height_on_segment(curr_pos[:2],target_pos[:2],all_buildings,DRONE_MAX_ALTITUDE)
            f_alt=min(max(obs_h+DRONE_FLIGHT_CLEARANCE,MIN_DRONE_CRUISE_ALTITUDE),DRONE_MAX_ALTITUDE)
            if abs(curr_pos[2]-f_alt)>0.1: curr_pos[2]=f_alt; path.append(np.copy(curr_pos))
            curr_pos[:2]=target_pos[:2]; path.append(np.copy(curr_pos))
    if not np.allclose(curr_pos[:2],target_pos[:2]): curr_pos[:2]=target_pos[:2]; path.append(np.copy(curr_pos))
    if abs(curr_pos[2]-target_pos[2])>0.1: curr_pos[2]=target_pos[2]; path.append(np.copy(curr_pos))
    if not np.allclose(path[-1],target_pos): path.append(np.copy(target_pos))
    return_path=path[::-1]; full_path=path+return_path[1:]
    return clean_path_nodes(full_path)

def generate_vehicle_ground_path(depot, customer_target_xy, vehicle_z, grid_map_data_vehicle): # (Same)
    depot_start_xy = depot['position_xy']
    depot_start_3d = np.array([depot_start_xy[0], depot_start_xy[1], vehicle_z])
    customer_ground_target_3d = np.array([customer_target_xy[0], customer_target_xy[1], vehicle_z])
    grid_map, world_to_grid, grid_to_world = grid_map_data_vehicle
    path_nodes = [np.copy(depot_start_3d)]
    start_grid, end_grid = world_to_grid(depot_start_xy), world_to_grid(customer_target_xy)
    astar_grid_path = a_star_search(grid_map, start_grid, end_grid, allow_diagonal=True)
    if astar_grid_path:
        for grid_node_pos in astar_grid_path[1:]:
            world_wp_xy = grid_to_world(grid_node_pos)
            path_nodes.append(np.array([world_wp_xy[0], world_wp_xy[1], vehicle_z]))
    else:
        print(f"Vehicle A* FAIL for target {customer_target_xy}. Direct path.")
        path_nodes.append(np.copy(customer_ground_target_3d))
    if not path_nodes or not np.allclose(path_nodes[-1][:2], customer_ground_target_3d[:2]):
        path_nodes.append(np.copy(customer_ground_target_3d))
    return_path_nodes = path_nodes[::-1]; full_vehicle_path = path_nodes + return_path_nodes[1:]
    return clean_path_nodes(full_vehicle_path)

# --- Main Simulation Setup ---
fig = plt.figure(figsize=(17,14)); ax = fig.add_subplot(111,projection='3d')
plt.style.use('seaborn-v0_8-notebook')

random.seed(60); np.random.seed(60) # Seed for consistency
generated_buildings, generated_customers, generated_depots = generate_city_elements()

GRID_MAP_DATA_DRONE = get_grid_map_and_conversion_functions(MAP_SIZE,ASTAR_GRID_CELL_SIZE_DRONE,generated_buildings,DRONE_TALL_BUILDING_THRESHOLD_FOR_ASTAR)
GRID_MAP_DATA_VEHICLE = get_grid_map_and_conversion_functions(MAP_SIZE,VEHICLE_ASTAR_GRID_CELL_SIZE,generated_buildings,0) # All buildings are obstacles

for b in generated_buildings: plot_cuboid(ax,b['position_xy'],[b['size_lw'][0],b['size_lw'][1],b['height']],b['color'],alpha=b['alpha'],linewidth=0.2,edgecolor='dimgrey')
for i,d in enumerate(generated_depots): 
    plot_cuboid(ax,d['position_xy'],[d['size_lw'][0],d['size_lw'][1],d['height']],d['color'],alpha=0.99,edgecolor='saddlebrown',linewidth=0.7)
    ax.text(d['position_xy'][0],d['position_xy'][1],d['height']+6,f"D{i}",color='black',fontsize=6,ha='center',fontweight='bold')
# Customer markers will be plotted by scenario logic to distinguish targets

# --- SCENARIO PATH DEFINITIONS ---
scenario_drone_paths_lists = {}
scenario_vehicle_paths_lists = {}

# SCENARIO 1: "Hybrid Drone & Vehicle Delivery"
hybrid_drone_paths = []
hybrid_vehicle_paths = []
# Assign customers to drone or vehicle (e.g., first N/2 to drones, rest to vehicles)
num_drone_deliveries = NUM_CUSTOMERS // 2 
# or fixed, e.g., num_drone_deliveries = 3 if NUM_CUSTOMERS == 6 else NUM_CUSTOMERS // 2

plotted_customer_ids_for_markers = set()

for cust_idx, customer in enumerate(generated_customers):
    depot = generated_depots[cust_idx % len(generated_depots)] # Round-robin depot assignment
    
    if cust_idx < num_drone_deliveries: # This customer gets drone delivery
        print(f"Hybrid: Drone path for C{customer['id'][1:]} from D{depot['id'][-1]}")
        drone_path = generate_drone_smart_path(depot, customer, generated_buildings, GRID_MAP_DATA_DRONE)
        if drone_path: hybrid_drone_paths.append(drone_path)
        if customer['id'] not in plotted_customer_ids_for_markers:
            plot_customer_marker(ax, customer, is_vehicle_target=False)
            plotted_customer_ids_for_markers.add(customer['id'])
    else: # This customer gets vehicle delivery
        customer_ground_xy = customer['target_position_3d'][:2]
        print(f"Hybrid: Vehicle path for C{customer['id'][1:]} to XY({customer_ground_xy}) from D{depot['id'][-1]}")
        vehicle_path = generate_vehicle_ground_path(depot, customer_ground_xy, depot['vehicle_launch_z'], GRID_MAP_DATA_VEHICLE)
        if vehicle_path: hybrid_vehicle_paths.append(vehicle_path)
        if customer['id'] not in plotted_customer_ids_for_markers:
            plot_customer_marker(ax, customer, is_vehicle_target=True)
            plotted_customer_ids_for_markers.add(customer['id'])
            
scenario_drone_paths_lists["Hybrid Delivery"] = hybrid_drone_paths
scenario_vehicle_paths_lists["Hybrid Delivery"] = hybrid_vehicle_paths


# SCENARIO 2: "All Drones Smart Delivery"
s_all_drones_paths = []
for cust_idx, customer in enumerate(generated_customers):
    depot = generated_depots[cust_idx % len(generated_depots)]
    drone_path = generate_drone_smart_path(depot, customer, generated_buildings, GRID_MAP_DATA_DRONE)
    if drone_path: s_all_drones_paths.append(drone_path)
    # Plot drone target markers if this scenario is selected (or do it once initially)
    # For simplicity, markers plotted once above. Can be scenario specific if needed.
scenario_drone_paths_lists["All Drones Smart Delivery"] = s_all_drones_paths
scenario_vehicle_paths_lists["All Drones Smart Delivery"] = []

# SCENARIO 3: "All Vehicles Ground Delivery"
s_all_vehicles_paths = []
for cust_idx, customer in enumerate(generated_customers):
    depot = generated_depots[cust_idx % len(generated_depots)]
    customer_ground_xy = customer['target_position_3d'][:2] 
    vehicle_path = generate_vehicle_ground_path(depot, customer_ground_xy, depot['vehicle_launch_z'], GRID_MAP_DATA_VEHICLE)
    if vehicle_path: s_all_vehicles_paths.append(vehicle_path)
scenario_drone_paths_lists["All Vehicles Ground Delivery"] = []
scenario_vehicle_paths_lists["All Vehicles Ground Delivery"] = s_all_vehicles_paths


SCENARIOS = list(scenario_drone_paths_lists.keys()) # All scenarios will be keys
if not SCENARIOS and scenario_vehicle_paths_lists: # If only vehicle scenarios exist
    SCENARIOS = list(scenario_vehicle_paths_lists.keys())

current_scenario_name = SCENARIOS[0] if SCENARIOS else "No Scenarios Defined"


# --- Global GFX storage & Animation Update ---
current_drone_gfx_objs, current_vehicle_gfx_objs = [],[]
current_drone_path_lines, current_vehicle_path_lines = [],[]

def update_animation(frame_t_normalized):
    global current_drone_gfx_objs,current_vehicle_gfx_objs,current_drone_path_lines,current_vehicle_path_lines
    for obj_list in [current_drone_gfx_objs,current_vehicle_gfx_objs,current_drone_path_lines,current_vehicle_path_lines]:
        for item in obj_list: item.remove()
        obj_list.clear()

    if not current_scenario_name or current_scenario_name == "No Scenarios Defined":
        ax.set_title("No scenario selected or defined."); fig.canvas.draw_idle(); return
    
    active_drone_paths=scenario_drone_paths_lists.get(current_scenario_name,[])
    active_vehicle_paths=scenario_vehicle_paths_lists.get(current_scenario_name,[])

    for i,path_nodes in enumerate(active_drone_paths):
        if not path_nodes: continue
        pos=get_point_on_path(path_nodes,frame_t_normalized)
        if pos is None: continue
        dx,dy,dz=zip(*path_nodes); color=MULTI_DRONE_COLORS[i%len(MULTI_DRONE_COLORS)]
        line, = ax.plot(dx,dy,dz,color=color,ls=':',lw=PATH_LINEWIDTH,alpha=PATH_ALPHA,zorder=4) # Drones above vehicles
        current_drone_path_lines.append(line)
        gfx=ax.scatter([pos[0]],[pos[1]],[pos[2]],color=color,s=50,depthshade=True,label=f"Drone {i+1}",edgecolors='k',lw=0.3,zorder=11)
        current_drone_gfx_objs.append(gfx)

    for i,path_nodes in enumerate(active_vehicle_paths):
        if not path_nodes: continue
        pos=get_point_on_path(path_nodes,frame_t_normalized)
        if pos is None: continue
        dx,dy,dz=zip(*path_nodes)
        line, = ax.plot(dx,dy,dz,color=VEHICLE_PATH_COLOR,ls='-',lw=PATH_LINEWIDTH+0.6,alpha=PATH_ALPHA,zorder=3)
        current_vehicle_path_lines.append(line)
        gfx=ax.scatter([pos[0]],[pos[1]],[pos[2]],color=VEHICLE_COLOR,s=70,marker='s',depthshade=True,label=f"Vehicle {i+1}",edgecolors='k',lw=0.3,zorder=10)
        current_vehicle_gfx_objs.append(gfx)

    ax.set_title(f"Scenario: {current_scenario_name} (Time: {frame_t_normalized:.2f})",fontsize=10)
    handles,labels=ax.get_legend_handles_labels(); by_label=dict(zip(labels,handles))
    if by_label: ax.legend(by_label.values(),by_label.keys(),loc='upper right',bbox_to_anchor=(0.99,0.99),fontsize='xx-small')
    fig.canvas.draw_idle()

# --- Matplotlib UI Setup ---
plt.subplots_adjust(left=0.02,bottom=0.12,right=0.99,top=0.94) # More space for UI
ax_slider=plt.axes([0.20,0.03,0.65,0.025]); time_slider=Slider(ax_slider,'Time',0,1,valinit=0,valstep=0.002,color='teal',track_color='lightcyan')
ax_radio=plt.axes([0.01,0.75,0.20,0.20],frame_on=False) # Radio buttons area
if SCENARIOS and current_scenario_name != "No Scenarios Defined":
    scenario_radio=RadioButtons(ax_radio,SCENARIOS,active=SCENARIOS.index(current_scenario_name),activecolor='darkcyan')
    scenario_radio.on_clicked(lambda label:(setattr(sys.modules[__name__],'current_scenario_name',label),time_slider.reset(),update_animation(0.0)))
else: ax_radio.text(0.5,0.5,"No Scenarios",ha='center')
time_slider.on_changed(lambda val:update_animation(time_slider.val))

# --- Plotting Configuration ---
ax.set_xlabel("X (m)",fontsize=8); ax.set_ylabel("Y (m)",fontsize=8); ax.set_zlabel("Z (m)",fontsize=8)
ax.tick_params(axis='both',which='major',labelsize=6); plot_margin=50
ax.set_xlim([-MAP_SIZE[0]/2-plot_margin,MAP_SIZE[0]/2+plot_margin]); ax.set_ylim([-MAP_SIZE[1]/2-plot_margin,MAP_SIZE[1]/2+plot_margin])
ax.set_zlim([GROUND_Z,DRONE_MAX_ALTITUDE+5]); ax.view_init(elev=35,azim=-120) # Higher elevation view

if current_scenario_name != "No Scenarios Defined": update_animation(0.0)
else: ax.set_title("Simulation Environment Ready. Please define scenarios.",fontsize=10)
plt.suptitle("3D Hybrid Drone & Vehicle Delivery Simulation",fontsize=13,y=0.98)
plt.show()