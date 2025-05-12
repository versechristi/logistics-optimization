import re
import tkinter as tk
from tkinter import filedialog
import math
import folium # For map generation
from folium.plugins import AntPath # Optional for animated routes
from collections import defaultdict

# --- 0. Constants (UPDATED FOR DRONES) ---
DRONE_CAPACITY = 5.0      # kg (Changed from VEHICLE_CAPACITY)
COST_PER_KM_DRONE = 1.0   # yuan (Changed from COST_PER_KM)
DRONE_SPEED_KMPH = 60.0   # km/h (Changed from VEHICLE_SPEED_KMPH)

# --- 1. Data Structures ---
class Point:
    def __init__(self, id, name, p_type, latitude, longitude, demand=0.0):
        self.id = id
        self.name = name
        self.type = p_type # "Logistics Center", "Sales Outlet", "Customer"
        self.coords = (latitude, longitude)
        self.demand = demand
        self.remaining_demand = demand # Used during routing

    def __repr__(self):
        return f"{self.name} ({self.type}) @ {self.coords}"

# --- 2. Data Extraction (Should be robust from previous iterations) ---
def extract_detailed_map_data(file_path):
    logistics_centers = []
    sales_outlets = []
    customers = []
    
    if not file_path:
        print("No file selected.")
        return None, None, None
        
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        # This pattern assumes each marker's full JS definition is somewhat isolated
        # or at least that subsequent L.marker calls act as delimiters.
        marker_blocks_pattern = re.compile(
            r'(var\s+(marker_\w+)\s*=\s*L\.marker\(\s*\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]\s*,\s*\{\}\s*\)\s*\.addTo\([^)]+\);'
            r'[\s\S]*?' # Non-greedy match for all JS related to this marker
            # Positive lookahead for the start of the next marker variable declaration, 
            # or the layer_control variable, or end of script tag, to delimit the block.
            r'(?=\s*var\s+marker_\w+\s*=\s*L\.marker|\s*var\s+layer_control_|\s*<\/script>))'
        )
        
        icon_pattern_in_block = re.compile(r'L\.AwesomeMarkers\.icon\([^)]*?"icon":\s*"([^"]+)"')
        # Regex to capture names like "Logistics Center 1", "Sales Outlet X", "Customer Y" from typical popup/tooltip HTML
        name_pattern_in_block = re.compile(r'\.(?:bindPopup|bindTooltip)\([^<]*?<div[^>]*>([\w\s.-]+?\d*)\s*(?:<br>|\(Demand:|\<\/div>)')
        demand_pattern_in_block = re.compile(r'Demand:\s*(-?\d+\.\d+)')


        lc_id_counter, so_id_counter, cust_id_counter = 1, 1, 1

        for match in marker_blocks_pattern.finditer(html_content):
            block_content = match.group(1) # The entire JS block for one marker
            # marker_var_name = match.group(2) # e.g., marker_2f62665dc3a96194101c930dbc9052dc
            lat_str = match.group(3)
            lon_str = match.group(4)
            
            lat = float(lat_str)
            lon = float(lon_str)

            icon_match = icon_pattern_in_block.search(block_content)
            icon_type_raw = "unknown"
            if icon_match:
                icon_type_raw = icon_match.group(1)

            name_match = name_pattern_in_block.search(block_content)
            point_name_raw = "Unknown Point"
            if name_match: # Prioritize name from popup/tooltip
                point_name_raw = name_match.group(1).strip().replace("  ", " ")
            
            point_type = "Unknown"
            demand = 0.0

            if icon_type_raw == "home":
                point_type = "Logistics Center"
                name = point_name_raw if "Logistics Center" in point_name_raw else f"Logistics Center {lc_id_counter}"
                logistics_centers.append(Point(f"LC{lc_id_counter}", name, point_type, lat, lon))
                lc_id_counter += 1
            elif icon_type_raw == "store":
                point_type = "Sales Outlet"
                name = point_name_raw if "Sales Outlet" in point_name_raw else f"Sales Outlet {so_id_counter}"
                sales_outlets.append(Point(f"SO{so_id_counter}", name, point_type, lat, lon))
                so_id_counter += 1
            elif icon_type_raw == "user":
                point_type = "Customer"
                name = point_name_raw if "Customer" in point_name_raw else f"Customer {cust_id_counter}"
                
                demand_match = demand_pattern_in_block.search(block_content)
                if demand_match:
                    demand = float(demand_match.group(1))
                customers.append(Point(f"C{cust_id_counter}", name, point_type, lat, lon, demand=demand))
                cust_id_counter += 1
        
        if not (logistics_centers and sales_outlets and customers):
            print("Warning: Not all types of points (LC, SO, Customer) were extracted. Please check HTML parsing and file content.")
            if not logistics_centers : print("Specifically, no Logistics Centers found.")
            if not sales_outlets : print("Specifically, no Sales Outlets found.")
            if not customers : print("Specifically, no Customers found.")


    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during data extraction: {e}")
        import traceback
        traceback.print_exc() 
        return None, None, None
        
    return logistics_centers, sales_outlets, customers


def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select the HTML map file",
        filetypes=(("HTML files", "*.html;*.htm"), ("All files", "*.*"))
    )
    return file_path

# --- 3. Geographical Calculations ---
def haversine_distance(coord1, coord2):
    R = 6371 
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# --- 4. Assignment Logic (Nearest Neighbor) ---
def assign_points(logistics_centers, sales_outlets, customers):
    so_to_lc_assignment = {} 
    customer_to_so_assignment = {} 
    
    if not logistics_centers:
        print("Error: No logistics centers found. Cannot proceed.")
        return None, None
    for so in sales_outlets:
        nearest_lc = min(logistics_centers, key=lambda lc: haversine_distance(so.coords, lc.coords))
        so_to_lc_assignment[so.id] = nearest_lc.id
        
    if not sales_outlets:
        print("Warning: No sales outlets found. Customers cannot be assigned to Sales Outlets.")
    else:
        for cust in customers:
            nearest_so = min(sales_outlets, key=lambda so_point: haversine_distance(cust.coords, so_point.coords))
            customer_to_so_assignment[cust.id] = nearest_so.id
            
    return so_to_lc_assignment, customer_to_so_assignment

# --- 5. Greedy Routing Algorithm (Now for Drones) ---
def plan_drone_routes_greedy(logistics_centers, sales_outlets, customers, so_to_lc, customer_to_so):
    all_tours = []
    drone_id_counter = 1 # Changed from vehicle_id_counter
    
    lc_map = {lc.id: lc for lc in logistics_centers}
    so_map = {so.id: so for so in sales_outlets}
    cust_map = {c.id: c for c in customers}

    for c in customers:
        c.remaining_demand = c.demand

    customers_grouped_by_so = defaultdict(list)
    if customer_to_so:
        for cust_id, so_id in customer_to_so.items():
            if cust_map.get(cust_id) and so_map.get(so_id):
                 customers_grouped_by_so[so_id].append(cust_map[cust_id])

    for so_id, so_customers in customers_grouped_by_so.items():
        current_so = so_map.get(so_id)
        if not current_so: continue
        
        assigned_lc_id = so_to_lc.get(so_id)
        if not assigned_lc_id: continue
        current_lc = lc_map.get(assigned_lc_id)
        if not current_lc: continue

        customers_needing_service = [c for c in so_customers if c.remaining_demand > 0]

        while any(c.remaining_demand > 0 for c in customers_needing_service):
            tour_path_nodes = [current_lc, current_so]
            tour_path_names = [current_lc.name, current_so.name]
            tour_distance = haversine_distance(current_lc.coords, current_so.coords)
            
            current_drone_load = 0.0 # Changed variable name
            current_location_node = current_so
            served_customer_details_on_tour = []

            while current_drone_load < DRONE_CAPACITY: # Use DRONE_CAPACITY
                next_customer_to_visit = None
                min_dist_to_next_customer = float('inf')
                eligible_customers = [c for c in customers_needing_service if c.remaining_demand > 0]
                if not eligible_customers: break

                for cust in eligible_customers:
                    dist = haversine_distance(current_location_node.coords, cust.coords)
                    if dist < min_dist_to_next_customer:
                        min_dist_to_next_customer = dist
                        next_customer_to_visit = cust
                
                if not next_customer_to_visit: break

                can_load_amount = min(next_customer_to_visit.remaining_demand, 
                                      DRONE_CAPACITY - current_drone_load) # Use DRONE_CAPACITY

                if can_load_amount > 0:
                    tour_path_nodes.append(next_customer_to_visit)
                    tour_path_names.append(next_customer_to_visit.name)
                    tour_distance += haversine_distance(current_location_node.coords, next_customer_to_visit.coords)
                    current_drone_load += can_load_amount
                    next_customer_to_visit.remaining_demand -= can_load_amount
                    served_customer_details_on_tour.append((next_customer_to_visit.name, can_load_amount))
                    current_location_node = next_customer_to_visit
                else: break
            
            if served_customer_details_on_tour:
                tour_path_nodes.append(current_lc)
                tour_path_names.append(current_lc.name)
                tour_distance += haversine_distance(current_location_node.coords, current_lc.coords)
                
                tour_time = tour_distance / DRONE_SPEED_KMPH if DRONE_SPEED_KMPH > 0 else 0 # Use DRONE_SPEED
                tour_cost = tour_distance * COST_PER_KM_DRONE # Use COST_PER_KM_DRONE
                
                all_tours.append({
                    "drone_id": f"D{drone_id_counter}", # Changed to drone_id
                    "origin_lc": current_lc.name,
                    "serviced_so": current_so.name,
                    "path_nodes_for_map": [p.coords for p in tour_path_nodes],
                    "path_names_for_report": tour_path_names,
                    "customers_served_details": served_customer_details_on_tour,
                    "total_load_kg": current_drone_load,
                    "distance_km": tour_distance,
                    "time_hours": tour_time,
                    "cost_yuan": tour_cost
                })
                drone_id_counter += 1
            
            customers_needing_service = [c for c in so_customers if c.remaining_demand > 0]
            if not customers_needing_service: break
    
    return all_tours

# --- 6. Report Generation (Text and Map) ---
def generate_text_report(filename, lcs, sos, custs, tours, total_cost, total_time, total_distance, delivery_mode="Drone"):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"--- {delivery_mode} Logistics Optimization Report ---\n\n") # Added delivery_mode
        f.write("1. Point Information:\n")
        f.write("  Logistics Centers:\n")
        for lc in lcs: f.write(f"    - {lc.name}: Lat={lc.coords[0]:.6f}, Lon={lc.coords[1]:.6f}\n")
        f.write("\n  Sales Outlets:\n")
        for so in sos: f.write(f"    - {so.name}: Lat={so.coords[0]:.6f}, Lon={so.coords[1]:.6f}\n")
        f.write("\n  Customers:\n")
        for c in custs: f.write(f"    - {c.name}: Lat={c.coords[0]:.6f}, Lon={c.coords[1]:.6f}, Demand={c.demand:.2f} kg\n")
        f.write("\n\n2. Delivery Route Information:\n")
        if not tours: f.write("  No routes generated.\n")
        for tour in tours:
            f.write(f"\n  {delivery_mode} ID: {tour.get('drone_id') or tour.get('vehicle_id')}\n") # Use generic or specific id
            f.write(f"    Route: {' -> '.join(tour['path_names_for_report'])}\n")
            f.write(f"    Origin LC: {tour['origin_lc']}, Serviced SO: {tour['serviced_so']}\n")
            f.write(f"    Customers Served & Quantity:\n")
            for cust_name, amount in tour['customers_served_details']: f.write(f"      - {cust_name}: {amount:.2f} kg\n")
            f.write(f"    Total Load: {tour['total_load_kg']:.2f} kg\n")
            f.write(f"    Distance: {tour['distance_km']:.2f} km\n")
            f.write(f"    Time: {tour['time_hours']:.2f} hours\n")
            f.write(f"    Cost: {tour['cost_yuan']:.2f} yuan\n")
        f.write("\n\n3. Summary:\n")
        f.write(f"  Total {delivery_mode}s Used (Trips): {len(tours)}\n") # Changed
        f.write(f"  Total Distance Covered: {total_distance:.2f} km\n")
        f.write(f"  Total Time Taken: {total_time:.2f} hours\n")
        f.write(f"  Total Operational Cost: {total_cost:.2f} yuan\n")
    print(f"Text report saved to {filename}")

def generate_interactive_map(filename, lcs, sos, custs, tours, delivery_mode="Drone"):
    if not (lcs or sos or custs): 
        map_center = [39.89, 116.40] 
        m = folium.Map(location=map_center, zoom_start=10)
        print(f"Interactive map saved to {filename} (empty as no points data).")
        m.save(filename)
        return

    if lcs: map_center = lcs[0].coords
    elif sos: map_center = sos[0].coords
    elif custs: map_center = custs[0].coords
    else: map_center = [39.89, 116.40] 

    m = folium.Map(location=map_center, zoom_start=11)
    folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    
    for lc in lcs:
        folium.Marker(location=lc.coords, popup=f"<b>{lc.name}</b><br>Logistics Center", tooltip=lc.name,
                      icon=folium.Icon(color='green', icon='home', prefix='fa')).add_to(m)
    for so in sos:
        folium.Marker(location=so.coords, popup=f"<b>{so.name}</b><br>Sales Outlet", tooltip=so.name,
                      icon=folium.Icon(color='blue', icon='store', prefix='fa')).add_to(m)
    for c in custs:
        folium.Marker(location=c.coords, popup=f"<b>{c.name}</b><br>Customer<br>Demand: {c.demand:.2f} kg", 
                      tooltip=f"{c.name} (Demand: {c.demand:.2f}kg)",
                      icon=folium.Icon(color='red', icon='user', prefix='fa')).add_to(m)

    route_colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9']
    color_index = 0
    
    if tours:
        fg_routes = folium.FeatureGroup(name=f"{delivery_mode} Routes", show=True) # Changed
        m.add_child(fg_routes)
        for tour in tours:
            path_coords = tour['path_nodes_for_map']
            color = route_colors[color_index % len(route_colors)]
            color_index += 1
            folium.PolyLine(
                locations=path_coords,
                tooltip=(f"{delivery_mode} {tour.get('drone_id') or tour.get('vehicle_id')}<br>" # Changed
                         f"Route: {' -> '.join(tour['path_names_for_report'])}<br>"
                         f"Distance: {tour['distance_km']:.2f} km"),
                color=color, weight=2.5, opacity=0.8
            ).add_to(fg_routes)

    legend_html = f"""
         <div style="position: fixed; 
                     bottom: 30px; left: 10px; width: 200px; height: auto; 
                     border:2px solid grey; z-index:9999; font-size:12px;
                     background-color:white; opacity:0.90; padding: 10px; border-radius: 5px;">
           <h4 style="margin-top:0; margin-bottom:5px; font-weight: bold; text-align:center;">Legend</h4>
           <div style="margin-bottom: 3px;"><i class="fa fa-home" style="color:green; font-size:14px;"></i>&nbsp; Logistics Center</div>
           <div style="margin-bottom: 3px;"><i class="fa fa-store" style="color:blue; font-size:14px;"></i>&nbsp; Sales Outlet</div>
           <div style="margin-bottom: 5px;"><i class="fa fa-user" style="color:red; font-size:14px;"></i>&nbsp; Customer</div>
           <hr style="border-top: 1px solid #bbb; margin: 4px 0;">
           <div style="margin-bottom: 3px;">
             <svg width="15" height="10"><line x1="0" y1="5" x2="15" y2="5" style="stroke:#e6194B;stroke-width:2.5"/></svg>
             &nbsp;{delivery_mode} Route <span style="font-size:10px;">(Colors vary)</span> 
           </div>
         </div>
    """ # Changed
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    
    m.save(filename)
    print(f"Interactive map saved to {filename}")

# --- 7. Main Execution ---
if __name__ == "__main__":
    html_file_input_path = select_file()
    
    if html_file_input_path:
        print(f"Processing file for DRONE delivery: {html_file_input_path}")
        lcs, sos, custs = extract_detailed_map_data(html_file_input_path)

        if lcs is None or sos is None or custs is None :
             print("Data extraction returned None. Cannot proceed.")
        elif not lcs:
            print("No Logistics Centers extracted. Routing cannot start.")
        elif not custs:
            print("No Customers extracted. No deliveries to plan.")
        # Note: If no SOs, the current logic for cust_so_assign and customers_grouped_by_so
        # will result in no tours, which is logical if SOs are mandatory for the LC->SO->C chain.
        else:
            print(f"Data extracted: {len(lcs)} LCs, {len(sos)} SOs, {len(custs)} Customers.")

            so_lc_assign, cust_so_assign = assign_points(lcs, sos, custs)

            if so_lc_assign is None : 
                print("Failed to assign Sales Outlets to Logistics Centers. Exiting.")
            else:
                for c_obj in custs: c_obj.remaining_demand = c_obj.demand

                # Call the drone routing function
                planned_drone_tours = plan_drone_routes_greedy(lcs, sos, custs, so_lc_assign, cust_so_assign)
                
                total_cost_sum = sum(tour['cost_yuan'] for tour in planned_drone_tours)
                total_time_sum = sum(tour['time_hours'] for tour in planned_drone_tours)
                total_distance_sum = sum(tour['distance_km'] for tour in planned_drone_tours)

                print(f"\n--- DRONE Planning Complete ---")
                print(f"Total drone trips: {len(planned_drone_tours)}")
                print(f"Total distance: {total_distance_sum:.2f} km")
                print(f"Total time: {total_time_sum:.2f} hours")
                print(f"Total cost: {total_cost_sum:.2f} yuan")

                output_map_file = "drone_delivery_routes_map.html"   # Changed filename
                output_report_file = "drone_delivery_report.txt" # Changed filename

                generate_text_report(output_report_file, lcs, sos, custs, planned_drone_tours, 
                                     total_cost_sum, total_time_sum, total_distance_sum, delivery_mode="Drone")
                generate_interactive_map(output_map_file, lcs, sos, custs, planned_drone_tours, delivery_mode="Drone")
                
                print(f"\nOutputs generated: {output_map_file}, {output_report_file}")
    else:
        print("No file selected. Exiting.")