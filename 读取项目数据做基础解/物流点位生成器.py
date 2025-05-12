import tkinter as tk
from tkinter import ttk, messagebox
import random
import numpy as np
import folium
from folium.features import DivIcon # 用于自定义HTML标记
import webbrowser
import os
import time
import math

# --- Configuration ---
DEFAULT_CENTER_LAT = 39.9042  # Beijing approximate center latitude
DEFAULT_CENTER_LON = 116.4074 # Beijing approximate center longitude
DEFAULT_GENERATION_RADIUS_KM = 10.0 # Default radius in km
MIN_INTER_POINT_DIST_KM = 0.2   # Minimum distance between any two generated points in km

R_EARTH_KM = 6371.0  # Earth's radius in kilometers

MAP_OUTPUT_DIR = "generated_maps_beijing_custom_icons" # Changed directory name
MAP_FILENAME_TEMPLATE = "locations_map_beijing_custom_{timestamp}.html"

# --- Haversine Distance Calculation ---
def haversine_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_EARTH_KM * c

def get_destination_point(start_lat, start_lon, bearing_deg, distance_km):
    lat1_rad = math.radians(start_lat)
    lon1_rad = math.radians(start_lon)
    bearing_rad = math.radians(bearing_deg)
    d_div_r = distance_km / R_EARTH_KM
    lat2_rad = math.asin(math.sin(lat1_rad) * math.cos(d_div_r) +
                         math.cos(lat1_rad) * math.sin(d_div_r) * math.cos(bearing_rad))
    lon2_rad = lon1_rad + math.atan2(math.sin(bearing_rad) * math.sin(d_div_r) * math.cos(lat1_rad),
                                     math.cos(d_div_r) - math.sin(lat1_rad) * math.sin(lat2_rad))
    return math.degrees(lat2_rad), math.degrees(lon2_rad)

# --- Data Generation Logic (same as before) ---
def generate_points_in_radius(num_points, point_type_name,
                              center_lat, center_lon, radius_km,
                              existing_points_coords, min_dist_km_inter_points, start_id=0):
    newly_generated_points = []
    all_coords_for_check = list(existing_points_coords)
    delta_lat_deg_bound = radius_km / R_EARTH_KM * (180.0 / math.pi)
    cos_center_lat = math.cos(math.radians(center_lat))
    if abs(cos_center_lat) < 1e-6 : cos_center_lat = 1e-6 
    delta_lon_deg_bound = radius_km / (R_EARTH_KM * cos_center_lat) * (180.0 / math.pi)
    lat_min_bound = center_lat - delta_lat_deg_bound
    lat_max_bound = center_lat + delta_lat_deg_bound
    lon_min_bound = center_lon - delta_lon_deg_bound
    lon_max_bound = center_lon + delta_lon_deg_bound

    for i in range(num_points):
        point_id_prefix = {"Logistics Center": "D", "Sales Outlet": "S", "Customer": "C"}.get(point_type_name, "P")
        point_id = f"{point_id_prefix}{start_id + i + 1}"
        placed = False
        for attempt in range(1000):
            cand_lat = random.uniform(lat_min_bound, lat_max_bound)
            cand_lon = random.uniform(lon_min_bound, lon_max_bound)
            candidate_coord = (cand_lat, cand_lon)
            if haversine_distance((center_lat, center_lon), candidate_coord) > radius_km:
                continue
            too_close_to_others = False
            if min_dist_km_inter_points > 0:
                for p_coord in all_coords_for_check:
                    if haversine_distance(p_coord, candidate_coord) < min_dist_km_inter_points:
                        too_close_to_others = True; break
            if not too_close_to_others:
                point_data = {'id': point_id, 'lat': cand_lat, 'lon': cand_lon, 'type': point_type_name}
                newly_generated_points.append(point_data)
                all_coords_for_check.append(candidate_coord)
                placed = True; break
        if not placed:
            for fallback_attempt in range(500):
                angle_deg = random.uniform(0, 360)
                dist_factor = math.sqrt(random.random()) 
                actual_dist_km = radius_km * dist_factor
                cand_lat, cand_lon = get_destination_point(center_lat, center_lon, angle_deg, actual_dist_km)
                candidate_coord = (cand_lat, cand_lon)
                too_close_to_others = False
                if min_dist_km_inter_points > 0:
                    for p_coord in all_coords_for_check:
                        if haversine_distance(p_coord, candidate_coord) < min_dist_km_inter_points:
                            too_close_to_others = True; break
                if not too_close_to_others:
                    point_data = {'id': point_id, 'lat': cand_lat, 'lon': cand_lon, 'type': point_type_name}
                    newly_generated_points.append(point_data)
                    all_coords_for_check.append(candidate_coord)
                    placed = True
                    print(f"Note: Placed {point_type_name} {point_id} using fallback polar generation.")
                    break
        if not placed:
            angle_deg = random.uniform(0, 360); dist_factor = math.sqrt(random.random())
            actual_dist_km = radius_km * dist_factor
            cand_lat, cand_lon = get_destination_point(center_lat, center_lon, angle_deg, actual_dist_km)
            point_data = {'id': point_id, 'lat': cand_lat, 'lon': cand_lon, 'type': point_type_name}
            newly_generated_points.append(point_data)
            all_coords_for_check.append((cand_lat, cand_lon))
            print(f"Warning: Could not place {point_type_name} {point_id} satisfying all distance constraints. Placed within radius.")
    return newly_generated_points

# --- Map Visualization Logic (Updated for Custom Icons and Legend) ---
def create_and_show_map(depots, satellites, customers, center_lat, center_lon, map_tile_choice, filename="locations_map.html"):
    if not depots and not satellites and not customers:
        print("No points to display on the map.")
        return None

    all_points = depots + satellites + customers
    map_view_center = [np.mean([p['lat'] for p in all_points])] if all_points else [center_lat]
    map_view_center.append(np.mean([p['lon'] for p in all_points]) if all_points else center_lon)
    zoom_start = 11 if all_points else 10

    m = folium.Map(location=map_view_center, zoom_start=zoom_start)

    tile_layers = {
        "OpenStreetMap": "openstreetmap",
        "CartoDB Positron": "cartodbpositron"
    }
    
    # Add selected primary tile layer first
    folium.TileLayer(tile_layers.get(map_tile_choice, "openstreetmap"), name=map_tile_choice).add_to(m)
    # Add the other one for LayerControl
    other_tile_name = "CartoDB Positron" if map_tile_choice == "OpenStreetMap" else "OpenStreetMap"
    folium.TileLayer(tile_layers[other_tile_name], name=other_tile_name).add_to(m)

    # --- Custom DivIcon Definitions ---
    # Style for the letters inside icons
    common_text_style = "font-family: Arial, sans-serif; font-size: 13px; color: white; font-weight: bold;"

    # Depot Icon (Blue Square "D")
    depot_html = f"""
    <div style="background-color: #007bff; /* Blue */
                width: 24px; height: 24px; border-radius: 3px;
                display: flex; align-items: center; justify-content: center;
                box-shadow: 1px 1px 3px rgba(0,0,0,0.4);">
        <span style="{common_text_style}">D</span>
    </div>"""

    # Satellite Icon (Orange Diamond "S")
    # The outer div is for positioning, inner for diamond shape and content
    satellite_html = f"""
    <div style="width: 28px; height: 28px; /* Adjusted for perceived size of diamond */
                display: flex; align-items: center; justify-content: center;">
        <div style="width: 20px; height: 20px; background-color: #fd7e14; /* Orange */
                    transform: rotate(45deg);
                    display: flex; align-items: center; justify-content: center;
                    box-shadow: 1px 1px 3px rgba(0,0,0,0.4);">
            <span style="{common_text_style} transform: rotate(-45deg);">S</span>
        </div>
    </div>"""
    
    # Customer Icon (Green Circle "C")
    customer_html = f"""
    <div style="background-color: #28a745; /* Green */
                width: 24px; height: 24px; border-radius: 50%;
                display: flex; align-items: center; justify-content: center;
                box-shadow: 1px 1px 3px rgba(0,0,0,0.4);">
        <span style="{common_text_style}">C</span>
    </div>"""

    point_configs = {
        "Logistics Center": {"html": depot_html, "icon_size": (24,24), "icon_anchor": (12,12)},
        "Sales Outlet": {"html": satellite_html, "icon_size": (28,28), "icon_anchor": (14,14)}, # Anchor center of effective area
        "Customer": {"html": customer_html, "icon_size": (24,24), "icon_anchor": (12,12)}
    }

    for point_list, point_type_key in [(depots, "Logistics Center"), 
                                       (satellites, "Sales Outlet"), 
                                       (customers, "Customer")]:
        config = point_configs[point_type_key]
        for point in point_list:
            div_icon = DivIcon(
                icon_size=config["icon_size"],
                icon_anchor=config["icon_anchor"],
                html=config["html"]
            )
            folium.Marker(
                location=[point['lat'], point['lon']],
                popup=f"<b>{point['type']} {point['id']}</b><br>Lat: {point['lat']:.5f}<br>Lon: {point['lon']:.5f}",
                tooltip=f"{point['type']} {point['id']}",
                icon=div_icon
            ).add_to(m)
            
    # --- Custom Legend ---
    legend_header_style = "font-family: Arial, sans-serif; font-size:16px; font-weight:bold; margin-bottom:5px; text-align:left;"
    legend_item_style = "font-family: Arial, sans-serif; font-size:13px; display:flex; align-items:center; margin-bottom:4px;"
    legend_icon_common_style = "color:white; display:flex; align-items:center; justify-content:center; font-weight:bold; margin-right:8px;"
    
    legend_html = f"""
     <div style="position: fixed; 
                 bottom: 30px; left: 10px; width: auto; min-width:180px; height: auto; 
                 border:1px solid grey; z-index:9999; font-size:14px;
                 background-color:rgba(255,255,255,0.9); padding: 8px; border-radius:5px;">
       <div style="{legend_header_style}">&nbsp;图例 (Legend)</div>
       <div style="{legend_item_style}">
         <div style="width:20px; height:20px; background-color:#007bff; border-radius:3px; {legend_icon_common_style}">D</div>
         物流中心 (Depot)
       </div>
       <div style="{legend_item_style}">
         <div style="width:23px; height:23px; display:flex; align-items:center; justify-content:center; margin-right:8px;">
            <div style="width:15px; height:15px; background-color:#fd7e14; transform:rotate(45deg); {legend_icon_common_style} position:relative;">
                <span style="transform:rotate(-45deg); font-size:11px;">S</span>
            </div>
         </div>
         销售网点 (Satellite)
       </div>
       <div style="{legend_item_style}">
         <div style="width:20px; height:20px; background-color:#28a745; border-radius:50%; {legend_icon_common_style}">C</div>
         客户 (Customer)
       </div>
     </div>
    """
    m.get_root().add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    if not os.path.exists(MAP_OUTPUT_DIR):
        os.makedirs(MAP_OUTPUT_DIR)
    
    filepath = os.path.join(MAP_OUTPUT_DIR, filename)
    m.save(filepath)
    print(f"Map saved to: {os.path.abspath(filepath)}")
    
    try:
        webbrowser.open(f"file://{os.path.abspath(filepath)}")
    except Exception as e:
        print(f"Could not open map in browser: {e}")
    return filepath

# --- GUI Application (mostly same as before) ---
class LocationGeneratorApp:
    def __init__(self, root_window):
        self.root = root_window
        root_window.title("物流点位生成器（自定义图标）")
        mainframe = ttk.Frame(root_window, padding="10 10 10 10")
        mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        row_idx = 0
        ttk.Label(mainframe, text="中心纬度:").grid(column=0, row=row_idx, sticky=tk.W, pady=2)
        self.center_lat_var = tk.StringVar(value=str(DEFAULT_CENTER_LAT))
        ttk.Entry(mainframe, width=12, textvariable=self.center_lat_var).grid(column=1, row=row_idx, sticky=(tk.W, tk.E), pady=2)
        row_idx += 1
        ttk.Label(mainframe, text="中心经度:").grid(column=0, row=row_idx, sticky=tk.W, pady=2)
        self.center_lon_var = tk.StringVar(value=str(DEFAULT_CENTER_LON))
        ttk.Entry(mainframe, width=12, textvariable=self.center_lon_var).grid(column=1, row=row_idx, sticky=(tk.W, tk.E), pady=2)
        row_idx += 1
        ttk.Label(mainframe, text="生成半径 (公里):").grid(column=0, row=row_idx, sticky=tk.W, pady=2)
        self.radius_km_var = tk.StringVar(value=str(DEFAULT_GENERATION_RADIUS_KM))
        ttk.Entry(mainframe, width=12, textvariable=self.radius_km_var).grid(column=1, row=row_idx, sticky=(tk.W, tk.E), pady=2)
        row_idx += 1
        ttk.Label(mainframe, text=f"最小点间距 (公里):").grid(column=0, row=row_idx, sticky=tk.W, pady=2)
        ttk.Label(mainframe, text=str(MIN_INTER_POINT_DIST_KM)).grid(column=1, row=row_idx, sticky=tk.W, pady=2)
        row_idx +=1
        ttk.Label(mainframe, text="物流中心数量:").grid(column=0, row=row_idx, sticky=tk.W, pady=2)
        self.num_depots_var = tk.StringVar(value="2")
        ttk.Entry(mainframe, width=12, textvariable=self.num_depots_var).grid(column=1, row=row_idx, sticky=(tk.W, tk.E), pady=2)
        row_idx += 1
        ttk.Label(mainframe, text="销售网点数量:").grid(column=0, row=row_idx, sticky=tk.W, pady=2)
        self.num_satellites_var = tk.StringVar(value="5")
        ttk.Entry(mainframe, width=12, textvariable=self.num_satellites_var).grid(column=1, row=row_idx, sticky=(tk.W, tk.E), pady=2)
        row_idx += 1
        ttk.Label(mainframe, text="客户点数量:").grid(column=0, row=row_idx, sticky=tk.W, pady=2)
        self.num_customers_var = tk.StringVar(value="20")
        ttk.Entry(mainframe, width=12, textvariable=self.num_customers_var).grid(column=1, row=row_idx, sticky=(tk.W, tk.E), pady=2)
        row_idx += 1
        ttk.Label(mainframe, text="地图样式:").grid(column=0, row=row_idx, sticky=tk.W, pady=2)
        self.map_tile_var = tk.StringVar(value="OpenStreetMap")
        map_tile_options = ["OpenStreetMap", "CartoDB Positron"]
        ttk.Combobox(mainframe, textvariable=self.map_tile_var, values=map_tile_options, state="readonly", width=18).grid(column=1, row=row_idx, sticky=(tk.W, tk.E), pady=2)
        row_idx += 1
        ttk.Button(mainframe, text="生成并显示地图", command=self.generate_and_display).grid(column=0, row=row_idx, columnspan=2, pady=10)
        row_idx += 1
        self.status_var = tk.StringVar()
        ttk.Label(mainframe, textvariable=self.status_var).grid(column=0, row=row_idx, columnspan=2, sticky=tk.W, pady=5)
        for child in mainframe.winfo_children(): child.grid_configure(padx=5)
        root_window.columnconfigure(0, weight=1); root_window.rowconfigure(0, weight=1)
        mainframe.columnconfigure(1, weight=1)

    def generate_and_display(self):
        try:
            center_lat = float(self.center_lat_var.get())
            center_lon = float(self.center_lon_var.get())
            radius_km = float(self.radius_km_var.get())
            num_depots = int(self.num_depots_var.get())
            num_satellites = int(self.num_satellites_var.get())
            num_customers = int(self.num_customers_var.get())
            map_tile_choice = self.map_tile_var.get()

            if radius_km <= 0: messagebox.showerror("输入错误", "生成半径必须大于0。"); return
            if num_depots < 0 or num_satellites < 0 or num_customers < 0: messagebox.showerror("输入错误", "点位数量不能为负数。"); return

            self.status_var.set("正在生成点位..."); self.root.update_idletasks()
            all_generated_coords = [] 
            depots = generate_points_in_radius(num_depots, "Logistics Center", center_lat, center_lon, radius_km, all_generated_coords, MIN_INTER_POINT_DIST_KM, start_id=0)
            for p in depots: all_generated_coords.append((p['lat'], p['lon']))
            satellites = generate_points_in_radius(num_satellites, "Sales Outlet", center_lat, center_lon, radius_km, all_generated_coords, MIN_INTER_POINT_DIST_KM, start_id=0)
            for p in satellites: all_generated_coords.append((p['lat'], p['lon']))
            customers = generate_points_in_radius(num_customers, "Customer", center_lat, center_lon, radius_km, all_generated_coords, MIN_INTER_POINT_DIST_KM, start_id=0)
            
            self.status_var.set("正在生成地图..."); self.root.update_idletasks()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            map_filename = MAP_FILENAME_TEMPLATE.format(timestamp=timestamp)
            filepath = create_and_show_map(depots, satellites, customers, center_lat, center_lon, map_tile_choice, filename=map_filename)
            self.status_var.set(f"地图已生成: {os.path.basename(filepath)}" if filepath else "地图生成失败或无点位可显示。")
        except ValueError: messagebox.showerror("输入错误", "请输入有效的数字。"); self.status_var.set("输入错误，请检查参数。")
        except Exception as e: messagebox.showerror("发生错误", str(e)); self.status_var.set(f"错误: {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__":
    try: import folium
    except ImportError as e: print(f"错误：缺少库: {e}. 请运行: pip install folium numpy"); exit(1)
    root = tk.Tk()
    app = LocationGeneratorApp(root)
    root.mainloop()