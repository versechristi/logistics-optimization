import re
import tkinter as tk
from tkinter import filedialog

def extract_detailed_map_data(file_path):
    """
    Reads an HTML file and extracts coordinates, types (Logistics Center, 
    Sales Outlet, Customer), and demand for customer points.

    Args:
        file_path (str): The path to the HTML file.

    Returns:
        dict: A dictionary where keys are point types (str) and values are lists 
              of dictionaries. Each dictionary contains point details.
              For Customers: {'latitude': float, 'longitude': float, 'demand': float}
              For Others:    {'latitude': float, 'longitude': float}
              Returns None if the file is not found or an error occurs.
    """
    points_data = {
        "Logistics Center": [],
        "Sales Outlet": [],
        "Customer": [],
        "Unknown": []
    }

    if not file_path:
        print("No file selected.")
        return None
        
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        # Pattern to isolate blocks of JavaScript for each marker.
        # It looks for a marker definition and captures everything until the next marker definition
        # or known end-of-marker-related script sections.
        marker_blocks_pattern = re.compile(
            # Start of a marker definition with coordinates
            r'(var\s+(marker_\w+)\s*=\s*L\.marker\(\s*\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]\s*,\s*\{\}\s*\)\s*\.addTo\([^)]+\);'
            # Capture all associated JS code for this marker (non-greedy)
            r'[\s\S]*?'
            # Lookahead to ensure we stop before the next marker definition or a clear boundary
            r'(?=\s*var\s+marker_\w+\s*=\s*L\.marker|\s*var\s+layer_control_|\s*<\/script>))'
        )
        
        # Patterns to find details within each marker's block
        icon_pattern = re.compile(r'L\.AwesomeMarkers\.icon\([^)]*?"icon":\s*"([^"]+)"')
        demand_popup_pattern = re.compile(r'Customer \d+<br>Demand:\s*(-?\d+\.\d+)')
        demand_tooltip_pattern = re.compile(r'Customer \d+ \(Demand:\s*(-?\d+\.\d+)\)')

        for match in marker_blocks_pattern.finditer(html_content):
            marker_block_content = match.group(1) # The full block for this marker
            # marker_var_name = match.group(2) # e.g., marker_2f62665dc3a96194101c930dbc9052dc
            lat_str = match.group(3)
            lon_str = match.group(4)

            try:
                lat = float(lat_str)
                lon = float(lon_str)
                
                icon_match = icon_pattern.search(marker_block_content)
                icon_type_raw = "unknown"
                if icon_match:
                    icon_type_raw = icon_match.group(1)

                point_type = "Unknown"
                point_details = {'latitude': lat, 'longitude': lon}

                if icon_type_raw == "home":
                    point_type = "Logistics Center"
                elif icon_type_raw == "store":
                    point_type = "Sales Outlet"
                elif icon_type_raw == "user":
                    point_type = "Customer"
                    demand = None
                    # Try to find demand in popup content first
                    demand_match_popup = demand_popup_pattern.search(marker_block_content)
                    if demand_match_popup:
                        demand = float(demand_match_popup.group(1))
                    else:
                        # If not in popup, try tooltip content
                        demand_match_tooltip = demand_tooltip_pattern.search(marker_block_content)
                        if demand_match_tooltip:
                            demand = float(demand_match_tooltip.group(1))
                    
                    if demand is not None:
                        point_details['demand'] = demand
                    else:
                        point_details['demand'] = 'Not Found' # Or None, if preferred

                points_data.setdefault(point_type, []).append(point_details)

            except ValueError as ve:
                print(f"Warning: Could not convert data for a marker. Lat: {lat_str}, Lon: {lon_str}. Error: {ve}. Skipping.")
            except Exception as e_inner:
                print(f"Warning: Error processing a marker block: {e_inner}. Skipping.")
                    
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        return None
        
    return points_data

def select_file():
    """
    Opens a file dialog for the user to select an HTML file.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an HTML map file",
        filetypes=(("HTML files", "*.html;*.htm"), ("All files", "*.*"))
    )
    return file_path

if __name__ == "__main__":
    html_file_path = select_file()
    
    if html_file_path:
        extracted_data = extract_detailed_map_data(html_file_path)
        
        if extracted_data:
            print(f"\nSuccessfully extracted data from: {html_file_path}\n")
            total_points_found = 0
            for point_type, data_list in extracted_data.items():
                if data_list:
                    total_points_found += len(data_list)
                    print(f"--- {point_type} ({len(data_list)}) ---")
                    for i, details in enumerate(data_list):
                        coord_info = f"  Point {i+1}: Latitude = {details['latitude']}, Longitude = {details['longitude']}"
                        if point_type == "Customer" and 'demand' in details:
                            coord_info += f", Demand = {details['demand']}"
                        print(coord_info)
                    print("") 
            
            if total_points_found == 0:
                print("No data matching the expected patterns was found in the file.")
        else:
            print("Data extraction failed or returned no data.")
    else:
        print("File selection cancelled.")