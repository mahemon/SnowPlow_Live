import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import folium

def plot_map(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    print(f"Processing file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    with open(file_path, 'r') as f:
        data = f.readlines() 
    print(f"Loaded {len(data)} lines from the file.")

    file_path = os.path.join(folder_path, file_name)
    plot_folder = os.path.join(folder_path, "plots")
    data_rows = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            values = line.split('|')
            
            latitude = float(values[0])
            longitude = float(values[1])
            timestamp = int(values[2])
            total_lane = int(values[3])
            current_lane = int(values[4])
            
            lane_width_values = list(map(float, values[5].split(':')))
            if len(lane_width_values) == 1:
                lane_width = lane_width_values[0]
            else:
                lane_width = sum(lane_width_values) / len(lane_width_values)
            
            deviation = float(values[6])
            departure_warning = int(values[7])
            
            object_detection_warning = list(map(int, values[8].split(';')))
            left_obj_warning, red_obj_count, yellow_obj_count = object_detection_warning
            
            lane_dep_visual_warning = int(values[9])
            lane_dep_audio_warning = int(values[10])
            object_detection_visual_warning = int(values[11])
            object_detection_audio_warning = int(values[12])
            shoulder_left_type = int(values[13])
            shoulder_left_width_ft = float(values[14])
            shoulder_right_type = int(values[15])
            shoulder_right_width_ft = float(values[16])
            map_available = int(values[17])
            speed = int(values[18])

            data_rows.append({
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": timestamp,
                "total_lane": total_lane,
                "current_lane": current_lane,
                "lane_width": lane_width,
                "deviation": deviation,
                "departure_warning": departure_warning,
                "left_obj_warning": left_obj_warning,
                "red_obj_count": red_obj_count,
                "yellow_obj_count": yellow_obj_count,
                "lane_dep_visual_warning": lane_dep_visual_warning,
                "lane_dep_audio_warning": lane_dep_audio_warning,
                "object_detection_visual_warning": object_detection_visual_warning,
                "object_detection_audio_warning": object_detection_audio_warning,
                "shoulder_left_type": shoulder_left_type,
                "shoulder_left_width_ft": shoulder_left_width_ft,
                "shoulder_right_type": shoulder_right_type,
                "shoulder_right_width_ft": shoulder_right_width_ft,
                "map_available": map_available,
                "speed": speed
            })

    df = pd.DataFrame(data_rows)
    df.to_csv(f'{folder_path}/parse.csv', index=False)

    deviation_threshold_min = -2.5
    deviation_threshold_max =  3.5

    outside_deviation = df[(df['deviation'] < deviation_threshold_min) | (df['deviation'] > deviation_threshold_max)]
    outside_count = outside_deviation.shape[0]

    plt.figure(figsize=(10, 6))
    plt.hist(df['deviation'], bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(deviation_threshold_min, color='r', linestyle='--', label='Left Threshold')
    plt.axvline(deviation_threshold_max, color='g', linestyle='--', label='Right Threshold')
    plt.title('Distribution of Lane Deviation distance (ft)')
    plt.xlabel('Deviation')
    plt.ylabel('Frequenc)')
    plt.legend()
    plt.savefig(f'{plot_folder}/distribution_of_deviation.png')

    red_only = df[(df['red_obj_count'] > 0) & (df['yellow_obj_count'] == 0)]
    yellow_only = df[(df['yellow_obj_count'] > 0) & (df['red_obj_count'] == 0)]
    both_red_yellow = df[(df['red_obj_count'] > 0) & (df['yellow_obj_count'] > 0)]

    red_count = red_only.shape[0]
    yellow_count = yellow_only.shape[0]
    both_count = both_red_yellow.shape[0]

    non_zero_df = df[(df['red_obj_count'] > 0) | (df['yellow_obj_count'] > 0)]

    plt.figure(figsize=(10, 6))

    plt.hist(non_zero_df['red_obj_count'], bins=range(1, int(non_zero_df['red_obj_count'].max()) + 2), 
            color='red', alpha=0.7, label='Red Zone Objects')

    # Plot yellow object counts
    plt.hist(non_zero_df['yellow_obj_count'], bins=range(1, int(non_zero_df['yellow_obj_count'].max()) + 2), 
            color='yellow', alpha=0.7, label='Yellow Zone Objects')

    plt.title('Distribution of Red and Yellow Zone Radar Object Counts')
    plt.xlabel('Radar Object Count')
    plt.ylabel('Frequency')
    plt.xticks(range(1, int(max(non_zero_df['red_obj_count'].max(), non_zero_df['yellow_obj_count'].max())) + 1))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{plot_folder}/filtered_distribution_of_red_yellow_obj_count.png')


    # Bar Chart for Object Counts in Different Zones
    counts = {
        'Only Red Zone': non_zero_df[(non_zero_df['red_obj_count'] > 0) & (non_zero_df['yellow_obj_count'] == 0)].shape[0],
        'Only Yellow Zone ': non_zero_df[(non_zero_df['yellow_obj_count'] > 0) & (non_zero_df['red_obj_count'] == 0)].shape[0],
        'Red and Yellow Zone': non_zero_df[(non_zero_df['red_obj_count'] > 0) & (non_zero_df['yellow_obj_count'] > 0)].shape[0],
    }

    plt.figure(figsize=(8, 6))
    plt.bar(counts.keys(), counts.values(), color=['red', 'yellow', 'orange'], edgecolor='black', alpha=0.7)
    plt.title('Counts of Radar Objects in Zones')
    plt.xlabel('Zone Type')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{plot_folder}/filtered_counts_red_yellow_obj.png')


    # Create Radar Map
    radar_map = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=15)

    # Add markers for radar data
    for _, row in df.iterrows():
        if row['red_obj_count'] > 0 and row['yellow_obj_count'] == 0:
            # Only red objects
            color = 'red'
            popup_info = f"Red Zone Objects: {row['red_obj_count']}"
        elif row['yellow_obj_count'] > 0 and row['red_obj_count'] == 0:
            # Only yellow objects
            color = 'lightgreen'
            popup_info = f"Yellow Zone Objects: {row['yellow_obj_count']}"
        elif row['red_obj_count'] > 0 and row['yellow_obj_count'] > 0:
            # Both red and yellow objects
            color = 'blue'
            popup_info = f"Red Zone: {row['red_obj_count']}, Yellow Zone: {row['yellow_obj_count']}"
        else:
            continue  # Skip rows with no objects

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup_info,
            icon=folium.Icon(color=color)
        ).add_to(radar_map)

    # Save the map
    radar_map.save(f"{plot_folder}/radar_map_updated.html")


    # Create Deviation Map
    deviation_map = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=15)

    # Add markers for deviation data
    for _, row in outside_deviation.iterrows():
        if row['deviation'] > deviation_threshold_max:
            # Positive deviation (above maximum threshold)
            color = 'gray'
            popup_info = f"Deviation: {row['deviation']} (Above Max Threshold)"
        elif row['deviation'] < deviation_threshold_min:
            # Negative deviation (below minimum threshold)
            color = 'red'
            popup_info = f"Deviation: {row['deviation']} (Below Min Threshold)"
        else:
            continue  # This should not happen as we filtered outside deviations only

        # Add marker with explicit color
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup_info,
            icon=folium.Icon(color=color)
        ).add_to(deviation_map)

    # Save the map
    deviation_map.save(f"{plot_folder}/deviation_map_updated.html")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process folder and file path for the script.")
    parser.add_argument(
        "--folder", 
        type=str, 
        default="/Users/anwar/Documents/GitHub/snowplow_dec13/ui_string", 
        help=f"Path to the folder containing the file."
    )
    parser.add_argument(
        "--file", 
        type=str, 
        default="2024_12_12_13_44_16_436.txt", 
        help=f"Name of the file to process."
    )
    args = parser.parse_args()
    plot_map(folder_path=args.folder, file_name=args.file)

