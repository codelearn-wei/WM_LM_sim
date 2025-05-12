import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib import rcParams
from scipy.interpolate import interp1d

def load_lane_change_data(lane_change_path):
    """Load lane change data from pickle file"""
    with open(lane_change_path, 'rb') as f:
        return pickle.load(f)

def load_map_data(map_path):
    """Load map data from JSON file"""
    with open(map_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def visualize_merging_points(lane_change_info, map_data, n_segments=10):
    """
    Visualize the distribution of merging points on the map with scientific publication standards
    using equal-length segments to measure density
    
    Parameters:
    -----------
    lane_change_info: dict
        Dictionary containing lane change information
    map_data: dict
        Dictionary containing map information
    n_segments: int, default=20
        Number of equal-length segments to divide the auxiliary lane
    """

    
    # Set scientific publication style
    rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 11,
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.8,
        'axes.grid': True, 
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True
    })
    
    # Create figure with appropriate size for scientific publication
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    
    # Plot map elements
    # Main road
    main_road_x_up = map_data['上沿边界']['x']
    main_road_y_up = map_data['上沿边界']['y']
    ax.plot(main_road_x_up, main_road_y_up, 'k-', linewidth=1.5, label='Main Road')
    
    main_road_x_down = map_data['主道下边界']['x']
    main_road_y_down = map_data['主道下边界']['y']
    ax.plot(main_road_x_down, main_road_y_down, 'k-', linewidth=1.5)
    
    # Auxiliary lane
    aux_lane_x = map_data['辅道虚线']['x']
    aux_lane_y = map_data['辅道虚线']['y']
    ax.plot(aux_lane_x, aux_lane_y, 'k--', linewidth=1, label='Auxiliary Lane')
    
    # Calculate cumulative distance along the auxiliary lane
    aux_points = np.array([aux_lane_x, aux_lane_y]).T
    aux_dists = [0]
    for i in range(1, len(aux_points)):
        segment_length = np.sqrt(np.sum((aux_points[i] - aux_points[i-1])**2))
        aux_dists.append(aux_dists[-1] + segment_length)
    
    total_length = aux_dists[-1]
    
    # Create equally spaced segments along the path
    segment_length = total_length / n_segments
    segment_breaks = np.linspace(0, total_length, n_segments+1)
    
    # Interpolate to get x,y coordinates for each segment boundary
    x_interp = interp1d(aux_dists, aux_lane_x)
    y_interp = interp1d(aux_dists, aux_lane_y)
    
    segment_x = x_interp(segment_breaks)
    segment_y = y_interp(segment_breaks)
    
    # Collect all merging points
    merge_points = []
    for file_id, file_data in lane_change_info.items():
        for scene_id, scene_changes in file_data.items():
            for lane_change in scene_changes:
                cross_point = lane_change['crossing_point']['nearest_point']
                merge_points.append(np.array(cross_point))
    
    merge_points = np.array(merge_points)
    
    # Count merging points in each segment
    segment_counts = np.zeros(n_segments)
    
    for point in merge_points:
        # Find closest point on the auxiliary lane
        distances = np.sqrt(np.sum((aux_points - point)**2, axis=1))
        closest_idx = np.argmin(distances)
        dist_along_path = aux_dists[closest_idx]
        
        # Find which segment this belongs to
        segment_idx = int(dist_along_path / segment_length)
        if segment_idx >= n_segments:
            segment_idx = n_segments - 1
        
        segment_counts[segment_idx] += 1
    
    # Normalize counts for visualization
    max_count = max(segment_counts) if segment_counts.any() else 1
    norm_counts = segment_counts / max_count
    
    # Create colored segments - using a simple red colormap
    cmap = plt.cm.Reds  # 使用内置的Reds颜色映射，从白色到深红色
    
    patches = []
    segment_colors = []
    segment_width = 2  # Width of the density representation
    
    for i in range(n_segments):
        # Calculate direction for this segment
        dx = segment_x[i+1] - segment_x[i]
        dy = segment_y[i+1] - segment_y[i]
        length = np.sqrt(dx**2 + dy**2)
        
        # Normalize direction vector
        if length > 0:
            dx /= length
            dy /= length
        
        # Create perpendicular vector for segment width
        perp_dx = -dy
        perp_dy = dx
        
        # Create polygon points for this segment
        pts = np.array([
            [segment_x[i] + perp_dx * segment_width/2, segment_y[i] + perp_dy * segment_width/2],
            [segment_x[i+1] + perp_dx * segment_width/2, segment_y[i+1] + perp_dy * segment_width/2],
            [segment_x[i+1] - perp_dx * segment_width/2, segment_y[i+1] - perp_dy * segment_width/2],
            [segment_x[i] - perp_dx * segment_width/2, segment_y[i] - perp_dy * segment_width/2]
        ])
        
        polygon = plt.Polygon(pts, closed=True)
        patches.append(polygon)
        segment_colors.append(norm_counts[i])
    
    # Create patch collection with specified colormap
    p = PatchCollection(patches, cmap=cmap, alpha=0.9)
    p.set_array(np.array(segment_colors))
    ax.add_collection(p)
    
    # Add colorbar for density with scientific styling
    cbar = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Merging Density', fontsize=10)
    cbar.ax.tick_params(labelsize=9, width=0.8, length=3)
    cbar.outline.set_linewidth(0.8)
    
    plt.title(f'Equal-Segment Distribution of Lane Merging Density', 
            fontsize=14, 
            pad=10)
    
    plt.xlabel('X Coordinate (m)', fontsize=11)
    plt.ylabel('Y Coordinate (m)', fontsize=11)
    
    # Equal aspect ratio for proper spatial visualization
    ax.set_aspect('equal')
    
    # Add legend in the middle of the plot with no frame
    handles = [
        plt.Line2D([0], [0], color='k', linestyle='-', lw=1.5, label='Main Road'),
        plt.Line2D([0], [0], color='k', linestyle='--', lw=1, label='Auxiliary Lane')
    ]
    legend = ax.legend(handles=handles, frameon=False, fontsize=9, 
                      loc='upper center', bbox_to_anchor=(0.5, 0.98),
                      markerscale=1.2)
    
    # Set tick parameters with scientific styling
    ax.tick_params(axis='both', which='major', labelsize=9, width=0.8, length=4)
    ax.tick_params(axis='both', which='minor', width=0.6, length=2)
    
    # Enable minor ticks
    ax.minorticks_on()
    
    # Tight layout with scientific margins
    plt.tight_layout()
    
    return fig


def analyze_merging_stats(lane_change_info, map_data):
    """
    Analyze statistics of merging points and create a secondary visualization
    with scientific publication standards
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import rcParams
    
    # Set scientific publication style
    rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 11,
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'xtick.major.width': 0.8, 
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True
    })
    
    # Create figure with scientific dimensions
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    # Get auxiliary lane length for reference
    aux_lane_x = map_data['辅道虚线']['x']
    aux_lane_y = map_data['辅道虚线']['y']
    
    # Calculate total length of auxiliary lane
    total_length = 0
    for i in range(len(aux_lane_x)-1):
        segment_length = np.sqrt((aux_lane_x[i+1]-aux_lane_x[i])**2 + 
                               (aux_lane_y[i+1]-aux_lane_y[i])**2)
        total_length += segment_length
    
    # Create bins for merging points along the lane
    num_bins = 10
    bin_edges = np.linspace(0, total_length, num_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate cumulative distance along the lane
    cum_distance = [0]
    for i in range(len(aux_lane_x)-1):
        segment_length = np.sqrt((aux_lane_x[i+1]-aux_lane_x[i])**2 + 
                               (aux_lane_y[i+1]-aux_lane_y[i])**2)
        cum_distance.append(cum_distance[-1] + segment_length)
    
    # Collect merging positions
    merge_positions = []
    
    for file_id, file_data in lane_change_info.items():
        for scene_id, scene_changes in file_data.items():
            for lane_change in scene_changes:
                segment_idx = lane_change['crossing_point']['line_segment_idx']
                if 0 <= segment_idx < len(aux_lane_x)-1:
                    # Calculate exact position along the lane
                    nearest_point = lane_change['crossing_point']['nearest_point']
                    
                    # Find the distance from start of lane to this segment
                    segment_start_dist = cum_distance[segment_idx]
                    
                    # Add distance within this segment
                    point_dist = np.sqrt((nearest_point[0]-aux_lane_x[segment_idx])**2 + 
                                       (nearest_point[1]-aux_lane_y[segment_idx])**2)
                    
                    # Total distance from start of lane
                    total_dist = segment_start_dist + point_dist
                    merge_positions.append(total_dist)
    
    # Create histogram
    counts, bins = np.histogram(merge_positions, bins=bin_edges)
    
    # Plot bars with scientific styling
    width = bin_edges[1] - bin_edges[0]
    bars = ax.bar(bin_centers, counts, width=width*0.8, 
                color='#4575b4', edgecolor='#333333', alpha=0.8, 
                linewidth=0.6)
    
    # Calculate and plot trend line with scientific styling
    hist_x = bin_centers
    hist_y = counts
    z = np.polyfit(hist_x, hist_y, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(bin_centers[0], bin_centers[-1], 100)
    trend_line = ax.plot(x_trend, p(x_trend), 'r-', linewidth=1.5, 
                       label=f'Polynomial Fit (Order 2)')
    
    # Calculate R-squared value
    mean_y = np.mean(hist_y)
    ss_tot = np.sum((hist_y - mean_y) ** 2)
    ss_res = np.sum((hist_y - p(hist_x)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Add scientific annotations
    ax.text(0.05, 0.92, f'$R^2 = {r_squared:.3f}$', 
          transform=ax.transAxes, fontsize=9)
    
    plt.title('Distribution of Merging Points Along Auxiliary Lane', 
            fontsize=12, pad=10)
    
    plt.xlabel('Distance Along Auxiliary Lane (m)', fontsize=11)
    plt.ylabel('Number of Merging Events', fontsize=11)
    
    # Add legend with no frame
    legend = ax.legend(frameon=False, fontsize=9, 
                     loc='best', 
                     labelspacing=0.6)
    
    # Enable minor ticks for scientific appearance
    ax.minorticks_on()
    
    # Set tick parameters with scientific styling
    ax.tick_params(axis='both', which='major', labelsize=9, width=0.8, length=4)
    ax.tick_params(axis='both', which='minor', width=0.6, length=2)
    
    # Tight layout with scientific margins
    plt.tight_layout()
    
    return fig

def main():
    # Define file paths
    data_path = r"data_process\behaviour\data\lane_change_info.pkl"
    map_path = r"LM_data\map\DR_CHN_Merging_ZS.json"
    output_path = r"data_process\behaviour\results\LM_scene"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load data
    print("Loading lane change data...")
    lane_change_info = load_lane_change_data(data_path)
    
    print("Loading map data...")
    map_data = load_map_data(map_path)
    
    # Count total merges
    total_merges = sum(len(scene_changes) 
                       for file_data in lane_change_info.values() 
                       for scene_changes in file_data.values())
    
    print(f"Found {total_merges} merging events. Creating visualizations...")
    
    # Create distribution visualization
    fig1 = visualize_merging_points(lane_change_info, map_data)
    fig1.savefig(os.path.join(output_path, 'merging_points_distribution.png'), 
                bbox_inches='tight', dpi=300)
    fig1.savefig(os.path.join(output_path, 'merging_points_distribution.pdf'), 
                bbox_inches='tight')
    
    # Create statistical analysis visualization
    fig2 = analyze_merging_stats(lane_change_info, map_data)
    fig2.savefig(os.path.join(output_path, 'merging_points_histogram.png'), 
                bbox_inches='tight', dpi=300)
    fig2.savefig(os.path.join(output_path, 'merging_points_histogram.pdf'), 
                bbox_inches='tight')
    
    print(f"Visualizations saved to {output_path}")
    
    # Show plots (optional)
    plt.show()

if __name__ == "__main__":
    main()
    
    
    
    
