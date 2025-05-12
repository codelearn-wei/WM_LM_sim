import pygame
import numpy as np
from scipy.interpolate import splprep, splev

# Global constants (duplicated here for completeness; ideally, these could be in a shared config)
WINDOW_WIDTH = 1300
WINDOW_HEIGHT = 200
ROAD_COLOR = (180, 180, 180)      # Light gray
BOUNDARY_COLOR = (255, 255, 255)  # White
BACKGROUND_COLOR = (30, 30, 30)   # Dark gray

class Renderer:
    def __init__(self, env):
        """Initialize the renderer with the environment instance."""
        self.env = env
        self.screen = None
        self.clock = None
        self.font = None
        if env.render_mode is not None:
            self._setup_rendering()

    def _setup_rendering(self):
        """Set up the Pygame rendering environment."""
        pygame.init()
        
        # Calculate coordinate ranges and scaling
        all_points = np.vstack((self.env.map_dict['upper_boundary'], 
                              self.env.map_dict['main_lower_boundary'], 
                              self.env.reference_line))
        if 'auxiliary_dotted_line' in self.env.map_dict and len(self.env.map_dict['auxiliary_dotted_line']) > 0:
            all_points = np.vstack((all_points, self.env.map_dict['auxiliary_dotted_line']))
        self.min_x, self.min_y = np.min(all_points, axis=0)
        self.max_x, self.max_y = np.max(all_points, axis=0)
        
        padding = 1.0
        self.min_x -= padding
        self.max_x += padding
        self.min_y -= padding
        self.max_y += padding
        
        range_x = self.max_x - self.min_x
        range_y = self.max_y - self.min_y
        self.scale = min(WINDOW_WIDTH / range_x, WINDOW_HEIGHT / range_y)
        self.scale_x = self.scale
        self.scale_y = self.scale
        
        self._prepare_pixel_points()
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Traffic Merge RL Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)

    def _prepare_pixel_points(self):
        """Prepare pixel coordinates for map elements."""
        upper_points = self.env.map_dict['upper_boundary']
        lower_points = self.env.map_dict['main_lower_boundary'][::-1]
        road_points = np.vstack((upper_points, lower_points))
        
        self.road_pixel_points = [self._map_to_pixel(x, y) for x, y in road_points]
        self.upper_pixel_points = [self._map_to_pixel(x, y) for x, y in self.env.map_dict['upper_boundary']]
        self.lower_pixel_points = [self._map_to_pixel(x, y) for x, y in self.env.map_dict['main_lower_boundary']]
        self.reference_pixel_points = [self._map_to_pixel(x, y) for x, y in self.env.smooth_reference_line]
        self.aux_reference_pixel_points = [self._map_to_pixel(x, y) for x, y in self.env.aux_reference_line]
        
        if 'auxiliary_dotted_line' in self.env.map_dict:
            self.auxiliary_dotted_line_pixel_points = [self._map_to_pixel(x, y) for x, y in self.env.map_dict['auxiliary_dotted_line']]
        else:
            self.auxiliary_dotted_line_pixel_points = []

    def _map_to_pixel(self, x, y):
        """Convert map coordinates to pixel coordinates."""
        pixel_x = (x - self.min_x) * self.scale_x
        pixel_y = WINDOW_HEIGHT - (y - self.min_y) * self.scale_y
        return int(pixel_x), int(pixel_y)

    def render_frame(self):
        """Render the current state of the environment."""
        if self.env.render_mode is None:
            return None
        
        if self.screen is None:
            self._setup_rendering()
        
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw road and boundaries
        pygame.draw.polygon(self.screen, ROAD_COLOR, self.road_pixel_points)
        pygame.draw.lines(self.screen, BOUNDARY_COLOR, False, self.upper_pixel_points, 3)
        pygame.draw.lines(self.screen, BOUNDARY_COLOR, False, self.lower_pixel_points, 3)
        
        # Draw reference lines
        pygame.draw.lines(self.screen, (0, 0, 0), False, self.reference_pixel_points, 1)
        pygame.draw.lines(self.screen, (0, 0, 0), False, self.aux_reference_pixel_points, 1)
        
        # Draw dotted line if present
        if self.auxiliary_dotted_line_pixel_points:
            for i in range(0, len(self.auxiliary_dotted_line_pixel_points) - 1, 2):
                if i + 1 < len(self.auxiliary_dotted_line_pixel_points):
                    start_pos = self.auxiliary_dotted_line_pixel_points[i]
                    end_pos = self.auxiliary_dotted_line_pixel_points[i + 1]
                    pygame.draw.line(self.screen, BOUNDARY_COLOR, start_pos, end_pos, 2)
        
        # Draw vehicles
        for vid, vehicle in self.env.vehicles.items():
            center_x = vehicle.x
            center_y = vehicle.y
            heading = vehicle.heading
            length = vehicle.length
            width = vehicle.width
            
            half_length = length / 2
            half_width = width / 2
            local_points = [
                [-half_length, -half_width],
                [-half_length, half_width],
                [half_length, half_width],
                [half_length, -half_width]
            ]
            
            cos_theta = np.cos(heading)
            sin_theta = np.sin(heading)
            rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            
            global_points = []
            for local_x, local_y in local_points:
                rotated_point = np.dot(rotation_matrix, [local_x, local_y])
                global_x = center_x + rotated_point[0]
                global_y = center_y + rotated_point[1]
                global_points.append([global_x, global_y])
            
            pixel_points = [self._map_to_pixel(x, y) for x, y in global_points]
            
            if vid == self.env.ego_vehicle_id:
                color = (0, 0, 255)  # Blue for ego vehicle
                front_center = [center_x + np.cos(heading) * half_length, 
                              center_y + np.sin(heading) * half_length]
                direction_end = [front_center[0] + np.cos(heading) * (length / 2), 
                               front_center[1] + np.sin(heading) * (length / 2)]
                pygame.draw.line(self.screen, (255, 255, 0), 
                               self._map_to_pixel(front_center[0], front_center[1]),
                               self._map_to_pixel(direction_end[0], direction_end[1]), 2)
            else:
                color = (0, 255, 0)  # Green for environment vehicles
            
            pygame.draw.polygon(self.screen, color, pixel_points)
        
        # Draw info text
        info_text = f"Frame: {self.env.current_step} | Vehicles: {len(self.env.vehicles)}"
        info_surface = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(info_surface, (10, 10))
        
        if self.env.ego_vehicle_id in self.env.vehicles:
            ego_vehicle = self.env.vehicles[self.env.ego_vehicle_id]
            ego_speed = np.linalg.norm(ego_vehicle.v)
            ego_heading_deg = np.degrees(ego_vehicle.heading) % 360
            ego_info = f"Ego Speed: {ego_speed:.2f} m/s | Heading: {ego_heading_deg:.1f}°"
            ego_surface = self.font.render(ego_info, True, (0, 255, 255))
            self.screen.blit(ego_surface, (10, 30))
        
        pygame.display.flip()
        self.clock.tick(self.env.metadata["render_fps"])
        
        if self.env.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        """Clean up rendering resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None