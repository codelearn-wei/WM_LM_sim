import numpy as np
from LM_env.utils.CollisionDetect import ColliTest

class Monitor:
    def __init__(self, env):
        """Initialize the monitor with the environment instance."""
        self.env = env

    def get_distance_to_reference_line(self, position):
        """Calculate distance from a position to the reference line."""
        distances = np.sqrt(np.sum((self.env.smooth_reference_line - position) ** 2, axis=1))
        return np.min(distances)

    def check_env_collision(self):
        """
        Check for collisions between environment vehicles.
        
        Returns:
            tuple: (bool, list) - Whether a collision occurred, list of colliding vehicle ID pairs.
        """
        collision_occurred = False
        colliding_pairs = []
        
        for vid1, vehicle1 in self.env.vehicles.items():
            for vid2, vehicle2 in self.env.vehicles.items():
                if vid1 != vid2 and vid1 < vid2:
                    distance = np.linalg.norm(np.array([vehicle1.x, vehicle1.y]) - np.array([vehicle2.x, vehicle2.y]))
                    collision_threshold = (vehicle1.length + vehicle2.length + vehicle1.width + vehicle2.width) / 4.0
                    if distance < collision_threshold:
                        collision_occurred = True
                        colliding_pairs.append((vid1, vid2))
        
        return collision_occurred, colliding_pairs

    def check_ego_collision(self):
        """
        Check if the ego vehicle collides with environment vehicles.
        
        Returns:
            tuple: (bool, list) - Whether ego collided, list of environment vehicle IDs involved.
        """
        if not hasattr(self.env, 'ego_vehicle_id') or self.env.ego_vehicle_id not in self.env.vehicles:
            return False, []
            
        ego_collision = False
        colliding_with_ego = []
        ego_vehicle = self.env.vehicles[self.env.ego_vehicle_id]
        
        for vid, vehicle in self.env.vehicles.items():
            if vid != self.env.ego_vehicle_id:
                is_collision = ColliTest(ego_vehicle, vehicle, ego_vehicle.length, ego_vehicle.width)
                if is_collision:
                    ego_collision = True
                    colliding_with_ego.append(vid)
        
        return ego_collision, colliding_with_ego

    def check_env_off_road(self):
        """
        Check if environment vehicles are off-road.
        
        Returns:
            tuple: (bool, list) - Whether any vehicle is off-road, list of off-road vehicle IDs.
        """
        off_road_occurred = False
        off_road_vehicles = []
        
        for vid, vehicle in self.env.vehicles.items():
            if hasattr(self.env, 'ego_vehicle_id') and vid == self.env.ego_vehicle_id:
                continue
                
            position = [vehicle.x, vehicle.y]
            distance_to_ref = self.get_distance_to_reference_line(position)
            road_width_threshold = 3.0
            if distance_to_ref > road_width_threshold:
                off_road_occurred = True
                off_road_vehicles.append(vid)
        
        return off_road_occurred, off_road_vehicles

    def check_ego_off_road(self):
        """
        Check if the ego vehicle is off-road.
        
        Returns:
            bool: Whether the ego vehicle is off-road.
        """
        if not hasattr(self.env, 'ego_vehicle_id') or self.env.ego_vehicle_id not in self.env.vehicles:
            return False
            
        ego_vehicle = self.env.vehicles[self.env.ego_vehicle_id]
        position = [ego_vehicle.x, ego_vehicle.y]
        distance_to_ref = self.get_distance_to_reference_line(position)
        road_width_threshold = 200
        return distance_to_ref > road_width_threshold

    def check_reach_end(self):
        """
        Check if the ego vehicle has reached the end of the road.
        
        Returns:
            bool: Whether the ego vehicle has reached the end.
        """
        if self.env.ego_vehicle_id is not None:
            ego_vehicle = self.env.vehicles[self.env.ego_vehicle_id]
            if 1032 < ego_vehicle.x < 1035 and 0 < ego_vehicle.y < 2:
                return True
        return False

    def check_environment_status(self):
        """
        Comprehensive environment status check.
        
        Returns:
            dict: Environment status including collisions and off-road conditions.
        """
        env_collision_status, env_collision_pairs = self.check_env_collision()
        ego_collision_status, ego_collision_vehicles = self.check_ego_collision()
        env_off_road_status, env_off_road_ids = self.check_env_off_road()
        ego_off_road_status = self.check_ego_off_road()
        
        return {
            "env_collision": env_collision_status,
            "env_colliding_pairs": env_collision_pairs,
            "ego_collision": ego_collision_status,
            "ego_colliding_with": ego_collision_vehicles,
            "env_off_road": env_off_road_status,
            "env_off_road_ids": env_off_road_ids,
            "ego_off_road": ego_off_road_status
        }

    def check_termination(self):
        """
        Check if the episode should terminate.
        
        Returns:
            bool: Whether the episode should terminate.
        """
        ego_is_collision, _ = self.check_ego_collision()
        ego_off_road = self.check_ego_off_road()
        ego_reach_end = self.check_reach_end()
        return ego_is_collision or ego_off_road or ego_reach_end