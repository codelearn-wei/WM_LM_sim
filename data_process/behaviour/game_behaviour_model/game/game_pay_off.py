from typing import Dict
import numpy as np
import math

class PayoffCalculator:
    def __init__(self, params: Dict):
        self.params = params
        self.merge_point = params.get('merge_point', (0, 0))

    def assess_collision_risk(self, f_future, z_future, payoff_info):
        """
        Assess collision risk between two vehicles considering their dimensions
        
        Parameters:
        - f_future: Future trajectory points for FV vehicle [(x, y, heading), ...]
        - z_future: Future trajectory points for ZV vehicle [(x, y, heading), ...]
        - fv_vehicle: FV vehicle object with dimensions (optional)
        - zv_vehicle: ZV vehicle object with dimensions (optional)
        
        Returns:
        - Risk score (0-1), where 0 means high risk of collision and 1 means safe
        """
        # Define default vehicle dimensions if not provided
        fv_length = getattr(payoff_info['FV']["length"], 'length', 4.5)  # meters
        fv_width = getattr(payoff_info['FV']["width"], 'width', 1.8)    # meters
        zv_length = getattr(payoff_info['ZV']["length"], 'length', 4.5)  # meters
        zv_width = getattr(payoff_info['ZV']["width"], 'width', 1.8)    # meters
        
        # Calculate safety thresholds based on vehicle dimensions
        # Adding buffer zones around vehicles
        buffer_distance = 0.5  # additional safety buffer in meters
        
        lateral_safety_threshold = (fv_width/2 + zv_width/2) + buffer_distance
        longitudinal_safety_threshold = (fv_length/2 + zv_length/2) + buffer_distance
        
        # Track the highest risk encountered during trajectory analysis
        max_risk_factor = 0.0
        
        # Check point-to-point proximity risk
        for i in range(min(len(f_future), len(z_future))):
            f_pos = f_future[i]
            z_pos = z_future[i]
            
            # Calculate distances between vehicle centers
            lateral_distance = abs(f_pos[1] - z_pos[1])
            longitudinal_distance = abs(f_pos[0] - z_pos[0])
            
            # For precise collision detection, consider vehicle orientation
            if len(f_pos) > 2 and len(z_pos) > 2:  # If heading information is available
                f_heading = f_pos[2]
                z_heading = z_pos[2]
                
                # Adjust safety thresholds based on vehicle orientations
                # When vehicles are perpendicular, lateral and longitudinal thresholds are different
                heading_diff = abs((f_heading - z_heading) % (2 * math.pi))
                perpendicular_factor = abs(math.sin(heading_diff))
                
                effective_lateral_threshold = lateral_safety_threshold * (1 - perpendicular_factor) + longitudinal_safety_threshold * perpendicular_factor
                effective_long_threshold = longitudinal_safety_threshold * (1 - perpendicular_factor) + lateral_safety_threshold * perpendicular_factor
            else:
                effective_lateral_threshold = lateral_safety_threshold
                effective_long_threshold = longitudinal_safety_threshold
            
            # Calculate risk based on distances
            # Direct collision risk - vehicles overlapping
            if lateral_distance < effective_lateral_threshold and longitudinal_distance < effective_long_threshold:
                lateral_overlap = 1 - lateral_distance / effective_lateral_threshold if effective_lateral_threshold > 0 else 1
                longitudinal_overlap = 1 - longitudinal_distance / effective_long_threshold if effective_long_threshold > 0 else 1
                # Higher value for greater overlap
                collision_risk = 0.7 + 0.3 * (lateral_overlap * longitudinal_overlap)
            # Near-miss risk
            elif lateral_distance < 2 * effective_lateral_threshold and longitudinal_distance < 2 * effective_long_threshold:
                # Scale down risk for near misses
                lateral_factor = max(0, 1 - lateral_distance / (2 * effective_lateral_threshold))
                longitudinal_factor = max(0, 1 - longitudinal_distance / (2 * effective_long_threshold))
                collision_risk = 0.3 * (lateral_factor * longitudinal_factor)
            else:
                # Safe distance
                collision_risk = 0.0
            
            max_risk_factor = max(max_risk_factor, collision_risk)
        
        # Check for trajectory segment intersections
        # This catches cases where vehicles might cross paths between sampled points
        for i in range(len(f_future)-1):
            for j in range(len(z_future)-1):
                if self.segments_intersect(f_future[i][0:2], f_future[i+1][0:2], 
                                        z_future[j][0:2], z_future[j+1][0:2]):
                    # Check time overlap to see if they'd be there at the same time
                    if len(f_future[i]) > 3 and len(z_future[j]) > 3:  # If time information is available
                        f_t1, f_t2 = f_future[i][3], f_future[i+1][3]
                        z_t1, z_t2 = z_future[j][3], z_future[j+1][3]
                        
                        # Check if time ranges overlap
                        if max(f_t1, z_t1) <= min(f_t2, z_t2):
                            # Time overlap exists, indicating higher collision probability
                            intersection_risk = 0.85
                        else:
                            # Paths cross but at different times
                            time_diff = min(abs(f_t1 - z_t2), abs(f_t2 - z_t1))
                            # Lower risk for greater time separation
                            intersection_risk = 0.6 * max(0, 1 - time_diff/2.0)
                    else:
                        # Without time info, assume higher risk for intersecting paths
                        intersection_risk = 0.75
                    
                    max_risk_factor = max(max_risk_factor, intersection_risk)
        
        # Analyze relative velocities for additional risk assessment
        if len(f_future) > 1 and len(z_future) > 1:
            # Calculate velocities from consecutive points
            f_vel_x = (f_future[1][0] - f_future[0][0]) / (f_future[1][3] - f_future[0][3]) if len(f_future[0]) > 3 else 0
            f_vel_y = (f_future[1][1] - f_future[0][1]) / (f_future[1][3] - f_future[0][3]) if len(f_future[0]) > 3 else 0
            z_vel_x = (z_future[1][0] - z_future[0][0]) / (z_future[1][3] - z_future[0][3]) if len(z_future[0]) > 3 else 0
            z_vel_y = (z_future[1][1] - z_future[0][1]) / (z_future[1][3] - z_future[0][3]) if len(z_future[0]) > 3 else 0
            
            # Calculate relative velocity magnitude
            rel_vel = math.sqrt((f_vel_x - z_vel_x)**2 + (f_vel_y - z_vel_y)**2)
            
            # Higher relative velocities increase risk if vehicles are close
            if max_risk_factor > 0.3 and rel_vel > 5.0:  # 5 m/s threshold
                velocity_risk_factor = min(0.2, (rel_vel - 5.0) / 20.0)  # Scales up to 0.2 for very high relative velocities
                max_risk_factor = min(0.95, max_risk_factor + velocity_risk_factor)
        
        # Convert risk factor to safety score (1 - risk)
        max_risk_factor
        
        return max_risk_factor

    def segments_intersect(self, p1, p2, p3, p4):
        """
        Check if line segments (p1,p2) and (p3,p4) intersect
        Each point is a tuple or list (x, y)
        """
        # Helper function for orientation of triplet (p, q, r)
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-10:  # Check for collinearity with small epsilon for floating point
                return 0
            return 1 if val > 0 else 2
        
        # Helper function to check if point q lies on segment pr
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        # Get orientations needed for general and special cases
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        # General case: different orientations for points
        if o1 != o2 and o3 != o4:
            return True
        
        # Special Cases: collinear points
        if o1 == 0 and on_segment(p1, p3, p2): return True
        if o2 == 0 and on_segment(p1, p4, p2): return True
        if o3 == 0 and on_segment(p3, p1, p4): return True
        if o4 == 0 and on_segment(p3, p2, p4): return True
        
        # No intersection
        return False

    def calculate_oriented_bounding_box(self, position, heading, length, width):
        """
        Calculate the four corners of an oriented bounding box
        
        Parameters:
        - position: (x, y) center position
        - heading: orientation angle in radians
        - length: vehicle length
        - width: vehicle width
        
        Returns:
        - List of (x, y) coordinates for the four corners
        """
        x, y = position
        half_length = length / 2
        half_width = width / 2
        
        # Calculate corner offsets
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        
        # Calculate the four corners (front-left, front-right, rear-right, rear-left)
        corners = [
            (x + half_length * cos_h - half_width * sin_h, 
            y + half_length * sin_h + half_width * cos_h),
            (x + half_length * cos_h + half_width * sin_h, 
            y + half_length * sin_h - half_width * cos_h),
            (x - half_length * cos_h + half_width * sin_h, 
            y - half_length * sin_h - half_width * cos_h),
            (x - half_length * cos_h - half_width * sin_h, 
            y - half_length * sin_h + half_width * cos_h)
        ]
        
        return corners

    def check_polygon_intersection(self, poly1, poly2):
        """
        Check if two convex polygons (defined by lists of vertices) intersect
        Uses the Separating Axis Theorem (SAT)
        
        Parameters:
        - poly1, poly2: Lists of (x, y) points representing polygon vertices
        
        Returns:
        - True if polygons intersect, False otherwise
        """
        # Helper function to project a polygon onto an axis
        def project_polygon(poly, axis):
            min_proj = float('inf')
            max_proj = float('-inf')
            
            for point in poly:
                projection = point[0] * axis[0] + point[1] * axis[1]
                min_proj = min(min_proj, projection)
                max_proj = max(max_proj, projection)
                
            return min_proj, max_proj
        
        # Get all edges for both polygons
        edges = []
        for i in range(len(poly1)):
            edges.append((poly1[i], poly1[(i+1) % len(poly1)]))
        for i in range(len(poly2)):
            edges.append((poly2[i], poly2[(i+1) % len(poly2)]))
        
        # Check each edge's normal as a potential separating axis
        for edge in edges:
            # Calculate the edge normal (perpendicular)
            edge_vec = (edge[1][0] - edge[0][0], edge[1][1] - edge[0][1])
            normal = (-edge_vec[1], edge_vec[0])  # 90 degree rotation
            
            # Normalize the normal
            length = math.sqrt(normal[0]**2 + normal[1]**2)
            if length < 1e-10:  # Avoid division by zero
                continue
            normal = (normal[0] / length, normal[1] / length)
            
            # Project both polygons onto this axis
            proj1 = project_polygon(poly1, normal)
            proj2 = project_polygon(poly2, normal)
            
            # Check if projections overlap
            if proj1[1] < proj2[0] or proj2[1] < proj1[0]:
                # Found a separating axis, so polygons don't intersect
                return False
        
        # No separating axis found, so polygons must intersect
        return True

    def precise_collision_check(self, f_position, f_heading, z_position, z_heading, fv_length=4.5, fv_width=1.8, zv_length=4.5, zv_width=1.8):
        """
        Perform a precise collision check between two vehicles using oriented bounding boxes
        
        Parameters:
        - f_position: (x, y) position of first vehicle
        - f_heading: heading angle of first vehicle (radians)
        - z_position: (x, y) position of second vehicle
        - z_heading: heading angle of second vehicle (radians)
        - fv_length, fv_width: dimensions of first vehicle
        - zv_length, zv_width: dimensions of second vehicle
        
        Returns:
        - True if collision detected, False otherwise
        """
        # Calculate oriented bounding boxes
        fv_box = self.calculate_oriented_bounding_box(f_position, f_heading, fv_length, fv_width)
        zv_box = self.calculate_oriented_bounding_box(z_position, z_heading, zv_length, zv_width)
        
        # Check for polygon intersection using SAT (Separating Axis Theorem)
        return self.check_polygon_intersection(fv_box, zv_box)

    def calculate(self, fv, zv, f_future, z_future, pay_off_info):
        # Calculate main road vehicle (ZV) payoff
        zv_payoff = self._calculate_zv_payoff(f_future, z_future, pay_off_info)
        
        # Calculate ramp vehicle (FV) payoff
        fv_payoff = self._calculate_fv_payoff(f_future, z_future, pay_off_info)
        
        return zv_payoff, fv_payoff
    
    def _calculate_zv_payoff(self, f_future, z_future, pay_off_info):
        # Get parameters
        comfort = pay_off_info['ZV']['a_ZV']
        comfort_weight = self.params.get('comfort_weight', 0.15)
        
        # Speed component
        target_zv_v = self.params.get('target_zv_v', 30.0)
        # 应该使用1s之后的速度
        # zv_v = pay_off_info['ZV']['v_ZV']
        zv_v = z_future[9][3]
        speed_weight = self.params.get('speed_weight', 0.20)
        speed_diff_penalty = -abs(zv_v - target_zv_v) / max(1.0, target_zv_v)
        speed_component = speed_weight * speed_diff_penalty
        
        # Distance component
        target_zv_2_lzv_d = self.params.get('target_zv_2_lzv_d', 10.0)
        # 应该使用1s之后的距离
        zv_2_lzv_d = pay_off_info['ZV']['ZV_dis_2_LZV']
        distance_weight = self.params.get('distance_weight', 0.15)
        distance_diff_penalty = -abs(zv_2_lzv_d - target_zv_2_lzv_d) / max(1.0, target_zv_2_lzv_d)
        distance_component = distance_weight * distance_diff_penalty
        
        # Time headway component (ZV to FV)
        target_thw_zv_2_fv = self.params.get('target_thw_zv_2_fv', 1.0)
        
        # thw_zv_2_fv = pay_off_info['ZV']['thw_ZV_2_FV']
        # 应该使用1s之后的车头时距
        thw_zv_2_fv = abs(z_future[9][0] - f_future[9][0])/z_future[0][3]
        thw_weight = self.params.get('thw_weight', 0.10)
        thw_diff_penalty = -abs(thw_zv_2_fv - target_thw_zv_2_fv) / max(0.5, target_thw_zv_2_fv)
        thw_component = thw_weight * thw_diff_penalty
        
        # BZV impact component
        # target_thw_bzv_2_zv = self.params.get('target_thw_bzv_2_zv', 1.0)
        # thw_bzv_2_zv = pay_off_info['ZV']['thw_BZV_2_ZV']
        # bzv_impact_weight = self.params.get('bzv_impact_weight', 0.10)
        # bzv_impact_penalty = -abs(thw_bzv_2_zv - target_thw_bzv_2_zv) / max(0.5, target_thw_bzv_2_zv)
        # bzv_impact_component = bzv_impact_weight * bzv_impact_penalty
        
        # Collision risk component
        collision_risk = self.assess_collision_risk(f_future, z_future , pay_off_info)
        collision_weight = self.params.get('collision_weight', 0.9)
        collision_component = collision_weight * (1.0 - collision_risk)
        
        # Comfort component
        comfort_component = comfort_weight * comfort
        
        # Calculate total payoff
        # zv_payoff = (comfort_component + 
        #             speed_component + 
        #             distance_component + 
        #             collision_component +  
        #             thw_component +
        #             bzv_impact_component)
        zv_payoff = (comfort_component + 
                    speed_component + 
                    distance_component + 
                    collision_component +  
                    thw_component 
                    )
        
        # Normalize to range
        zv_min = self.params.get('zv_payoff_min', -3.0)
        zv_max = self.params.get('zv_payoff_max', 3.0)
        
        return max(zv_min, min(zv_max, zv_payoff))

    def _calculate_fv_payoff(self, f_future, z_future, pay_off_info):
        # Comfort component
        comfort_fv = pay_off_info['FV']['a_FV']
        comfort_weight_fv = self.params.get('comfort_weight_fv', 0.10)
        comfort_component = comfort_weight_fv * comfort_fv
        
        # FV speed component
        target_fv_v = self.params.get('target_fv_v', 30.0)
        # fv_v = pay_off_info['FV']['v_FV']
        fv_v = f_future[9][3]
        speed_weight_fv = self.params.get('speed_weight_fv', 0.15)
        speed_diff_penalty_fv = -abs(fv_v - target_fv_v) / max(1.0, target_fv_v)
        speed_component = speed_weight_fv * speed_diff_penalty_fv
        
        # # FV distance component
        # target_fv_2_lzv_d = self.params.get('target_fv_2_lzv_d', 10.0)
        # fv_2_lzv_d = pay_off_info['FV']['FV_dis_2_LZV']
        # distance_weight_fv = self.params.get('distance_weight_fv', 0.10)
        # distance_diff_penalty_fv = -abs(fv_2_lzv_d - target_fv_2_lzv_d) / max(1.0, target_fv_2_lzv_d)
        # distance_component = distance_weight_fv * distance_diff_penalty_fv
        
        # # LFV speed component
        # target_v_lfv = self.params.get('target_v_lfv', target_fv_v)
        # lfv_speed_weight = self.params.get('lfv_speed_weight', 0.10)
        # speed_diff_penalty_LFV = -abs(target_v_lfv - pay_off_info['FV']['v_LFV']) / max(1.0, pay_off_info['FV']['v_LFV'])
        # lfv_speed_component = lfv_speed_weight * speed_diff_penalty_LFV
        
        # FV to ZV time headway component
        target_thw_fv_2_zv = self.params.get('target_thw_fv_2_zv', 1.0)
        thw_fv_2_zv = abs(z_future[9][0] - f_future[9][0])/z_future[0][3]
        thw_weight_fv = self.params.get('thw_weight_fv', 0.10)
        thw_diff_penalty_fv = -abs(thw_fv_2_zv - target_thw_fv_2_zv) / max(0.5, target_thw_fv_2_zv)
        thw_component = thw_weight_fv * thw_diff_penalty_fv
        
        # Merge pressure component
        fv_dis_2_merge = pay_off_info['FV']['FV_dis_2_merge']
        merge_critical_distance = self.params.get('merge_critical_distance', 50.0)
        merge_pressure_weight = self.params.get('merge_pressure_weight', 0.20)
        
        if fv_dis_2_merge <= merge_critical_distance:
            merge_pressure_penalty = -1.0 * (1.0 - fv_dis_2_merge / merge_critical_distance)**2
        else:
            merge_pressure_penalty = -0.1 * (merge_critical_distance / fv_dis_2_merge)
        
        merge_pressure_component = merge_pressure_weight * merge_pressure_penalty
        
        # Collision risk component
        collision_risk = self.assess_collision_risk(f_future, z_future, pay_off_info)
        collision_weight_fv = self.params.get('collision_weight_fv', 0.25)
        collision_component = collision_weight_fv * (1.0 - collision_risk)
        
        # Calculate total payoff
        # fv_payoff = (comfort_component + 
        #             speed_component + 
        #             distance_component +
        #             collision_component +
        #             thw_component +
        #             merge_pressure_component +
        #             lfv_speed_component)
        fv_payoff = (comfort_component + 
                    speed_component + 
                    collision_component +
                    thw_component +
                    merge_pressure_component)
        
        # Normalize to range
        fv_min = self.params.get('fv_payoff_min', -1.0)
        fv_max = self.params.get('fv_payoff_max', 1.0)
        
        return max(fv_min, min(fv_max, fv_payoff))