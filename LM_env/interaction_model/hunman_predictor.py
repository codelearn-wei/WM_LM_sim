import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
from LM_env.interaction_model.env_config.env_config import AwarenessParams, IDMParams
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TrajectoryPredictor:
    def __init__(self, awareness_params: AwarenessParams):
        self.awareness_params = awareness_params
        self.dt = awareness_params.dt
        self.prediction_horizon = awareness_params.prediction_horizon

    def predict_ego_trajectories(self, ego_vehicle: Dict[str, Any], main_front_vehicle: Optional[Dict[str, Any]], auxiliary_vehicles: List[Dict[str, Any]], reference_path=None, aux_reference_path=None) -> Dict[str, List[Dict[str, Any]]]:
        trajectories = {}
        for i, aux_vehicle in enumerate(auxiliary_vehicles):
            trajectories[f"follow_auxiliary_{aux_vehicle.get('vehicle_id', i)}"] = self._predict_idm_following(ego_vehicle, aux_vehicle, self.awareness_params.auxiliary_idm_params.as_dict())
        if main_front_vehicle:
            trajectories["follow_main_front"] = self._predict_idm_following_with_reference(ego_vehicle, main_front_vehicle, reference_path, self.awareness_params.main_idm_params.as_dict())
        else:
            trajectories["follow_main_front"] = self._predict_free_driving(ego_vehicle, self.awareness_params.free_driving_params.desired_velocity, reference_path)
        return trajectories

    def predict_auxiliary_vehicle_trajectories(self, auxiliary_vehicles: List[Dict[str, Any]], reference_path=None, aux_reference_path=None, num_samples=None) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
        if num_samples is None:
            num_samples = self.awareness_params.num_samples
        predictions = {}
        merge_base_distance = self.awareness_params.merge_params.merge_point_distance
        tra_interval = self.awareness_params.merge_params.tra_interval
        merge_base_duration = self.awareness_params.merge_params.merge_duration
        merge_duration_interval = self.awareness_params.merge_params.merge_duration_interval
        sample_params = [
            {
                'merge_point_distance': merge_base_distance + tra_interval * i,
                'merge_duration': merge_base_duration + merge_duration_interval * i,
                'desired_velocity': 3.0 + 0.5 * i,
                'idm_params': IDMParams(desired_velocity=3.0 + 0.5 * i, time_headway=1.5, minimum_spacing=1.5).as_dict()
            } for i in range(num_samples)
        ]
        ref_array = np.array(reference_path) if reference_path is not None and len(reference_path) > 0 else None
        for vehicle in auxiliary_vehicles:
            vehicle_id = vehicle['vehicle_id']
            vehicle_predictions = {}
            vehicle_predictions["idm_driving"] = self._predict_idm_driving_on_reference(vehicle, 4.0, aux_reference_path)
            vehicle_pos = np.array(vehicle['position'])
            nearest_ref_point = None
            if ref_array is not None:
                nearest_ref_idx = self._find_closest_point_index(vehicle_pos, ref_array)
                nearest_ref_point = reference_path[nearest_ref_idx]
            for i, params in enumerate(sample_params):
                spatial_trajectory = self._generate_spatial_merge_trajectory(vehicle, ref_array, params['merge_point_distance'], params['merge_duration'], nearest_ref_point)
                full_trajectory = self._apply_idm_to_spatial_trajectory(vehicle, spatial_trajectory, params['idm_params'])
                vehicle_predictions[f"sampled_trajectory_{i+1}"] = full_trajectory
            predictions[vehicle_id] = vehicle_predictions
        return predictions

    def _predict_idm_driving_on_reference(self, vehicle: Dict[str, Any], desired_velocity: float, reference_path=None) -> List[Dict[str, Any]]:
        vehicle['position'] = vehicle['position'].tolist()
        trajectory = [dict(vehicle)]
        current_state = trajectory[0]
        current_velocity = current_state['velocity']
        params = self.awareness_params.aux_path_following_params
        ref_array = np.array(reference_path) if reference_path is not None and len(reference_path) >= 2 else None
        if ref_array is None:
            return self._predict_idm_driving_auxiliary(vehicle, {"desired_velocity": desired_velocity})
        for _ in range(self.prediction_horizon):
            velocity_diff = desired_velocity - current_velocity
            acc = np.clip(velocity_diff / 2.0, -4.0, 3.0)
            new_velocity = max(0, current_velocity + acc * self.dt)
            current_pos = np.array(current_state['position'])
            closest_idx = self._find_closest_point_index(current_pos, ref_array)
            lookahead_distance = min(5.0, new_velocity * params.lookahead_factor)
            target_idx = self._find_lookahead_point_index(closest_idx, ref_array, lookahead_distance)
            
            if target_idx < len(ref_array):
                target_point = ref_array[target_idx]
                direction_vec = target_point - current_pos
                distance = np.linalg.norm(direction_vec)
                if distance > 0.1:
                    heading = np.arctan2(direction_vec[1], direction_vec[0])
                    heading_diff = self._normalize_angle(heading - current_state['heading'])
                    heading = current_state['heading'] + np.clip(heading_diff, -params.max_heading_change, params.max_heading_change)
                else:
                    heading = current_state['heading']
            else:
                heading = current_state['heading']
                
            dx = new_velocity * self.dt * np.cos(heading)
            dy = new_velocity * self.dt * np.sin(heading)
            new_state = dict(current_state)
            new_state['position'] = [current_state['position'][0] + dx, current_state['position'][1] + dy]
            new_state['heading'] = heading
            new_state['velocity'] = new_velocity
            new_state['acceleration'] = acc
            
            if closest_idx < len(ref_array):
                closest_point = ref_array[closest_idx]
                lateral_vector = closest_point - np.array(new_state['position'])
                lateral_distance = np.linalg.norm(lateral_vector)
                if lateral_distance > params.lateral_correction_threshold:
                    correction_factor = min(0.3, lateral_distance * params.lateral_correction_factor)
                    new_state['position'][0] += lateral_vector[0] * correction_factor
                    new_state['position'][1] += lateral_vector[1] * correction_factor
            
            trajectory.append(new_state)
            current_state = new_state
            current_velocity = new_velocity
        
        return trajectory

    def _predict_idm_following(self, ego_vehicle: Dict[str, Any], leader_vehicle: Dict[str, Any], idm_params: Dict[str, float]) -> List[Dict[str, Any]]:
        trajectory = [dict(ego_vehicle)]
        current_state = trajectory[0]
        leader_state = dict(leader_vehicle)
        for _ in range(self.prediction_horizon):
            acc = self._calculate_idm_acceleration(current_state, leader_state, idm_params)
            new_velocity = max(0, current_state['velocity'] + acc * self.dt)
            heading = current_state['heading']
            dx = new_velocity * self.dt * np.cos(heading)
            dy = new_velocity * self.dt * np.sin(heading)
            new_state = dict(current_state)
            new_state['position'] = [current_state['position'][0] + dx, current_state['position'][1] + dy]
            new_state['velocity'] = new_velocity
            new_state['acceleration'] = acc
            trajectory.append(new_state)
            current_state = new_state
            leader_state['position'][0] += leader_state['velocity'] * self.dt * np.cos(leader_state['heading'])
            leader_state['position'][1] += leader_state['velocity'] * self.dt * np.sin(leader_state['heading'])
        return trajectory

    def _predict_idm_following_with_reference(self, ego_vehicle: Dict[str, Any], leader_vehicle: Dict[str, Any], reference_path, idm_params: Dict[str, float]) -> List[Dict[str, Any]]:
        trajectory = [dict(ego_vehicle)]
        current_state = trajectory[0]
        leader_state = dict(leader_vehicle)
        params = self.awareness_params.path_following_params
        ref_array = np.array(reference_path) if reference_path is not None and len(reference_path) > 0 else None
        for _ in range(self.prediction_horizon):
            acc = self._calculate_idm_acceleration(current_state, leader_state, idm_params)
            new_velocity = max(0, current_state['velocity'] + acc * self.dt)
            current_pos = np.array(current_state['position'])
            if ref_array is not None:
                closest_idx = self._find_closest_point_index(current_pos, ref_array)
                lookahead_distance = min(params.max_lookahead, new_velocity * params.lookahead_factor)
                target_idx = self._find_lookahead_point_index(closest_idx, ref_array, lookahead_distance)
                if target_idx < len(ref_array):
                    target_point = ref_array[target_idx]
                    direction_vec = target_point - current_pos
                    distance = np.linalg.norm(direction_vec)
                    heading = (np.arctan2(direction_vec[1], direction_vec[0]) if distance > 0.1 else current_state['heading'])
                    if distance > 0.1:
                        heading_diff = self._normalize_angle(heading - current_state['heading'])
                        heading = current_state['heading'] + np.clip(heading_diff, -params.max_heading_change, params.max_heading_change)
                else:
                    heading = current_state['heading']
            else:
                heading = current_state['heading']
            dx = new_velocity * self.dt * np.cos(heading)
            dy = new_velocity * self.dt * np.sin(heading)
            new_state = dict(current_state)
            new_state['position'] = [current_state['position'][0] + dx, current_state['position'][1] + dy]
            new_state['heading'] = heading
            new_state['velocity'] = new_velocity
            new_state['acceleration'] = acc
            trajectory.append(new_state)
            current_state = new_state
            leader_state['position'][0] += leader_state['velocity'] * self.dt * np.cos(leader_state['heading'])
            leader_state['position'][1] += leader_state['velocity'] * self.dt * np.sin(leader_state['heading'])
        return trajectory

    def _find_closest_point_index(self, position: np.ndarray, reference_path: np.ndarray) -> int:
        if reference_path is None or len(reference_path) == 0:
            return 0
        distances = np.linalg.norm(reference_path - position, axis=1)
        return np.argmin(distances)

    def _find_lookahead_point_index(self, start_idx: int, reference_path: np.ndarray, lookahead_distance: float) -> int:
        accumulated_distance = 0.0
        current_idx = start_idx
        while current_idx + 1 < len(reference_path) and accumulated_distance < lookahead_distance:
            segment_distance = np.linalg.norm(reference_path[current_idx + 1] - reference_path[current_idx])
            accumulated_distance += segment_distance
            current_idx += 1
        return current_idx

    def _predict_free_driving(self, ego_vehicle: Dict[str, Any], desired_velocity: float, reference_path=None) -> List[Dict[str, Any]]:
        trajectory = [dict(ego_vehicle)]
        current_state = trajectory[0]
        current_velocity = current_state['velocity']
        params = self.awareness_params.path_following_params
        ref_array = np.array(reference_path) if reference_path is not None and len(reference_path) > 0 else None
        for _ in range(self.prediction_horizon):
            velocity_diff = desired_velocity - current_velocity
            acc = np.clip(velocity_diff / 2.0, -4.0, 3.0)
            new_velocity = max(0, current_velocity + acc * self.dt)
            if ref_array is not None:
                current_pos = np.array(current_state['position'])
                closest_idx = self._find_closest_point_index(current_pos, ref_array)
                lookahead_distance = min(params.max_lookahead, new_velocity * params.lookahead_factor)
                target_idx = self._find_lookahead_point_index(closest_idx, ref_array, lookahead_distance)
                if target_idx < len(ref_array):
                    target_point = ref_array[target_idx]
                    direction_vec = target_point - current_pos
                    distance = np.linalg.norm(direction_vec)
                    heading = (np.arctan2(direction_vec[1], direction_vec[0]) if distance > 0.1 else current_state['heading'])
                    if distance > 0.1:
                        heading_diff = self._normalize_angle(heading - current_state['heading'])
                        heading = current_state['heading'] + np.clip(heading_diff, -params.max_heading_change, params.max_heading_change)
                else:
                    heading = current_state['heading']
            else:
                heading = current_state['heading']
            dx = new_velocity * self.dt * np.cos(heading)
            dy = new_velocity * self.dt * np.sin(heading)
            new_state = dict(current_state)
            new_state['position'] = [current_state['position'][0] + dx, current_state['position'][1] + dy]
            new_state['heading'] = heading
            new_state['velocity'] = new_velocity
            new_state['acceleration'] = acc
            trajectory.append(new_state)
            current_state = new_state
            current_velocity = new_velocity
        return trajectory

    def _predict_idm_driving_auxiliary(self, vehicle: Dict[str, Any], idm_params: Dict[str, float]) -> List[Dict[str, Any]]:
        trajectory = [dict(vehicle)]
        current_state = trajectory[0]
        current_velocity = current_state['velocity']
        desired_velocity = idm_params.get('desired_velocity', 3.0)
        max_acc = idm_params.get('max_acceleration', 3.0)
        for _ in range(self.prediction_horizon):
            velocity_ratio = current_velocity / desired_velocity if desired_velocity > 0 else 0
            acc = max_acc * (1 - velocity_ratio ** 4)
            new_velocity = max(0, current_velocity + acc * self.dt)
            heading = current_state['heading']
            dx = new_velocity * self.dt * np.cos(heading)
            dy = new_velocity * self.dt * np.sin(heading)
            new_state = dict(current_state)
            new_state['position'] = [current_state['position'][0] + dx, current_state['position'][1] + dy].tolist()
            new_state['velocity'] = new_velocity
            new_state['acceleration'] = acc
            trajectory.append(new_state)
            current_state = new_state
            current_velocity = new_velocity
        return trajectory

    def _generate_spatial_merge_trajectory(self, vehicle: Dict[str, Any], reference_path=None, merge_point_distance: float = 30.0, merge_duration: float = 3.0, cached_nearest_point=None) -> List[Dict[str, List[float]]]:
        merge_steps = min(int(merge_duration / self.dt), self.prediction_horizon)
        start_pos = np.array(vehicle['position'])
        start_heading = vehicle['heading']
        if cached_nearest_point is not None and reference_path is not None and len(reference_path) > 0:
            merge_point = self._find_merge_point_from_cached(start_pos, reference_path, merge_point_distance, cached_nearest_point)
        else:
            merge_point = self._find_merge_point(vehicle, reference_path, merge_point_distance)
        merge_heading = self._estimate_reference_heading(merge_point, reference_path)
        distance = np.linalg.norm(merge_point - start_pos)
        control_distance = distance / self.awareness_params.merge_params.control_point_factor
        control1 = start_pos + control_distance * np.array([np.cos(start_heading), np.sin(start_heading)])
        control2 = merge_point - control_distance * np.array([np.cos(merge_heading), np.sin(merge_heading)])
        spatial_trajectory = [{'position': start_pos.tolist(), 'heading': start_heading}]
        t_values = np.linspace(0, 1, merge_steps + 1)[1:]
        positions = [(1-t)**3 * start_pos + 3*(1-t)**2*t * control1 + 3*(1-t)*t**2 * control2 + t**3 * merge_point for t in t_values]
        for i, pos in enumerate(positions):
            heading = (np.arctan2(positions[i][1] - (start_pos[1] if i == 0 else positions[i-1][1]), 
                                 positions[i][0] - (start_pos[0] if i == 0 else positions[i-1][0])))
            spatial_trajectory.append({'position': pos.tolist(), 'heading': heading})
        remaining_steps = self.prediction_horizon - merge_steps
        if remaining_steps > 0:
            last_pos = np.array(spatial_trajectory[-1]['position'])
            last_heading = spatial_trajectory[-1]['heading']
            step_dx = 3.0 * self.dt * np.cos(last_heading)
            step_dy = 3.0 * self.dt * np.sin(last_heading)
            for i in range(1, remaining_steps + 1):
                new_pos = [last_pos[0] + i * step_dx, last_pos[1] + i * step_dy]
                spatial_trajectory.append({'position': new_pos.tolist(), 'heading': last_heading})
        return spatial_trajectory

    def _apply_idm_to_spatial_trajectory(self, vehicle: Dict[str, Any], spatial_trajectory: List[Dict[str, List[float]]], idm_params: Dict[str, float]) -> List[Dict[str, Any]]:
        full_trajectory = [dict(vehicle)]
        current_state = full_trajectory[0]
        current_velocity = current_state['velocity']
        desired_velocity = idm_params.get('desired_velocity', 3.0)
        max_acceleration = idm_params.get('max_acceleration', 3.0)
        max_deceleration = idm_params.get('comfortable_deceleration', 4.0)
        for i in range(1, len(spatial_trajectory)):
            target_pos = np.array(spatial_trajectory[i]['position'])
            target_heading = spatial_trajectory[i]['heading']
            current_pos = np.array(current_state['position'])
            distance_to_target = np.linalg.norm(target_pos - current_pos)
            target_velocity = (min(np.linalg.norm(np.array(spatial_trajectory[i + 1]['position']) - target_pos) / self.dt, desired_velocity) 
                              if i < len(spatial_trajectory) - 1 else desired_velocity)
            velocity_diff = target_velocity - current_velocity
            acc = np.clip(velocity_diff / self.dt, -max_deceleration, max_acceleration)
            new_velocity = max(0, current_velocity + acc * self.dt)
            actual_heading = target_heading if new_velocity < 0.1 else (np.arctan2(target_pos[1] - current_pos[1], target_pos[0] - current_pos[0]) 
                                                                      if distance_to_target > 0.01 else target_heading)
            new_state = dict(current_state)
            new_state['position'] = target_pos.tolist()
            new_state['heading'] = actual_heading
            new_state['velocity'] = new_velocity
            new_state['acceleration'] = acc
            full_trajectory.append(new_state)
            current_state = new_state
            current_velocity = new_velocity
        self._smooth_trajectory_headings(full_trajectory)
        return full_trajectory

    def _smooth_trajectory_headings(self, trajectory: List[Dict[str, Any]], window_size=3):
        if len(trajectory) <= window_size:
            return
        headings = np.array([state['heading'] for state in trajectory])
        smoothed_headings = headings.copy()
        half_window = window_size // 2
        for i in range(half_window, len(trajectory) - half_window):
            window = headings[max(0, i - half_window):min(len(trajectory), i + half_window + 1)]
            complex_sum = np.sum(np.exp(1j * window))
            smoothed_headings[i] = np.angle(complex_sum)
        for i, state in enumerate(trajectory):
            state['heading'] = smoothed_headings[i]

    def _find_merge_point(self, vehicle: Dict[str, Any], reference_path, merge_distance: float) -> np.ndarray:
        vehicle_pos = np.array(vehicle['position'])
        if reference_path is None or len(reference_path) == 0:
            return vehicle_pos + merge_distance * np.array([np.cos(vehicle['heading']), np.sin(vehicle['heading'])])
        ref_array = np.array(reference_path)
        distances = np.linalg.norm(ref_array - vehicle_pos, axis=1)
        projection_point = np.argmin(distances)
        target_index = projection_point
        accumulated_distance = 0
        while accumulated_distance < merge_distance and target_index + 1 < len(ref_array):
            segment_distance = np.linalg.norm(ref_array[target_index + 1] - ref_array[target_index])
            accumulated_distance += segment_distance
            target_index += 1
        return ref_array[min(target_index, len(ref_array) - 1)]

    def _estimate_reference_heading(self, point: np.ndarray, reference_path) -> float:
        if reference_path is None or len(reference_path) < 2:
            return 0.0
        ref_array = np.array(reference_path)
        distances = np.linalg.norm(ref_array - point, axis=1)
        nearest_idx = np.argmin(distances)
        if nearest_idx < len(ref_array) - 1:
            p1, p2 = ref_array[nearest_idx], ref_array[nearest_idx + 1]
            return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        elif nearest_idx > 0:
            p1, p2 = ref_array[nearest_idx - 1], ref_array[nearest_idx]
            return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        return 0.0

    def _normalize_angle(self, angle: float) -> float:
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _calculate_idm_acceleration(self, ego_vehicle: Dict[str, Any], leader_vehicle: Dict[str, Any], idm_params: Dict[str, float]) -> float:
        v0 = idm_params.get('desired_velocity', 5.0)
        a_max = idm_params.get('max_acceleration', 3.0)
        b = idm_params.get('comfortable_deceleration', 4.0)
        s0 = idm_params.get('minimum_spacing', 0.5)
        T = idm_params.get('time_headway', 1.0)
        v_ego = ego_vehicle['velocity']
        v_leader = leader_vehicle['velocity']
        distance = np.linalg.norm(np.array(leader_vehicle['position']) - np.array(ego_vehicle['position']))
        s = max(0.1, distance - leader_vehicle.get('length', 4.0))
        delta_v = v_ego - v_leader
        s_star = s0 + max(0, v_ego * T + (v_ego * delta_v) / (2 * np.sqrt(a_max * b)))
        acc = a_max * (1 - (v_ego / v0) ** 4 - (s_star / s) ** 2)
        return np.clip(acc, -b, a_max)

    def _find_merge_point_from_cached(self, vehicle_pos: np.ndarray, reference_path, merge_distance: float, cached_nearest_point) -> np.ndarray:
        if reference_path is None or len(reference_path) == 0:
            return vehicle_pos + merge_distance * np.array([1.0, 0.0])
        ref_array = np.array(reference_path)
        projection_point = np.argmin(np.linalg.norm(ref_array - np.array(cached_nearest_point), axis=1))
        target_index = projection_point
        accumulated_distance = 0
        while accumulated_distance < merge_distance and target_index + 1 < len(ref_array):
            segment_distance = np.linalg.norm(ref_array[target_index + 1] - ref_array[target_index])
            accumulated_distance += segment_distance
            target_index += 1
        return ref_array[min(target_index, len(ref_array) - 1)]

def predict_trajectories(ego_vehicle, main_front_vehicle, auxiliary_vehicles, reference_path=None, auxiliary_reference_path=None, awareness_params=None):
    if awareness_params is None:
        awareness_params = AwarenessParams()
    predictor = TrajectoryPredictor(awareness_params)
    main_front = main_front_vehicle[0] if main_front_vehicle else None
    ego_trajectories = predictor.predict_ego_trajectories(ego_vehicle, main_front, auxiliary_vehicles, reference_path, auxiliary_reference_path)
    auxiliary_trajectories = predictor.predict_auxiliary_vehicle_trajectories(auxiliary_vehicles, reference_path, auxiliary_reference_path)
    return {"ego": ego_trajectories, "auxiliary": auxiliary_trajectories}

def plot_trajectories(ego_vehicle: Dict[str, Any], main_front_vehicle: Optional[List[Dict[str, Any]]], auxiliary_vehicles: List[Dict[str, Any]], predicted_trajectories: Dict[str, Any], reference_path=None, aux_reference_path=None, plot_size=(16, 6), save_path=None, dpi=300, arrow_length=2, show_all_headings=True):
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'lines.linewidth': 2.5,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.7
    })
    fig, ax = plt.subplots(figsize=plot_size)
    if reference_path is not None and len(reference_path) > 0:
        ref_x, ref_y = zip(*reference_path)
        ax.plot(ref_x, ref_y, 'k--', linewidth=1.5, label='参考路径', alpha=0.7)
    if aux_reference_path is not None and len(aux_reference_path) > 0:
        aux_ref_x, aux_ref_y = zip(*aux_reference_path)
        ax.plot(aux_ref_x, aux_ref_y, 'k:', linewidth=1.5, label='辅道参考路径', alpha=0.7)
    marker_style = {'markersize': 9, 'markeredgewidth': 1.5, 'alpha': 0.9}

    def draw_heading_arrow(position, heading, color, alpha=1.0):
        dx = arrow_length * np.cos(heading)
        dy = arrow_length * np.sin(heading)
        ax.arrow(position[0], position[1], dx, dy, head_width=0.4, head_length=0.4, fc=color, ec=color, alpha=alpha, length_includes_head=True)

    ax.plot(ego_vehicle['position'][0], ego_vehicle['position'][1], 'bo', label='主车', **marker_style)
    draw_heading_arrow(ego_vehicle['position'], ego_vehicle['heading'], 'blue')
    if main_front_vehicle:
        ax.plot(main_front_vehicle[0]['position'][0], main_front_vehicle[0]['position'][1], 'go', label='主车道前方车辆', **marker_style)
        draw_heading_arrow(main_front_vehicle[0]['position'], main_front_vehicle[0]['heading'], 'green')
    aux_label_used = False
    for aux_vehicle in auxiliary_vehicles:
        label = '辅道车辆' if not aux_label_used else None
        aux_label_used = True
        ax.plot(aux_vehicle['position'][0], aux_vehicle['position'][1], 'ro', label=label, **marker_style)
        draw_heading_arrow(aux_vehicle['position'], aux_vehicle['heading'], 'red')
    ego_colors = {'follow_main_front': '#1f77b4', 'follow_auxiliary': '#ff7f0e'}
    auxiliary_colors = {'idm_driving': '#2ca02c', 'sampled_trajectory': ['#d62728', '#9467bd', '#8c564b']}
    for strategy, trajectory in predicted_trajectories['ego'].items():
        traj_x = [state['position'][0] for state in trajectory]
        traj_y = [state['position'][1] for state in trajectory]
        if 'follow_auxiliary' in strategy:
            aux_id = strategy.split('_')[-1]
            color = ego_colors['follow_auxiliary']
            label = f'主车-跟随辅道车辆{aux_id}'
        else:
            color = ego_colors['follow_main_front']
            label = '主车-跟随主车道'
        ax.plot(traj_x, traj_y, color=color, linewidth=2.5, label=label)
        draw_heading_arrow(trajectory[0]['position'], trajectory[0]['heading'], color, alpha=0.8)
        draw_heading_arrow(trajectory[-1]['position'], trajectory[-1]['heading'], color, alpha=0.8)
        if show_all_headings:
            for i in range(5, len(trajectory), 5):
                draw_heading_arrow(trajectory[i]['position'], trajectory[i]['heading'], color, alpha=0.6)
    for vehicle_id, vehicle_trajectories in predicted_trajectories['auxiliary'].items():
        for strategy, trajectory in vehicle_trajectories.items():
            traj_x = [state['position'][0] for state in trajectory]
            traj_y = [state['position'][1] for state in trajectory]
            if 'idm_driving' in strategy:
                color = auxiliary_colors['idm_driving']
                linestyle = '-'
                label = f'辅道车 {vehicle_id}-IDM驱动'
            else:
                sample_num = int(strategy.split('_')[-1]) - 1
                color = auxiliary_colors['sampled_trajectory'][sample_num % len(auxiliary_colors['sampled_trajectory'])]
                linestyle = '--'
                label = f'辅道车 {vehicle_id}-采样轨迹{sample_num + 1}'
            ax.plot(traj_x, traj_y, color=color, linewidth=1.8, linestyle=linestyle, alpha=0.8, label=label)
            draw_heading_arrow(trajectory[0]['position'], trajectory[0]['heading'], color, alpha=0.8)
            draw_heading_arrow(trajectory[-1]['position'], trajectory[-1]['heading'], color, alpha=0.8)
            if show_all_headings:
                for i in range(5, len(trajectory), 5):
                    draw_heading_arrow(trajectory[i]['position'], trajectory[i]['heading'], color, alpha=0.6)
    ax.set_title('车辆轨迹预测分析', fontweight='bold', pad=15)
    ax.set_xlabel('X位置 (米)', labelpad=10)
    ax.set_ylabel('Y位置 (米)', labelpad=10)
    ax.axis('equal')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', frameon=True, shadow=True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()