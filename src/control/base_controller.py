"""
控制器基类
定义路径跟踪控制器的统一接口
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

class ControllerMetrics:
    """控制器性能指标"""
    
    def __init__(self):
        self.lateral_errors = []
        self.heading_errors = []
        self.control_efforts = []
        self.tracking_time = 0.0
        self.max_lateral_error = 0.0
        self.rms_lateral_error = 0.0
        self.max_heading_error = 0.0
        self.rms_heading_error = 0.0
        self.control_smoothness = 0.0
        self.success = False
        
    def update_errors(self, lateral_error: float, heading_error: float, control_effort: float):
        """更新误差记录"""
        self.lateral_errors.append(lateral_error)
        self.heading_errors.append(heading_error)
        self.control_efforts.append(control_effort)
        
    def calculate_final_metrics(self):
        """计算最终性能指标"""
        if not self.lateral_errors:
            return
            
        # 横向误差指标
        self.max_lateral_error = max(abs(e) for e in self.lateral_errors)
        self.rms_lateral_error = np.sqrt(np.mean([e**2 for e in self.lateral_errors]))
        
        # 航向误差指标
        self.max_heading_error = max(abs(e) for e in self.heading_errors)
        self.rms_heading_error = np.sqrt(np.mean([e**2 for e in self.heading_errors]))
        
        # 控制平滑度（控制输入变化率）
        if len(self.control_efforts) > 1:
            control_changes = [abs(self.control_efforts[i] - self.control_efforts[i-1]) 
                             for i in range(1, len(self.control_efforts))]
            self.control_smoothness = np.mean(control_changes)

class BaseController(ABC):
    """路径跟踪控制器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = ControllerMetrics()
        self.tracking_history = []
        
    @abstractmethod
    def compute_control(self, robot_state: Tuple[float, float, float], 
                       reference_path: List[Tuple[float, float]], 
                       current_target_idx: int, 
                       dt: float) -> Tuple[float, float]:
        """
        计算控制输入
        
        Args:
            robot_state: 机器人状态 (x, y, theta)
            reference_path: 参考路径
            current_target_idx: 当前目标点索引
            dt: 时间步长
            
        Returns:
            控制输入 (取决于具体控制器类型)
        """
        pass
    
    def track_path(self, robot, reference_path: List[Tuple[float, float]], 
                   dt: float = 0.1, max_time: float = 100.0, 
                   goal_tolerance: float = 0.5) -> Tuple[List, ControllerMetrics]:
        """
        执行路径跟踪
        
        Args:
            robot: 机器人对象
            reference_path: 参考路径
            dt: 时间步长
            max_time: 最大跟踪时间
            goal_tolerance: 目标容差
            
        Returns:
            (跟踪轨迹, 性能指标)
        """
        if len(reference_path) < 2:
            return [], self.metrics
            
        # 重置指标和历史
        self.metrics = ControllerMetrics()
        self.tracking_history.clear()
        
        # 初始化机器人位置到路径起点
        robot.set_state(reference_path[0][0], reference_path[0][1], 0)
        
        # 跟踪变量
        trajectory = []
        current_target_idx = 1
        start_time = time.time()
        
        while time.time() - start_time < max_time:
            current_time = time.time() - start_time
            
            # 获取机器人当前状态
            robot_pos = robot.get_position()
            robot_theta = robot.get_orientation()
            robot_state = (robot_pos[0], robot_pos[1], robot_theta)
            
            # 记录轨迹
            trajectory.append({
                'time': current_time,
                'position': robot_pos,
                'orientation': robot_theta,
                'target_idx': current_target_idx
            })
            
            # 检查是否到达目标
            goal_pos = reference_path[-1]
            distance_to_goal = np.sqrt((robot_pos[0] - goal_pos[0])**2 + 
                                     (robot_pos[1] - goal_pos[1])**2)
            
            if distance_to_goal < goal_tolerance:
                self.metrics.success = True
                break
            
            # 更新目标点索引
            current_target_idx = self._update_target_index(robot_pos, reference_path, current_target_idx)
            
            # 计算控制输入
            try:
                control_input = self.compute_control(robot_state, reference_path, 
                                                   current_target_idx, dt)
                
                # 更新机器人状态
                if hasattr(robot, 'update_state'):
                    if len(control_input) == 2:
                        robot.update_state(control_input[0], control_input[1], dt)
                    elif len(control_input) == 3:  # 阿克曼模型
                        robot.update_state(control_input[0], control_input[1], dt)
                
                # 计算跟踪误差
                lateral_error, heading_error = self._calculate_tracking_errors(
                    robot_state, reference_path, current_target_idx)
                
                # 控制力度（简单地使用控制输入的模长）
                control_effort = np.linalg.norm(control_input)
                
                # 更新指标
                self.metrics.update_errors(lateral_error, heading_error, control_effort)
                
            except Exception as e:
                print(f"Control computation failed: {e}")
                break
        
        # 计算最终指标
        self.metrics.tracking_time = time.time() - start_time
        self.metrics.calculate_final_metrics()
        
        return trajectory, self.metrics
    
    def _update_target_index(self, robot_pos: Tuple[float, float], 
                            reference_path: List[Tuple[float, float]], 
                            current_idx: int) -> int:
        """更新目标点索引"""
        # 找到距离机器人最近的路径点
        min_distance = float('inf')
        closest_idx = current_idx
        
        # 在当前索引附近搜索
        search_range = min(15, len(reference_path) - current_idx)
        
        for i in range(current_idx, min(current_idx + search_range, len(reference_path))):
            distance = np.sqrt((robot_pos[0] - reference_path[i][0])**2 + 
                             (robot_pos[1] - reference_path[i][1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        # 如果太接近当前目标点，移动到下一个点
        if min_distance < 0.3 and closest_idx < len(reference_path) - 1:
            return closest_idx + 1
        
        return max(closest_idx, current_idx)
    
    def _calculate_tracking_errors(self, robot_state: Tuple[float, float, float], 
                                  reference_path: List[Tuple[float, float]], 
                                  target_idx: int) -> Tuple[float, float]:
        """计算跟踪误差"""
        robot_x, robot_y, robot_theta = robot_state
        
        if target_idx >= len(reference_path):
            target_idx = len(reference_path) - 1
        
        # 横向误差：机器人到参考路径的距离
        if target_idx > 0:
            # 使用前一个点和当前目标点构成的线段
            p1 = np.array(reference_path[target_idx - 1])
            p2 = np.array(reference_path[target_idx])
            robot_pos = np.array([robot_x, robot_y])
            
            # 计算点到线段的距离
            line_vec = p2 - p1
            if np.linalg.norm(line_vec) > 1e-6:
                line_vec_normalized = line_vec / np.linalg.norm(line_vec)
                robot_vec = robot_pos - p1
                projection = np.dot(robot_vec, line_vec_normalized)
                projection = max(0, min(projection, np.linalg.norm(line_vec)))
                closest_point = p1 + projection * line_vec_normalized
                lateral_error = np.linalg.norm(robot_pos - closest_point)
                
                # 判断左右偏差的符号
                cross_product = np.cross(line_vec, robot_vec)
                lateral_error = lateral_error if cross_product >= 0 else -lateral_error
            else:
                lateral_error = np.linalg.norm(robot_pos - p1)
        else:
            target_pos = reference_path[target_idx]
            lateral_error = np.sqrt((robot_x - target_pos[0])**2 + (robot_y - target_pos[1])**2)
        
        # 航向误差：机器人朝向与路径切线的角度差
        if target_idx > 0 and target_idx < len(reference_path):
            p1 = reference_path[target_idx - 1]
            p2 = reference_path[target_idx]
            path_heading = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        else:
            path_heading = robot_theta  # 如果无法计算路径朝向，使用机器人当前朝向
        
        heading_error = robot_theta - path_heading
        
        # 将角度差归一化到[-π, π]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        return lateral_error, heading_error
    
    def visualize_tracking_result(self, trajectory: List, reference_path: List[Tuple[float, float]], 
                                 environment=None, ax=None):
        """可视化跟踪结果"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制环境
        if environment is not None:
            environment.visualize(ax=ax, show_grid=False)
        
        # 绘制参考路径
        if reference_path:
            ref_path_array = np.array(reference_path)
            ax.plot(ref_path_array[:, 0], ref_path_array[:, 1], 'r--', 
                   linewidth=2, label='Reference Path')
        
        # 绘制实际轨迹
        if trajectory:
            positions = [t['position'] for t in trajectory]
            traj_array = np.array(positions)
            ax.plot(traj_array[:, 0], traj_array[:, 1], 'b-', 
                   linewidth=2, label=f'{self.name} Trajectory')
            
            # 绘制起点和终点
            ax.plot(traj_array[0, 0], traj_array[0, 1], 'go', markersize=8, label='Start')
            ax.plot(traj_array[-1, 0], traj_array[-1, 1], 'ro', markersize=8, label='End')
        
        ax.legend()
        ax.set_title(f'{self.name} Path Tracking Result')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def get_control_statistics(self) -> Dict[str, Any]:
        """获取控制统计信息"""
        return {
            'controller': self.name,
            'success': self.metrics.success,
            'tracking_time': self.metrics.tracking_time,
            'max_lateral_error': self.metrics.max_lateral_error,
            'rms_lateral_error': self.metrics.rms_lateral_error,
            'max_heading_error': self.metrics.max_heading_error,
            'rms_heading_error': self.metrics.rms_heading_error,
            'control_smoothness': self.metrics.control_smoothness
        }

if __name__ == "__main__":
    # 测试基础功能
    print("Testing base controller functionality...")
    
    # 测试指标计算
    metrics = ControllerMetrics()
    
    # 模拟一些误差数据
    for i in range(100):
        lateral_error = 0.1 * np.sin(i * 0.1)
        heading_error = 0.05 * np.cos(i * 0.1)
        control_effort = 1.0 + 0.1 * np.sin(i * 0.05)
        
        metrics.update_errors(lateral_error, heading_error, control_effort)
    
    metrics.calculate_final_metrics()
    
    print(f"Max lateral error: {metrics.max_lateral_error:.3f}")
    print(f"RMS lateral error: {metrics.rms_lateral_error:.3f}")
    print(f"Max heading error: {metrics.max_heading_error:.3f}")
    print(f"RMS heading error: {metrics.rms_heading_error:.3f}")
    print(f"Control smoothness: {metrics.control_smoothness:.3f}")
