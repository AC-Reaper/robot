"""
简化的PID控制器实现
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.control.base_controller import BaseController

class SimplePIDController(BaseController):
    """简化的PID控制器"""
    
    def __init__(self, kp: float = 1.0, max_angular_velocity: float = 0.5):
        """
        初始化简化PID控制器
        
        Args:
            kp: 比例增益
            max_angular_velocity: 最大角速度
        """
        super().__init__("Simple_PID")
        self.kp = kp
        self.max_angular_velocity = max_angular_velocity
        
    def compute_control(self, robot_state: tuple, 
                       reference_path: list, 
                       current_target_idx: int, 
                       dt: float) -> tuple:
        """
        计算简化的PID控制输入
        """
        robot_x, robot_y, robot_theta = robot_state
        
        if current_target_idx >= len(reference_path):
            return (0.0, 0.0)
        
        # 找到最近的路径点
        closest_idx = self._find_closest_point(robot_state, reference_path)
        
        # 获取目标点（稍微超前一点）
        target_idx = min(closest_idx + 2, len(reference_path) - 1)
        target_x, target_y = reference_path[target_idx]
        
        # 计算到目标点的角度
        target_angle = np.arctan2(target_y - robot_y, target_x - robot_x)
        
        # 计算航向误差
        heading_error = target_angle - robot_theta
        
        # 归一化角度差
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # 计算距离到目标
        distance_to_target = np.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
        
        # 基于距离目标的速度调节
        if distance_to_target < 0.5:
            linear_velocity = 0.5
        elif distance_to_target > 3.0:
            linear_velocity = 1.5
        else:
            linear_velocity = 1.0
        
        # 简化的PID控制律（只使用比例项）
        angular_velocity = self.kp * heading_error
        
        # 限制角速度
        angular_velocity = np.clip(angular_velocity, 
                                 -self.max_angular_velocity, 
                                 self.max_angular_velocity)
        
        return (linear_velocity, angular_velocity)
    
    def _find_closest_point(self, robot_state: tuple, reference_path: list) -> int:
        """找到最近的路径点"""
        robot_x, robot_y, robot_theta = robot_state
        min_distance = float('inf')
        closest_idx = 0
        
        for i, (path_x, path_y) in enumerate(reference_path):
            distance = np.sqrt((robot_x - path_x)**2 + (robot_y - path_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        return closest_idx
