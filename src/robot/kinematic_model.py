"""
机器人运动学模型
支持差分驱动和阿克曼转向两种模型
"""

import numpy as np
from typing import Tuple, List
import math

class DifferentialDriveRobot:
    """差分驱动机器人模型"""
    
    def __init__(self, wheelbase: float = 0.5, max_speed: float = 2.0, max_angular_speed: float = np.pi):
        """
        初始化差分驱动机器人
        
        Args:
            wheelbase: 轮距
            max_speed: 最大线速度
            max_angular_speed: 最大角速度
        """
        self.wheelbase = wheelbase
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        
        # 机器人状态 [x, y, theta]
        self.state = np.array([0.0, 0.0, 0.0])
        
        # 机器人尺寸（用于碰撞检测）
        self.radius = wheelbase / 2
        
    def update_state(self, linear_vel: float, angular_vel: float, dt: float):
        """
        更新机器人状态
        
        Args:
            linear_vel: 线速度
            angular_vel: 角速度
            dt: 时间步长
        """
        # 限制速度
        linear_vel = np.clip(linear_vel, -self.max_speed, self.max_speed)
        angular_vel = np.clip(angular_vel, -self.max_angular_speed, self.max_angular_speed)
        
        # 更新状态
        x, y, theta = self.state
        
        # 运动学方程
        if abs(angular_vel) < 1e-6:  # 直线运动
            x += linear_vel * math.cos(theta) * dt
            y += linear_vel * math.sin(theta) * dt
        else:  # 曲线运动
            # 瞬时转向半径
            R = linear_vel / angular_vel
            
            # 瞬时转向中心
            ICR_x = x - R * math.sin(theta)
            ICR_y = y + R * math.cos(theta)
            
            # 更新位置和朝向
            theta += angular_vel * dt
            x = ICR_x + R * math.sin(theta)
            y = ICR_y - R * math.cos(theta)
        
        self.state = np.array([x, y, theta])
    
    def get_position(self) -> Tuple[float, float]:
        """获取当前位置"""
        return self.state[0], self.state[1]
    
    def get_orientation(self) -> float:
        """获取当前朝向"""
        return self.state[2]
    
    def set_state(self, x: float, y: float, theta: float):
        """设置机器人状态"""
        self.state = np.array([x, y, theta])
    
    def get_vertices(self) -> np.ndarray:
        """获取机器人外形顶点（矩形近似）"""
        x, y, theta = self.state
        
        # 机器人尺寸
        length = self.wheelbase * 1.2
        width = self.wheelbase * 0.8
        
        # 机器人中心相对顶点
        vertices_local = np.array([
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2],
            [-length/2, width/2]
        ])
        
        # 旋转矩阵
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # 旋转并平移
        vertices_global = vertices_local @ rotation_matrix.T + np.array([x, y])
        
        return vertices_global

class AckermannRobot:
    """阿克曼转向机器人模型"""
    
    def __init__(self, wheelbase: float = 2.5, max_speed: float = 10.0, max_steering_angle: float = np.pi/4):
        """
        初始化阿克曼转向机器人
        
        Args:
            wheelbase: 轴距
            max_speed: 最大速度
            max_steering_angle: 最大转向角
        """
        self.wheelbase = wheelbase
        self.max_speed = max_speed
        self.max_steering_angle = max_steering_angle
        
        # 机器人状态 [x, y, theta, v]
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        
        # 机器人尺寸
        self.length = wheelbase * 1.2
        self.width = wheelbase * 0.5
        
    def update_state(self, acceleration: float, steering_angle: float, dt: float):
        """
        更新机器人状态
        
        Args:
            acceleration: 加速度
            steering_angle: 转向角
            dt: 时间步长
        """
        # 限制转向角
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        x, y, theta, v = self.state
        
        # 更新速度
        v += acceleration * dt
        v = np.clip(v, -self.max_speed, self.max_speed)
        
        # 更新位置和朝向
        if abs(steering_angle) < 1e-6:  # 直线运动
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
        else:  # 阿克曼转向
            # 转向半径
            R = self.wheelbase / math.tan(steering_angle)
            
            # 角速度
            omega = v / R
            
            # 更新状态
            theta += omega * dt
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
        
        self.state = np.array([x, y, theta, v])
    
    def get_position(self) -> Tuple[float, float]:
        """获取当前位置"""
        return self.state[0], self.state[1]
    
    def get_orientation(self) -> float:
        """获取当前朝向"""
        return self.state[2]
    
    def get_speed(self) -> float:
        """获取当前速度"""
        return self.state[3]
    
    def set_state(self, x: float, y: float, theta: float, v: float = 0.0):
        """设置机器人状态"""
        self.state = np.array([x, y, theta, v])
    
    def get_vertices(self) -> np.ndarray:
        """获取机器人外形顶点（矩形近似）"""
        x, y, theta, _ = self.state
        
        # 机器人中心相对顶点
        vertices_local = np.array([
            [-self.length/2, -self.width/2],
            [self.length/2, -self.width/2],
            [self.length/2, self.width/2],
            [-self.length/2, self.width/2]
        ])
        
        # 旋转矩阵
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # 旋转并平移
        vertices_global = vertices_local @ rotation_matrix.T + np.array([x, y])
        
        return vertices_global

class RobotTrajectory:
    """机器人轨迹记录类"""
    
    def __init__(self):
        self.positions = []
        self.orientations = []
        self.timestamps = []
        self.control_inputs = []
        
    def add_state(self, position: Tuple[float, float], orientation: float, 
                  timestamp: float, control_input: Tuple[float, float] = None):
        """添加状态记录"""
        self.positions.append(position)
        self.orientations.append(orientation)
        self.timestamps.append(timestamp)
        if control_input is not None:
            self.control_inputs.append(control_input)
    
    def get_positions_array(self) -> np.ndarray:
        """获取位置数组"""
        return np.array(self.positions)
    
    def get_orientations_array(self) -> np.ndarray:
        """获取朝向数组"""
        return np.array(self.orientations)
    
    def clear(self):
        """清空轨迹"""
        self.positions.clear()
        self.orientations.clear()
        self.timestamps.clear()
        self.control_inputs.clear()

def create_robot(robot_type: str = "differential", **kwargs):
    """
    创建机器人实例的工厂函数
    
    Args:
        robot_type: 机器人类型 ("differential" 或 "ackermann")
        **kwargs: 机器人参数
    
    Returns:
        机器人实例
    """
    if robot_type.lower() == "differential":
        return DifferentialDriveRobot(**kwargs)
    elif robot_type.lower() == "ackermann":
        return AckermannRobot(**kwargs)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}. Available: 'differential', 'ackermann'")

if __name__ == "__main__":
    # 测试差分驱动机器人
    robot = DifferentialDriveRobot()
    robot.set_state(0, 0, 0)
    
    print("Testing Differential Drive Robot:")
    print(f"Initial state: {robot.state}")
    
    # 模拟运动
    dt = 0.1
    for i in range(10):
        robot.update_state(1.0, 0.5, dt)  # 线速度1.0, 角速度0.5
        print(f"Step {i+1}: Position: {robot.get_position()}, Orientation: {robot.get_orientation():.2f}")
    
    print("\nTesting Ackermann Robot:")
    ackermann = AckermannRobot()
    ackermann.set_state(0, 0, 0, 0)
    
    print(f"Initial state: {ackermann.state}")
    
    # 模拟运动
    for i in range(10):
        ackermann.update_state(1.0, 0.2, dt)  # 加速度1.0, 转向角0.2
        print(f"Step {i+1}: Position: {ackermann.get_position()}, Speed: {ackermann.get_speed():.2f}")
