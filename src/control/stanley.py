"""
Stanley 路径跟踪控制器
基于前轮位置的横向误差和航向误差的控制方法
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.control.base_controller import BaseController

class StanleyController(BaseController):
    """Stanley 控制器"""
    
    def __init__(self, k_crosstrack: float = 2.5, k_soft: float = 0.1, 
                 max_steering_angle: float = np.pi/3, wheelbase: float = 0.5):
        """
        初始化Stanley控制器
        
        Args:
            k_crosstrack: 横向误差增益
            k_soft: 软化参数，避免低速时的数值问题
            max_steering_angle: 最大转向角
            wheelbase: 车辆轴距
        """
        super().__init__("Stanley")
        self.k_crosstrack = k_crosstrack
        self.k_soft = k_soft
        self.max_steering_angle = max_steering_angle
        self.wheelbase = wheelbase
        
        # 记录控制详情（用于可视化）
        self.control_history = []
        
    def compute_control(self, robot_state: tuple, 
                       reference_path: list, 
                       current_target_idx: int, 
                       dt: float) -> tuple:
        """
        计算Stanley控制输入
        
        Returns:
            (speed, steering_angle) for Ackermann model
            or (linear_velocity, angular_velocity) for differential drive
        """
        robot_x, robot_y, robot_theta = robot_state
        
        if current_target_idx >= len(reference_path):
            return (0.0, 0.0)
        
        # 找到距离机器人前轮最近的路径点
        front_axle_pos = self._get_front_axle_position(robot_state)
        closest_idx = self._find_closest_point_index(front_axle_pos, reference_path)
        
        if closest_idx >= len(reference_path) - 1:
            closest_idx = len(reference_path) - 2
        
        # 计算路径切线方向
        path_heading = self._calculate_path_heading(reference_path, closest_idx)
        
        # 计算航向误差
        heading_error = robot_theta - path_heading
        
        # 将角度差归一化到[-π, π]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # 计算横向误差
        crosstrack_error = self._calculate_crosstrack_error(
            front_axle_pos, reference_path, closest_idx, path_heading
        )
        
        # 计算期望速度（可以根据路径曲率调整）
        desired_speed = self._calculate_desired_speed(reference_path, closest_idx)
        
        # Stanley控制律
        crosstrack_term = np.arctan2(self.k_crosstrack * crosstrack_error, 
                                   desired_speed + self.k_soft)
        
        # 总的转向角
        steering_angle = heading_error + crosstrack_term
        
        # 限制转向角
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        # 记录控制详情
        control_details = {
            'front_axle_pos': front_axle_pos,
            'closest_idx': closest_idx,
            'path_heading': path_heading,
            'heading_error': heading_error,
            'crosstrack_error': crosstrack_error,
            'crosstrack_term': crosstrack_term,
            'steering_angle': steering_angle,
            'desired_speed': desired_speed
        }
        self.control_history.append(control_details)
        
        # 如果是差分驱动机器人，转换为线速度和角速度
        if hasattr(self, '_convert_to_differential_drive'):
            return self._convert_to_differential_drive(desired_speed, steering_angle)
        
        return (desired_speed, steering_angle)
    
    def _get_front_axle_position(self, robot_state: tuple) -> tuple:
        """获取前轮轴位置"""
        robot_x, robot_y, robot_theta = robot_state
        
        # 对于差分驱动机器人，使用机器人中心作为参考点
        # 而不是前轮轴，因为差分驱动没有真正的前轮轴
        front_x = robot_x + 0.3 * np.cos(robot_theta)
        front_y = robot_y + 0.3 * np.sin(robot_theta)
        
        return (front_x, front_y)
    
    def _find_closest_point_index(self, position: tuple, reference_path: list) -> int:
        """找到距离给定位置最近的路径点索引"""
        min_distance = float('inf')
        closest_idx = 0
        
        for i, path_point in enumerate(reference_path):
            distance = np.sqrt((position[0] - path_point[0])**2 + 
                             (position[1] - path_point[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        return closest_idx
    
    def _calculate_path_heading(self, reference_path: list, index: int) -> float:
        """计算路径在给定点的切线方向"""
        if index >= len(reference_path) - 1:
            index = len(reference_path) - 2
        
        if index < 0:
            index = 0
        
        # 使用相邻点计算切线方向
        p1 = reference_path[index]
        p2 = reference_path[index + 1]
        
        path_heading = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        return path_heading
    
    def _calculate_crosstrack_error(self, position: tuple, reference_path: list, 
                                  closest_idx: int, path_heading: float) -> float:
        """计算横向误差"""
        if closest_idx >= len(reference_path):
            closest_idx = len(reference_path) - 1
        
        closest_point = reference_path[closest_idx]
        
        # 计算从最近路径点到前轮轴的向量
        dx = position[0] - closest_point[0]
        dy = position[1] - closest_point[1]
        
        # 计算横向误差（垂直于路径方向的误差）
        # 使用叉积判断误差方向
        crosstrack_error = -dx * np.sin(path_heading) + dy * np.cos(path_heading)
        
        return crosstrack_error
    
    def _calculate_desired_speed(self, reference_path: list, index: int) -> float:
        """根据路径曲率计算期望速度"""
        base_speed = 1.5  # 基础速度
        min_speed = 0.8   # 最小速度
        
        # 计算路径曲率
        curvature = self._calculate_path_curvature(reference_path, index)
        
        # 根据曲率调整速度
        if abs(curvature) > 0.05:
            speed_factor = 1.0 / (1.0 + abs(curvature) * 3.0)
            desired_speed = min_speed + (base_speed - min_speed) * speed_factor
        else:
            desired_speed = base_speed
        
        return desired_speed
    
    def _calculate_path_curvature(self, reference_path: list, index: int) -> float:
        """计算路径曲率"""
        if len(reference_path) < 3 or index < 1 or index >= len(reference_path) - 1:
            return 0.0
        
        # 使用三点法计算曲率
        p1 = np.array(reference_path[index - 1])
        p2 = np.array(reference_path[index])
        p3 = np.array(reference_path[index + 1])
        
        # 计算向量
        v1 = p2 - p1
        v2 = p3 - p2
        
        # 计算叉积和点积
        cross_product = np.cross(v1, v2)
        dot_product = np.dot(v1, v2)
        
        # 计算夹角
        magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if magnitude_product < 1e-6:
            return 0.0
        
        # 曲率近似
        curvature = 2 * cross_product / (magnitude_product * 
                    (np.linalg.norm(v1) + np.linalg.norm(v2)))
        
        return curvature
    
    def _convert_to_differential_drive(self, speed: float, steering_angle: float) -> tuple:
        """将Ackermann控制转换为差分驱动控制"""
        # 使用近似转换
        linear_velocity = speed
        
        # 根据转向角计算角速度
        if abs(steering_angle) < 1e-6:
            angular_velocity = 0.0
        else:
            # 转向半径，限制最小转向半径避免数值问题
            min_turning_radius = 0.2
            turning_radius = max(min_turning_radius, 
                               abs(self.wheelbase / np.tan(abs(steering_angle))))
            angular_velocity = speed / turning_radius
            
            # 保持转向方向
            if steering_angle < 0:
                angular_velocity = -angular_velocity
        
        # 限制角速度
        max_angular_velocity = np.pi / 4  # 减小最大角速度
        angular_velocity = np.clip(angular_velocity, -max_angular_velocity, max_angular_velocity)
        
        return (linear_velocity, angular_velocity)
    
    def track_path(self, robot, reference_path: list, 
                   dt: float = 0.1, max_time: float = 100.0, 
                   goal_tolerance: float = 0.5) -> tuple:
        """重写路径跟踪方法以重置控制历史"""
        # 重置控制历史
        self.control_history.clear()
        
        # 调用父类方法
        return super().track_path(robot, reference_path, dt, max_time, goal_tolerance)
    
    def visualize_control_details(self, robot_state: tuple, 
                                 reference_path: list, 
                                 ax=None):
        """可视化Stanley控制细节"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        robot_x, robot_y, robot_theta = robot_state
        
        # 绘制参考路径
        if reference_path:
            path_array = np.array(reference_path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'r--', linewidth=2, label='Reference Path')
        
        # 绘制机器人
        ax.plot(robot_x, robot_y, 'bo', markersize=10, label='Robot Center')
        
        # 绘制机器人朝向
        arrow_length = 1.0
        ax.arrow(robot_x, robot_y, 
                arrow_length * np.cos(robot_theta), 
                arrow_length * np.sin(robot_theta),
                head_width=0.3, head_length=0.2, fc='blue', ec='blue')
        
        # 绘制前轮轴位置
        front_pos = self._get_front_axle_position(robot_state)
        ax.plot(front_pos[0], front_pos[1], 'go', markersize=8, label='Front Axle')
        
        # 如果有控制历史，显示最新的控制信息
        if self.control_history:
            latest_control = self.control_history[-1]
            
            # 绘制最近路径点
            closest_idx = latest_control['closest_idx']
            if closest_idx < len(reference_path):
                closest_point = reference_path[closest_idx]
                ax.plot(closest_point[0], closest_point[1], 'mo', 
                       markersize=8, label='Closest Path Point')
                
                # 绘制横向误差线
                ax.plot([front_pos[0], closest_point[0]], 
                       [front_pos[1], closest_point[1]], 
                       'm-', linewidth=2, alpha=0.7, label='Crosstrack Error')
            
            # 显示控制参数
            ax.text(0.02, 0.98, f"Crosstrack Error: {latest_control['crosstrack_error']:.3f}m\n" +
                              f"Heading Error: {latest_control['heading_error']:.3f}rad\n" +
                              f"Steering Angle: {latest_control['steering_angle']:.3f}rad",
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Stanley Controller Details')
        
        return ax

def test_stanley():
    """测试Stanley控制器"""
    try:
        from src.environment.scenario import create_scenario_1
        from src.planning.a_star import AStarPlanner
        from src.robot.kinematic_model import DifferentialDriveRobot
        
        # 创建环境和规划器
        env = create_scenario_1()
        planner = AStarPlanner(resolution=0.3)
        
        # 规划路径
        path, _ = planner.plan_with_metrics(env.start, env.goal, env)
        
        if path is None or not path:
            print("Path planning failed!")
            return
        
        print(f"Path planned with {len(path)} points")
        
        # 创建机器人和控制器
        robot = DifferentialDriveRobot(wheelbase=0.5, max_speed=2.0)
        controller = StanleyController(k_crosstrack=1.0, wheelbase=0.5)
        
        # 设置转换函数
        controller._convert_to_differential_drive = controller._convert_to_differential_drive
        
        # 执行路径跟踪
        trajectory, metrics = controller.track_path(robot, path, dt=0.1, max_time=50.0)
        
        print(f"Tracking success: {metrics.success}")
        print(f"Tracking time: {metrics.tracking_time:.2f}s")
        print(f"Max lateral error: {metrics.max_lateral_error:.3f}")
        print(f"RMS lateral error: {metrics.rms_lateral_error:.3f}")
        
        # 可视化结果
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 跟踪结果
        controller.visualize_tracking_result(trajectory, path, env, ax=ax1)
        
        # 控制细节（最后状态）
        if trajectory:
            last_state = trajectory[-1]
            robot_state = (last_state['position'][0], last_state['position'][1], 
                          last_state['orientation'])
            controller.visualize_control_details(robot_state, path, ax=ax2)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError as e:
        print(f"Could not import required modules: {e}")

if __name__ == "__main__":
    test_stanley()
