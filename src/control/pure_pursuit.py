"""
Pure Pursuit 路径跟踪控制器
基于几何跟踪的简单有效控制方法
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.control.base_controller import BaseController

class PurePursuitController(BaseController):
    """Pure Pursuit 控制器"""
    
    def __init__(self, lookahead_distance: float = 2.0, 
                 min_lookahead: float = 0.5, 
                 max_lookahead: float = 5.0,
                 speed_gain: float = 0.1):
        """
        初始化Pure Pursuit控制器
        
        Args:
            lookahead_distance: 前视距离
            min_lookahead: 最小前视距离
            max_lookahead: 最大前视距离
            speed_gain: 速度增益
        """
        super().__init__("Pure Pursuit")
        self.lookahead_distance = lookahead_distance
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.speed_gain = speed_gain
        
        # 记录前视点历史（用于可视化）
        self.lookahead_points_history = []
        
    def compute_control(self, robot_state: tuple, 
                       reference_path: list, 
                       current_target_idx: int, 
                       dt: float) -> tuple:
        """
        计算Pure Pursuit控制输入
        
        Returns:
            (linear_velocity, angular_velocity) for differential drive
            or (acceleration, steering_angle) for Ackermann
        """
        robot_x, robot_y, robot_theta = robot_state
        
        # 根据速度自适应调整前视距离
        current_speed = getattr(self, '_last_speed', 1.0)
        adaptive_lookahead = np.clip(
            self.lookahead_distance + self.speed_gain * current_speed,
            self.min_lookahead, 
            self.max_lookahead
        )
        
        # 寻找前视点
        lookahead_point = self._find_lookahead_point(
            robot_state, reference_path, current_target_idx, adaptive_lookahead
        )
        
        if lookahead_point is None:
            # 如果找不到前视点，停止
            return (0.0, 0.0)
        
        # 记录前视点（用于可视化）
        self.lookahead_points_history.append(lookahead_point)
        
        # 计算前视点相对于机器人的位置
        dx = lookahead_point[0] - robot_x
        dy = lookahead_point[1] - robot_y
        
        # 转换到机器人坐标系
        cos_theta = np.cos(robot_theta)
        sin_theta = np.sin(robot_theta)
        
        # 机器人坐标系下的前视点
        local_x = dx * cos_theta + dy * sin_theta
        local_y = -dx * sin_theta + dy * cos_theta
        
        # Pure Pursuit几何计算
        distance_to_lookahead = np.sqrt(dx*dx + dy*dy)
        
        if distance_to_lookahead < 1e-3:
            return (0.0, 0.0)
        
        # 计算曲率
        curvature = 2 * local_y / (distance_to_lookahead**2)
        
        # 计算期望速度（基于曲率调整）
        max_speed = 2.0
        min_speed = 0.5
        
        # 根据曲率调整速度
        speed_factor = 1.0 / (1.0 + abs(curvature) * 2.0)
        desired_speed = min_speed + (max_speed - min_speed) * speed_factor
        
        # 保存当前速度（用于下次计算自适应前视距离）
        self._last_speed = desired_speed
        
        # 计算角速度
        angular_velocity = curvature * desired_speed
        
        # 限制角速度
        max_angular_velocity = np.pi / 2
        angular_velocity = np.clip(angular_velocity, -max_angular_velocity, max_angular_velocity)
        
        return (desired_speed, angular_velocity)
    
    def _find_lookahead_point(self, robot_state: tuple, 
                             reference_path: list, 
                             start_idx: int, 
                             lookahead_distance: float) -> tuple:
        """寻找前视点"""
        robot_x, robot_y, _ = robot_state
        
        if start_idx >= len(reference_path):
            return reference_path[-1] if reference_path else None
        
        # 从当前目标点开始搜索
        for i in range(start_idx, len(reference_path)):
            path_point = reference_path[i]
            distance = np.sqrt((path_point[0] - robot_x)**2 + (path_point[1] - robot_y)**2)
            
            if distance >= lookahead_distance:
                if i == 0:
                    return path_point
                
                # 在两个路径点之间插值获得精确的前视点
                prev_point = reference_path[i-1]
                prev_distance = np.sqrt((prev_point[0] - robot_x)**2 + (prev_point[1] - robot_y)**2)
                
                if prev_distance < lookahead_distance:
                    # 线性插值
                    segment_length = np.sqrt((path_point[0] - prev_point[0])**2 + 
                                           (path_point[1] - prev_point[1])**2)
                    
                    if segment_length > 1e-6:
                        # 使用几何方法找到圆与线段的交点
                        lookahead_point = self._circle_line_intersection(
                            (robot_x, robot_y), lookahead_distance,
                            prev_point, path_point
                        )
                        
                        if lookahead_point:
                            return lookahead_point
                
                return path_point
        
        # 如果没有找到合适的点，返回路径终点
        return reference_path[-1]
    
    def _circle_line_intersection(self, center: tuple, radius: float, 
                                 line_start: tuple, line_end: tuple) -> tuple:
        """计算圆与线段的交点"""
        cx, cy = center
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 线段向量
        dx = x2 - x1
        dy = y2 - y1
        
        # 从线段起点到圆心的向量
        fx = x1 - cx
        fy = y1 - cy
        
        # 二次方程系数 at^2 + bt + c = 0
        a = dx*dx + dy*dy
        b = 2*(fx*dx + fy*dy)
        c = (fx*fx + fy*fy) - radius*radius
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return None  # 无交点
        
        # 计算两个解
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)
        
        # 选择在线段上且更远的交点
        valid_ts = [t for t in [t1, t2] if 0 <= t <= 1]
        
        if not valid_ts:
            return None
        
        # 选择更远的交点（更大的t值）
        t = max(valid_ts)
        
        # 计算交点坐标
        intersection_x = x1 + t * dx
        intersection_y = y1 + t * dy
        
        return (intersection_x, intersection_y)
    
    def visualize_control_details(self, robot_state: tuple, 
                                 reference_path: list, 
                                 current_target_idx: int, 
                                 ax=None):
        """可视化Pure Pursuit控制细节"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        robot_x, robot_y, robot_theta = robot_state
        
        # 绘制参考路径
        if reference_path:
            path_array = np.array(reference_path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'r--', linewidth=2, label='Reference Path')
        
        # 绘制机器人
        ax.plot(robot_x, robot_y, 'bo', markersize=10, label='Robot')
        
        # 绘制机器人朝向
        arrow_length = 1.0
        ax.arrow(robot_x, robot_y, 
                arrow_length * np.cos(robot_theta), 
                arrow_length * np.sin(robot_theta),
                head_width=0.3, head_length=0.2, fc='blue', ec='blue')
        
        # 绘制前视圆
        current_speed = getattr(self, '_last_speed', 1.0)
        adaptive_lookahead = np.clip(
            self.lookahead_distance + self.speed_gain * current_speed,
            self.min_lookahead, 
            self.max_lookahead
        )
        
        circle = plt.Circle((robot_x, robot_y), adaptive_lookahead, 
                          fill=False, color='green', linestyle='--', alpha=0.7)
        ax.add_patch(circle)
        
        # 绘制前视点
        lookahead_point = self._find_lookahead_point(
            robot_state, reference_path, current_target_idx, adaptive_lookahead
        )
        
        if lookahead_point:
            ax.plot(lookahead_point[0], lookahead_point[1], 'go', 
                   markersize=8, label='Lookahead Point')
            
            # 绘制从机器人到前视点的连线
            ax.plot([robot_x, lookahead_point[0]], [robot_y, lookahead_point[1]], 
                   'g-', linewidth=2, alpha=0.7)
        
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Pure Pursuit Control (Lookahead: {adaptive_lookahead:.2f})')
        
        return ax

def test_pure_pursuit():
    """测试Pure Pursuit控制器"""
    try:
        from src.environment.scenario import create_scenario_1
        from src.planning.a_star import AStarPlanner
        from src.robot.kinematic_model import DifferentialDriveRobot
        
        # 创建环境和规划器
        env = create_scenario_1()
        planner = AStarPlanner(resolution=0.3)
        
        # 规划路径
        path, _ = planner.plan_with_metrics(env.start, env.goal, env)
        
        if not path:
            print("Path planning failed!")
            return
        
        print(f"Path planned with {len(path)} points")
        
        # 创建机器人和控制器
        robot = DifferentialDriveRobot(wheelbase=0.5, max_speed=2.0)
        controller = PurePursuitController(lookahead_distance=1.5)
        
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
            controller.visualize_control_details(robot_state, path, 
                                               last_state['target_idx'], ax=ax2)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError as e:
        print(f"Could not import required modules: {e}")

if __name__ == "__main__":
    test_pure_pursuit()
