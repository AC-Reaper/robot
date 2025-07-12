"""
PID 路径跟踪控制器
基于比例-积分-微分控制的路径跟踪方法
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.control.base_controller import BaseController

class PIDController(BaseController):
    """PID 控制器"""
    
    def __init__(self, kp_lateral: float = 1.5, ki_lateral: float = 0.02, kd_lateral: float = 0.8,
                 kp_heading: float = 2.0, ki_heading: float = 0.01, kd_heading: float = 0.5,
                 max_integral: float = 0.5, max_speed: float = 1.5, min_speed: float = 0.5):
        """
        初始化PID控制器
        
        Args:
            kp_lateral: 横向误差比例增益
            ki_lateral: 横向误差积分增益
            kd_lateral: 横向误差微分增益
            kp_heading: 航向误差比例增益
            ki_heading: 航向误差积分增益
            kd_heading: 航向误差微分增益
            max_integral: 最大积分值（防止积分饱和）
            max_speed: 最大速度
            min_speed: 最小速度
        """
        super().__init__("PID")
        
        # 横向误差PID参数
        self.kp_lateral = kp_lateral
        self.ki_lateral = ki_lateral
        self.kd_lateral = kd_lateral
        
        # 航向误差PID参数
        self.kp_heading = kp_heading
        self.ki_heading = ki_heading
        self.kd_heading = kd_heading
        
        # 其他参数
        self.max_integral = max_integral
        self.max_speed = max_speed
        self.min_speed = min_speed
        
        # PID状态变量
        self.reset_pid_state()
        
        # 记录控制详情（用于可视化）
        self.control_history = []
        
    def reset_pid_state(self):
        """重置PID状态"""
        # 横向误差PID状态
        self.lateral_error_integral = 0.0
        self.lateral_error_prev = 0.0
        
        # 航向误差PID状态
        self.heading_error_integral = 0.0
        self.heading_error_prev = 0.0
        
        # 时间记录
        self.prev_time = None
        
    def compute_control(self, robot_state: tuple, 
                       reference_path: list, 
                       current_target_idx: int, 
                       dt: float) -> tuple:
        """
        计算PID控制输入
        
        Returns:
            (linear_velocity, angular_velocity) for differential drive
        """
        robot_x, robot_y, robot_theta = robot_state
        
        if current_target_idx >= len(reference_path):
            return (0.0, 0.0)
        
        # 找到距离机器人最近的路径点
        closest_idx = self._find_closest_point_index((robot_x, robot_y), reference_path)
        
        # 计算目标点（可以是当前最近点的前几个点）
        lookahead_idx = min(closest_idx + 2, len(reference_path) - 1)
        target_point = reference_path[lookahead_idx]
        
        # 计算横向误差和航向误差
        lateral_error, heading_error = self._calculate_errors(
            robot_state, reference_path, closest_idx
        )
        
        # 计算PID控制输出
        lateral_control = self._compute_lateral_pid(lateral_error, dt)
        heading_control = self._compute_heading_pid(heading_error, dt)
        
        # 计算期望速度（基于横向误差）
        speed_factor = 1.0 / (1.0 + abs(lateral_error) * 1.0)
        desired_speed = self.min_speed + (self.max_speed - self.min_speed) * speed_factor
        
        # 计算角速度（结合横向和航向控制）
        angular_velocity = lateral_control + heading_control
        
        # 限制角速度
        max_angular_velocity = np.pi / 4  # 减小最大角速度
        angular_velocity = np.clip(angular_velocity, -max_angular_velocity, max_angular_velocity)
        
        # 记录控制详情
        control_details = {
            'closest_idx': closest_idx,
            'target_point': target_point,
            'lateral_error': lateral_error,
            'heading_error': heading_error,
            'lateral_control': lateral_control,
            'heading_control': heading_control,
            'desired_speed': desired_speed,
            'angular_velocity': angular_velocity,
            'lateral_integral': self.lateral_error_integral,
            'heading_integral': self.heading_error_integral
        }
        self.control_history.append(control_details)
        
        return (desired_speed, angular_velocity)
    
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
    
    def _calculate_errors(self, robot_state: tuple, reference_path: list, 
                         closest_idx: int) -> tuple:
        """计算横向误差和航向误差"""
        robot_x, robot_y, robot_theta = robot_state
        
        # 横向误差：机器人到参考路径的距离
        if closest_idx < len(reference_path) - 1:
            # 使用当前点和下一个点构成的线段
            p1 = np.array(reference_path[closest_idx])
            p2 = np.array(reference_path[closest_idx + 1])
            robot_pos = np.array([robot_x, robot_y])
            
            # 计算点到线段的距离
            line_vec = p2 - p1
            if np.linalg.norm(line_vec) > 1e-6:
                line_vec_normalized = line_vec / np.linalg.norm(line_vec)
                robot_vec = robot_pos - p1
                projection = np.dot(robot_vec, line_vec_normalized)
                projection = max(0, min(projection, np.linalg.norm(line_vec)))
                closest_point = p1 + projection * line_vec_normalized
                
                # 计算横向误差的符号
                cross_product = np.cross(line_vec, robot_vec)
                lateral_error = np.linalg.norm(robot_pos - closest_point)
                lateral_error = lateral_error if cross_product >= 0 else -lateral_error
            else:
                lateral_error = np.linalg.norm(robot_pos - p1)
        else:
            # 最后一个点
            target_pos = reference_path[closest_idx]
            lateral_error = np.sqrt((robot_x - target_pos[0])**2 + 
                                  (robot_y - target_pos[1])**2)
        
        # 航向误差：机器人朝向与路径切线的角度差
        if closest_idx < len(reference_path) - 1:
            p1 = reference_path[closest_idx]
            p2 = reference_path[closest_idx + 1]
            path_heading = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        else:
            # 使用前一个点计算方向
            if closest_idx > 0:
                p1 = reference_path[closest_idx - 1]
                p2 = reference_path[closest_idx]
                path_heading = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            else:
                path_heading = robot_theta
        
        heading_error = robot_theta - path_heading
        
        # 将角度差归一化到[-π, π]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        return lateral_error, heading_error
    
    def _compute_lateral_pid(self, lateral_error: float, dt: float) -> float:
        """计算横向误差的PID控制输出"""
        # 比例项
        proportional = self.kp_lateral * lateral_error
        
        # 积分项
        self.lateral_error_integral += lateral_error * dt
        # 积分饱和保护
        self.lateral_error_integral = np.clip(self.lateral_error_integral, 
                                            -self.max_integral, self.max_integral)
        integral = self.ki_lateral * self.lateral_error_integral
        
        # 微分项
        derivative = 0.0
        if dt > 0:
            derivative = self.kd_lateral * (lateral_error - self.lateral_error_prev) / dt
        
        # 更新前一次误差
        self.lateral_error_prev = lateral_error
        
        return proportional + integral + derivative
    
    def _compute_heading_pid(self, heading_error: float, dt: float) -> float:
        """计算航向误差的PID控制输出"""
        # 比例项
        proportional = self.kp_heading * heading_error
        
        # 积分项
        self.heading_error_integral += heading_error * dt
        # 积分饱和保护
        self.heading_error_integral = np.clip(self.heading_error_integral, 
                                            -self.max_integral, self.max_integral)
        integral = self.ki_heading * self.heading_error_integral
        
        # 微分项
        derivative = 0.0
        if dt > 0:
            derivative = self.kd_heading * (heading_error - self.heading_error_prev) / dt
        
        # 更新前一次误差
        self.heading_error_prev = heading_error
        
        return proportional + integral + derivative
    
    def track_path(self, robot, reference_path: list, 
                   dt: float = 0.1, max_time: float = 100.0, 
                   goal_tolerance: float = 0.5) -> tuple:
        """重写路径跟踪方法以重置PID状态"""
        # 重置PID状态
        self.reset_pid_state()
        
        # 调用父类方法
        return super().track_path(robot, reference_path, dt, max_time, goal_tolerance)
        self.reset_pid_state()
        
        # 调用父类方法
        return super().track_path(robot, reference_path, dt, max_time, goal_tolerance)
    
    def visualize_control_details(self, robot_state: tuple, 
                                 reference_path: list, 
                                 ax=None):
        """可视化PID控制细节"""
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
        
        # 如果有控制历史，显示最新的控制信息
        if self.control_history:
            latest_control = self.control_history[-1]
            
            # 绘制目标点
            target_point = latest_control['target_point']
            ax.plot(target_point[0], target_point[1], 'go', 
                   markersize=8, label='Target Point')
            
            # 绘制最近路径点
            closest_idx = latest_control['closest_idx']
            if closest_idx < len(reference_path):
                closest_point = reference_path[closest_idx]
                ax.plot(closest_point[0], closest_point[1], 'mo', 
                       markersize=8, label='Closest Path Point')
            
            # 显示控制参数
            ax.text(0.02, 0.98, 
                   f"Lateral Error: {latest_control['lateral_error']:.3f}m\n" +
                   f"Heading Error: {latest_control['heading_error']:.3f}rad\n" +
                   f"Lateral Control: {latest_control['lateral_control']:.3f}\n" +
                   f"Heading Control: {latest_control['heading_control']:.3f}\n" +
                   f"Angular Velocity: {latest_control['angular_velocity']:.3f}rad/s",
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('PID Controller Details')
        
        return ax
    
    def plot_pid_response(self, save_path=None):
        """绘制PID响应曲线"""
        if not self.control_history:
            print("No control history available")
            return
        
        time_steps = np.arange(len(self.control_history)) * 0.1
        
        lateral_errors = [c['lateral_error'] for c in self.control_history]
        heading_errors = [c['heading_error'] for c in self.control_history]
        lateral_controls = [c['lateral_control'] for c in self.control_history]
        heading_controls = [c['heading_control'] for c in self.control_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 横向误差
        axes[0, 0].plot(time_steps, lateral_errors, 'b-', linewidth=2)
        axes[0, 0].set_title('Lateral Error')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Error (m)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 航向误差
        axes[0, 1].plot(time_steps, heading_errors, 'r-', linewidth=2)
        axes[0, 1].set_title('Heading Error')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Error (rad)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 横向控制输出
        axes[1, 0].plot(time_steps, lateral_controls, 'g-', linewidth=2)
        axes[1, 0].set_title('Lateral Control Output')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Control Signal')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 航向控制输出
        axes[1, 1].plot(time_steps, heading_controls, 'm-', linewidth=2)
        axes[1, 1].set_title('Heading Control Output')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Control Signal')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def test_pid():
    """测试PID控制器"""
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
        controller = PIDController(
            kp_lateral=2.0, ki_lateral=0.1, kd_lateral=0.5,
            kp_heading=1.5, ki_heading=0.05, kd_heading=0.3
        )
        
        # 执行路径跟踪
        trajectory, metrics = controller.track_path(robot, path, dt=0.1, max_time=50.0)
        
        print(f"Tracking success: {metrics.success}")
        print(f"Tracking time: {metrics.tracking_time:.2f}s")
        print(f"Max lateral error: {metrics.max_lateral_error:.3f}")
        print(f"RMS lateral error: {metrics.rms_lateral_error:.3f}")
        
        # 可视化结果
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 跟踪结果
        controller.visualize_tracking_result(trajectory, path, env, ax=axes[0, 0])
        
        # 控制细节（最后状态）
        if trajectory:
            last_state = trajectory[-1]
            robot_state = (last_state['position'][0], last_state['position'][1], 
                          last_state['orientation'])
            controller.visualize_control_details(robot_state, path, ax=axes[0, 1])
        
        # PID响应曲线
        controller.plot_pid_response()
        
        plt.tight_layout()
        plt.show()
        
    except ImportError as e:
        print(f"Could not import required modules: {e}")

if __name__ == "__main__":
    test_pid()
