"""
人工势场法 (Artificial Potential Field) 路径规划算法
基于引力和斥力的实时路径规划方法
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.planning.base_planner import BasePlanner

class PotentialFieldPlanner(BasePlanner):
    """人工势场法路径规划器"""
    
    def __init__(self, attractive_gain: float = 1.0, repulsive_gain: float = 2.0,
                 influence_distance: float = 2.0, step_size: float = 0.1,
                 max_iterations: int = 100000, goal_tolerance: float = 0.3):
        """
        初始化势场规划器
        
        Args:
            attractive_gain: 引力增益
            repulsive_gain: 斥力增益  
            influence_distance: 障碍物影响距离
            step_size: 步长
            max_iterations: 最大迭代次数
            goal_tolerance: 目标容差
        """
        super().__init__("Potential Field")
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.influence_distance = influence_distance
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_tolerance = goal_tolerance
        
        # 可视化数据
        self.path_history = []
        self.force_history = []
        self.potential_field = None
        
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], 
             environment) -> List[Tuple[float, float]]:
        """
        势场法路径规划主函数
        """
        # 清空历史记录
        self.path_history.clear()
        self.force_history.clear()
        
        # 初始化当前位置
        current_pos = np.array(start, dtype=float)
        goal_pos = np.array(goal, dtype=float)
        
        path = [tuple(current_pos)]
        
        for iteration in range(self.max_iterations):
            # 计算总合力
            attractive_force = self._calculate_attractive_force(current_pos, goal_pos)
            repulsive_force = self._calculate_repulsive_force(current_pos, environment)
            total_force = attractive_force + repulsive_force
            
            # 记录历史
            self.path_history.append(current_pos.copy())
            self.force_history.append({
                'attractive': attractive_force,
                'repulsive': repulsive_force,
                'total': total_force
            })
            
            # 检查是否到达目标
            distance_to_goal = np.linalg.norm(current_pos - goal_pos)
            if distance_to_goal < self.goal_tolerance:
                path.append(tuple(goal_pos))
                self.metrics.num_nodes_explored = iteration + 1
                self.metrics.iterations = iteration + 1
                return path
            
            # 检查是否陷入局部最优（力很小但距离目标还很远）
            force_magnitude = np.linalg.norm(total_force)
            if force_magnitude < 1e-6 and distance_to_goal > self.goal_tolerance:
                print("Potential field planner stuck in local minimum")
                break
            
            # 归一化力向量并移动
            if force_magnitude > 0:
                direction = total_force / force_magnitude
                new_pos = current_pos + self.step_size * direction
                
                # 检查新位置是否有效
                if not environment.is_collision(tuple(new_pos)):
                    current_pos = new_pos
                    path.append(tuple(current_pos))
                else:
                    # 如果新位置有碰撞，尝试沿着障碍物边缘移动
                    tangent_direction = self._get_tangent_direction(current_pos, total_force, environment)
                    if tangent_direction is not None:
                        new_pos = current_pos + self.step_size * tangent_direction
                        if not environment.is_collision(tuple(new_pos)):
                            current_pos = new_pos
                            path.append(tuple(current_pos))
        
        # 规划失败或未到达目标
        self.metrics.num_nodes_explored = len(path)
        self.metrics.iterations = self.max_iterations
        print(f"Potential field planning did not reach goal. Final distance: {distance_to_goal:.3f}")
        return path
    
    def _calculate_attractive_force(self, current_pos: np.ndarray, 
                                  goal_pos: np.ndarray) -> np.ndarray:
        """计算引力"""
        direction_to_goal = goal_pos - current_pos
        distance = np.linalg.norm(direction_to_goal)
        
        if distance < 1e-6:
            return np.zeros(2)
        
        # 线性引力模型
        force_magnitude = self.attractive_gain * distance
        force_direction = direction_to_goal / distance
        
        return force_magnitude * force_direction
    
    def _calculate_repulsive_force(self, current_pos: np.ndarray, 
                                 environment) -> np.ndarray:
        """计算斥力"""
        total_repulsive_force = np.zeros(2)
        
        for obstacle in environment.obstacles:
            # 计算到障碍物的最近距离和方向
            min_distance, closest_point = self._distance_and_direction_to_obstacle(
                current_pos, obstacle
            )
            
            if min_distance < self.influence_distance:
                # 计算斥力大小
                if min_distance < 1e-6:
                    # 如果距离过小，施加很大的斥力
                    force_magnitude = 1e6
                else:
                    force_magnitude = self.repulsive_gain * (
                        1.0 / min_distance - 1.0 / self.influence_distance
                    ) / (min_distance ** 2)
                
                # 斥力方向：从障碍物指向当前位置
                if min_distance > 1e-6:
                    force_direction = (current_pos - closest_point) / min_distance
                else:
                    # 如果距离太小，随机选择一个方向
                    force_direction = np.array([1.0, 0.0])
                
                total_repulsive_force += force_magnitude * force_direction
        
        return total_repulsive_force
    
    def _distance_and_direction_to_obstacle(self, point: np.ndarray, 
                                          obstacle) -> Tuple[float, np.ndarray]:
        """计算点到障碍物的距离和最近点"""
        min_distance = float('inf')
        closest_point = point.copy()
        
        # 计算到障碍物各边的距离
        vertices = obstacle.vertices
        n = len(vertices)
        
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            
            # 计算点到线段的距离和最近点
            distance, nearest_point = self._point_to_line_segment_distance(point, p1, p2)
            
            if distance < min_distance:
                min_distance = distance
                closest_point = nearest_point
        
        return min_distance, closest_point
    
    def _point_to_line_segment_distance(self, point: np.ndarray, 
                                      line_start: np.ndarray, 
                                      line_end: np.ndarray) -> Tuple[float, np.ndarray]:
        """计算点到线段的距离和最近点"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_length_sq = np.dot(line_vec, line_vec)
        
        if line_length_sq < 1e-6:
            # 线段退化为点
            return np.linalg.norm(point - line_start), line_start
        
        # 投影参数
        t = np.dot(point_vec, line_vec) / line_length_sq
        t = max(0.0, min(1.0, t))  # 限制在[0,1]范围内
        
        # 最近点
        closest_point = line_start + t * line_vec
        distance = np.linalg.norm(point - closest_point)
        
        return distance, closest_point
    
    def _get_tangent_direction(self, current_pos: np.ndarray, 
                             desired_force: np.ndarray, 
                             environment) -> np.ndarray:
        """当无法直接移动时，获取切向移动方向"""
        # 找到最近的障碍物
        min_distance = float('inf')
        closest_obstacle = None
        
        for obstacle in environment.obstacles:
            distance, _ = self._distance_and_direction_to_obstacle(current_pos, obstacle)
            if distance < min_distance:
                min_distance = distance
                closest_obstacle = obstacle
        
        if closest_obstacle is None:
            return None
        
        # 计算切向量（垂直于到障碍物的方向）
        _, closest_point = self._distance_and_direction_to_obstacle(current_pos, closest_obstacle)
        normal_direction = current_pos - closest_point
        
        if np.linalg.norm(normal_direction) < 1e-6:
            return None
        
        normal_direction = normal_direction / np.linalg.norm(normal_direction)
        
        # 两个可能的切向量
        tangent1 = np.array([-normal_direction[1], normal_direction[0]])
        tangent2 = np.array([normal_direction[1], -normal_direction[0]])
        
        # 选择与期望方向更接近的切向量
        dot1 = np.dot(tangent1, desired_force)
        dot2 = np.dot(tangent2, desired_force)
        
        return tangent1 if dot1 > dot2 else tangent2
    
    def visualize_potential_field(self, environment, start: Tuple[float, float], 
                                goal: Tuple[float, float], resolution: int = 50):
        """可视化势场"""
        x_range = np.linspace(0, environment.width, resolution)
        y_range = np.linspace(0, environment.height, resolution)
        X, Y = np.meshgrid(x_range, y_range)
        
        # 计算每个点的势能
        potential = np.zeros_like(X)
        
        for i in range(resolution):
            for j in range(resolution):
                point = np.array([X[i, j], Y[i, j]])
                
                # 跳过障碍物内部的点
                if environment.is_collision((X[i, j], Y[i, j])):
                    potential[i, j] = np.inf
                    continue
                
                # 引力势能
                goal_distance = np.linalg.norm(point - np.array(goal))
                attractive_potential = 0.5 * self.attractive_gain * goal_distance**2
                
                # 斥力势能
                repulsive_potential = 0
                for obstacle in environment.obstacles:
                    distance, _ = self._distance_and_direction_to_obstacle(point, obstacle)
                    if distance < self.influence_distance:
                        repulsive_potential += 0.5 * self.repulsive_gain * (
                            1.0/distance - 1.0/self.influence_distance
                        )**2
                
                potential[i, j] = attractive_potential + repulsive_potential
        
        self.potential_field = (X, Y, potential)
        return X, Y, potential
    
    def _visualize_algorithm_specific(self, ax, step):
        """势场法特定的可视化"""
        if not self.path_history:
            return
        
        # 绘制路径历史
        if len(self.path_history) > 1:
            path_array = np.array(self.path_history)
            ax.plot(path_array[:, 0], path_array[:, 1], 'b-', 
                   linewidth=2, alpha=0.7, label='Potential Field Path')
        
        # 绘制当前位置
        if step < len(self.path_history):
            current_pos = self.path_history[step]
            ax.scatter(current_pos[0], current_pos[1], c='red', s=100, 
                      marker='o', label='Current Position', zorder=10)
            
            # 绘制力向量
            if step < len(self.force_history):
                forces = self.force_history[step]
                scale = 0.5  # 力向量缩放因子
                
                # 引力
                attractive = forces['attractive'] * scale
                ax.arrow(current_pos[0], current_pos[1], attractive[0], attractive[1],
                        head_width=0.2, head_length=0.1, fc='green', ec='green',
                        alpha=0.7, label='Attractive Force' if step == 0 else "")
                
                # 斥力
                repulsive = forces['repulsive'] * scale
                if np.linalg.norm(repulsive) > 1e-6:
                    ax.arrow(current_pos[0], current_pos[1], repulsive[0], repulsive[1],
                            head_width=0.2, head_length=0.1, fc='red', ec='red',
                            alpha=0.7, label='Repulsive Force' if step == 0 else "")
                
                # 合力
                total = forces['total'] * scale
                ax.arrow(current_pos[0], current_pos[1], total[0], total[1],
                        head_width=0.15, head_length=0.1, fc='blue', ec='blue',
                        alpha=0.9, label='Total Force' if step == 0 else "")
        
        ax.legend()
        ax.set_title(f'Potential Field Method - Step {step + 1}')

def test_potential_field():
    """测试势场法算法"""
    try:
        from src.environment.scenario import create_scenario_1
        
        # 创建测试环境
        env = create_scenario_1()
        
        # 测试势场法
        print("测试势场法算法...")
        pf_planner = PotentialFieldPlanner(
            attractive_gain=1.0,
            repulsive_gain=5.0,
            influence_distance=1.5,
            step_size=0.1
        )
        
        path, metrics = pf_planner.plan_with_metrics(env.start, env.goal, env)
        
        print(f"势场法规划成功: {metrics.success}")
        if path:
            print(f"路径长度: {metrics.path_length:.2f}m")
            print(f"规划时间: {metrics.planning_time:.3f}s")
            print(f"迭代次数: {metrics.iterations}")
        
        # 可视化结果
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 势场可视化
        X, Y, potential = pf_planner.visualize_potential_field(env, env.start, env.goal)
        
        # 限制势能值以便更好地可视化
        potential_clipped = np.clip(potential, 0, 50)
        
        contour = axes[0].contourf(X, Y, potential_clipped, levels=20, alpha=0.6, cmap='viridis')
        plt.colorbar(contour, ax=axes[0], label='Potential Energy')
        
        env.visualize(ax=axes[0], show_grid=False)
        
        if path:
            path_array = np.array(path)
            axes[0].plot(path_array[:, 0], path_array[:, 1], 'r-', 
                        linewidth=3, label='Potential Field Path')
        
        axes[0].set_title('Potential Field Visualization')
        axes[0].legend()
        
        # 规划过程可视化
        pf_planner.visualize_planning_process(env, env.start, env.goal, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
        
    except ImportError as e:
        print(f"Could not import required modules: {e}")

if __name__ == "__main__":
    test_potential_field()
