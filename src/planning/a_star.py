"""
A* 路径规划算法实现
基于启发式搜索的最优路径规划算法
"""

import heapq
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Set
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.planning.base_planner import BasePlanner, Node, reconstruct_path

class AStarPlanner(BasePlanner):
    """A*路径规划算法"""
    
    def __init__(self, resolution: float = 0.5, heuristic_weight: float = 1.0):
        """
        初始化A*规划器
        
        Args:
            resolution: 网格分辨率
            heuristic_weight: 启发式函数权重
        """
        super().__init__("A*")
        self.resolution = resolution
        self.heuristic_weight = heuristic_weight
        
        # 规划过程可视化数据
        self.open_set_history = []
        self.closed_set_history = []
        self.current_node_history = []
        
    def heuristic(self, node1: Node, node2: Node) -> float:
        """启发式函数（欧几里得距离）"""
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def _find_valid_nearby_points(self, point: Tuple[float, float], environment) -> List[Tuple[float, float]]:
        """找到点附近的有效网格点"""
        x, y = point
        candidates = []
        
        # 生成点周围的网格候选点
        for dx in [-self.resolution, 0, self.resolution]:
            for dy in [-self.resolution, 0, self.resolution]:
                # 标准网格对齐
                grid_x = round((x + dx) / self.resolution) * self.resolution
                grid_y = round((y + dy) / self.resolution) * self.resolution
                
                # 检查边界
                if (grid_x >= 0 and grid_x <= environment.width and 
                    grid_y >= 0 and grid_y <= environment.height):
                    
                    # 检查碰撞
                    if not environment.is_collision((grid_x, grid_y)):
                        candidates.append((grid_x, grid_y))
        
        # 如果没有找到候选点，添加原始点本身（如果有效）
        if not candidates and not environment.is_collision(point):
            candidates.append(point)
        
        return candidates
    
    def _connect_to_original_points(self, path: List[Tuple[float, float]], environment) -> List[Tuple[float, float]]:
        """将路径连接到原始起点和终点"""
        if not path:
            return path
        
        final_path = []
        
        # 添加到原始起点的连接
        if hasattr(self, 'original_start'):
            if not environment.is_line_collision(self.original_start, path[0]):
                final_path.append(self.original_start)
        
        # 添加规划的路径
        final_path.extend(path)
        
        # 添加到原始终点的连接
        if hasattr(self, 'original_goal'):
            if not environment.is_line_collision(path[-1], self.original_goal):
                final_path.append(self.original_goal)
        
        return final_path
    
    def get_neighbors(self, node: Node, environment) -> List[Node]:
        """获取节点的邻居节点"""
        neighbors = []
        
        # 8连通（可以对角移动）
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dx, dy in directions:
            new_x = node.x + dx * self.resolution
            new_y = node.y + dy * self.resolution
            
            # 检查边界
            if (new_x >= 0 and new_x <= environment.width and 
                new_y >= 0 and new_y <= environment.height):
                
                # 检查碰撞（包括线段碰撞）
                if (not environment.is_collision((new_x, new_y)) and 
                    not environment.is_line_collision((node.x, node.y), (new_x, new_y))):
                    neighbor = Node(new_x, new_y, node)
                    
                    # 计算移动代价
                    if dx != 0 and dy != 0:  # 对角移动
                        move_cost = self.resolution * np.sqrt(2)
                    else:  # 直线移动
                        move_cost = self.resolution
                    
                    neighbor.g_cost = node.g_cost + move_cost
                    neighbors.append(neighbor)
        
        return neighbors
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], 
             environment) -> List[Tuple[float, float]]:
        """
        A*路径规划主函数
        """
        # 清空历史记录
        self.open_set_history.clear()
        self.closed_set_history.clear()
        self.current_node_history.clear()
        
        # 找到起点和终点附近最好的网格点
        start_candidates = self._find_valid_nearby_points(start, environment)
        goal_candidates = self._find_valid_nearby_points(goal, environment)
        
        if not start_candidates:
            print("No valid start position found!")
            return []
        
        if not goal_candidates:
            print("No valid goal position found!")
            return []
        
        # 选择距离原始点最近的有效点
        start_x, start_y = min(start_candidates, key=lambda p: (p[0]-start[0])**2 + (p[1]-start[1])**2)
        goal_x, goal_y = min(goal_candidates, key=lambda p: (p[0]-goal[0])**2 + (p[1]-goal[1])**2)
        
        # 创建起点和终点节点
        start_node = Node(start_x, start_y)
        goal_node = Node(goal_x, goal_y)
        
        # 存储原始目标用于后处理
        self.original_start = start
        self.original_goal = goal
        
        # 初始化开放集和关闭集
        open_set = []
        closed_set: Set[Node] = set()
        open_set_dict = {}  # 用于快速查找开放集中的节点
        
        # 初始化起点
        start_node.g_cost = 0
        start_node.h_cost = self.heuristic(start_node, goal_node) * self.heuristic_weight
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        heapq.heappush(open_set, start_node)
        open_set_dict[start_node] = start_node
        
        iteration = 0
        max_iterations = 100000
        
        while open_set and iteration < max_iterations:
            iteration += 1
            
            # 获取f值最小的节点
            current_node = heapq.heappop(open_set)
            if current_node in open_set_dict:
                del open_set_dict[current_node]
            
            # 记录当前状态（用于可视化）
            if iteration % 10 == 0:  # 每10次迭代记录一次，减少内存使用
                self.open_set_history.append(list(open_set_dict.keys()).copy())
                self.closed_set_history.append(closed_set.copy())
                self.current_node_history.append(current_node)
            
            # 将当前节点加入关闭集
            closed_set.add(current_node)
            
            # 检查是否到达目标
            if current_node.distance_to(goal_node) < self.resolution:
                self.metrics.num_nodes_explored = len(closed_set)
                self.metrics.iterations = iteration
                # 重构路径并连接到原始目标
                path = reconstruct_path(current_node)
                return self._connect_to_original_points(path, environment)
            
            # 遍历邻居节点
            neighbors = self.get_neighbors(current_node, environment)
            
            for neighbor in neighbors:
                # 跳过已在关闭集中的节点
                if neighbor in closed_set:
                    continue
                
                # 计算邻居节点的启发式代价
                neighbor.h_cost = self.heuristic(neighbor, goal_node) * self.heuristic_weight
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                
                # 检查邻居是否在开放集中
                if neighbor in open_set_dict:
                    existing_neighbor = open_set_dict[neighbor]
                    if neighbor.g_cost < existing_neighbor.g_cost:
                        # 找到更好的路径，更新节点
                        existing_neighbor.g_cost = neighbor.g_cost
                        existing_neighbor.f_cost = neighbor.f_cost
                        existing_neighbor.parent = neighbor.parent
                        heapq.heapify(open_set)  # 重新堆化
                else:
                    # 新节点，加入开放集
                    heapq.heappush(open_set, neighbor)
                    open_set_dict[neighbor] = neighbor
        
        # 规划失败
        self.metrics.num_nodes_explored = len(closed_set)
        self.metrics.iterations = iteration
        print(f"A* planning failed after {iteration} iterations")
        return []
    
    def _visualize_algorithm_specific(self, ax, step):
        """A*算法特定的可视化"""
        if not self.open_set_history:
            return
        
        # 确定显示到哪一步
        if step == -1 or step >= len(self.open_set_history):
            step = len(self.open_set_history) - 1
        
        if step < 0:
            return
        
        # 绘制关闭集（已探索的节点）
        closed_set = self.closed_set_history[step]
        if closed_set:
            closed_x = [node.x for node in closed_set]
            closed_y = [node.y for node in closed_set]
            ax.scatter(closed_x, closed_y, c='lightblue', s=20, alpha=0.6, label='Closed Set')
        
        # 绘制开放集（待探索的节点）
        open_set = self.open_set_history[step]
        if open_set:
            open_x = [node.x for node in open_set]
            open_y = [node.y for node in open_set]
            ax.scatter(open_x, open_y, c='yellow', s=30, alpha=0.8, label='Open Set')
        
        # 绘制当前节点
        if step < len(self.current_node_history):
            current = self.current_node_history[step]
            ax.scatter(current.x, current.y, c='red', s=100, marker='*', 
                      label='Current Node', zorder=10)
        
        ax.legend()
        ax.set_title(f'A* Algorithm - Step {step + 1}')

def test_astar():
    """测试A*算法"""
    # 这里需要导入环境模块进行测试
    try:
        from src.environment.scenario import create_scenario_1
        
        # 创建测试环境
        env = create_scenario_1()
        
        # 创建A*规划器
        planner = AStarPlanner(resolution=0.3)
        
        # 执行规划
        path, metrics = planner.plan_with_metrics(env.start, env.goal, env)
        
        print(f"Planning success: {metrics.success}")
        print(f"Planning time: {metrics.planning_time:.3f}s")
        print(f"Path length: {metrics.path_length:.2f}")
        print(f"Nodes explored: {metrics.num_nodes_explored}")
        print(f"Iterations: {metrics.iterations}")
        
        if path:
            # 可视化结果
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 绘制最终结果
            env.visualize(ax=ax1)
            if path:
                path_array = np.array(path)
                ax1.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='A* Path')
                ax1.legend()
            ax1.set_title('A* Planning Result')
            
            # 绘制规划过程
            planner.visualize_planning_process(env, env.start, env.goal, ax=ax2)
            
            plt.tight_layout()
            plt.show()
    
    except ImportError as e:
        print(f"Could not import required modules: {e}")
        print("Please ensure all dependencies are installed.")

if __name__ == "__main__":
    test_astar()
