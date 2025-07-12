"""
Dijkstra 路径规划算法
经典的最短路径算法，适用于非负权重图
"""

import heapq
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Set, Dict
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.planning.base_planner import BasePlanner, Node, reconstruct_path

class DijkstraPlanner(BasePlanner):
    """Dijkstra路径规划算法"""
    
    def __init__(self, resolution: float = 0.5):
        """
        初始化Dijkstra规划器
        
        Args:
            resolution: 网格分辨率
        """
        super().__init__("Dijkstra")
        self.resolution = resolution
        
        # 规划过程可视化数据
        self.visited_nodes = []
        self.current_distances = {}
    
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
        
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], 
             environment) -> List[Tuple[float, float]]:
        """
        Dijkstra路径规划主函数
        """
        # 清空历史记录
        self.visited_nodes.clear()
        self.current_distances.clear()
        
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
        
        # 初始化距离字典和优先队列
        distances: Dict[Node, float] = {}
        previous: Dict[Node, Node] = {}
        visited: Set[Node] = set()
        priority_queue = []
        
        # 初始化起点
        start_node.g_cost = 0
        distances[start_node] = 0
        heapq.heappush(priority_queue, (0, start_node))
        
        iteration = 0
        max_iterations = 100000
        
        while priority_queue and iteration < max_iterations:
            iteration += 1
            
            # 获取距离最小的节点
            current_distance, current_node = heapq.heappop(priority_queue)
            
            # 如果已经访问过，跳过
            if current_node in visited:
                continue
            
            # 标记为已访问
            visited.add(current_node)
            self.visited_nodes.append(current_node)
            
            # 检查是否到达目标
            if self._nodes_equal(current_node, goal_node):
                # 重构路径并连接到原始目标
                path = self._reconstruct_path_dijkstra(current_node, previous)
                self.metrics.num_nodes_explored = len(visited)
                self.metrics.iterations = iteration
                return self._connect_to_original_points(path, environment)
            
            # 获取邻居节点
            neighbors = self._get_neighbors(current_node, environment)
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                # 计算到邻居的距离
                edge_cost = current_node.distance_to(neighbor)
                new_distance = current_distance + edge_cost
                
                # 如果找到更短的路径，更新距离
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
                    neighbor.g_cost = new_distance
                    
                    heapq.heappush(priority_queue, (new_distance, neighbor))
                    
                    # 记录当前距离（用于可视化）
                    self.current_distances[neighbor] = new_distance
        
        # 规划失败
        self.metrics.num_nodes_explored = len(visited)
        self.metrics.iterations = iteration
        print(f"Dijkstra planning failed after {iteration} iterations")
        return []
    
    def _get_neighbors(self, node: Node, environment) -> List[Node]:
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
                    neighbor = Node(new_x, new_y)
                    neighbors.append(neighbor)
        
        return neighbors
    
    def _nodes_equal(self, node1: Node, node2: Node) -> bool:
        """判断两个节点是否相等"""
        return (abs(node1.x - node2.x) < self.resolution * 0.5 and 
                abs(node1.y - node2.y) < self.resolution * 0.5)
    
    def _reconstruct_path_dijkstra(self, goal_node: Node, 
                                  previous: Dict[Node, Node]) -> List[Tuple[float, float]]:
        """重构Dijkstra路径"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append((current.x, current.y))
            current = previous.get(current)
        
        return path[::-1]  # 反转路径
    
    def _visualize_algorithm_specific(self, ax, step):
        """Dijkstra算法特定的可视化"""
        if not self.visited_nodes:
            return
        
        # 确定显示到哪一步
        if step == -1 or step >= len(self.visited_nodes):
            step = len(self.visited_nodes) - 1
        
        if step < 0:
            return
        
        # 绘制已访问的节点
        visited_subset = self.visited_nodes[:step+1]
        if visited_subset:
            # 根据访问顺序着色（早访问的淡一些）
            visit_order = list(range(len(visited_subset)))
            colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(visited_subset)))
            
            for i, node in enumerate(visited_subset):
                ax.scatter(node.x, node.y, c=[colors[i]], s=30, alpha=0.8, 
                          edgecolors='black', linewidth=0.5)
        
        # 绘制当前处理的节点
        if step < len(self.visited_nodes):
            current = self.visited_nodes[step]
            ax.scatter(current.x, current.y, c='red', s=100, marker='*', 
                      label='Current Node', zorder=10)
        
        ax.set_title(f'Dijkstra Algorithm - Step {step + 1}/{len(self.visited_nodes)}')
        if step == len(self.visited_nodes) - 1:
            ax.legend()

class UniformCostSearch(DijkstraPlanner):
    """统一代价搜索（UCS）- Dijkstra的变体"""
    
    def __init__(self, resolution: float = 0.5, cost_function=None):
        """
        初始化UCS规划器
        
        Args:
            resolution: 网格分辨率  
            cost_function: 自定义代价函数
        """
        super().__init__(resolution)
        self.name = "UCS"
        self.cost_function = cost_function or self._default_cost_function
    
    def _default_cost_function(self, from_node: Node, to_node: Node, environment) -> float:
        """默认代价函数（欧几里得距离）"""
        return from_node.distance_to(to_node)
    
    def _get_neighbors(self, node: Node, environment) -> List[Node]:
        """获取邻居节点并计算代价"""
        neighbors = super()._get_neighbors(node, environment)
        
        # 为每个邻居计算代价
        for neighbor in neighbors:
            cost = self.cost_function(node, neighbor, environment)
            neighbor.g_cost = node.g_cost + cost
        
        return neighbors

def create_weighted_cost_function(environment):
    """创建加权代价函数，靠近障碍物的代价更高"""
    def weighted_cost(from_node: Node, to_node: Node, env) -> float:
        base_cost = from_node.distance_to(to_node)
        
        # 计算到最近障碍物的距离
        min_clearance = float('inf')
        for obstacle in env.obstacles:
            clearance = obstacle._distance_to_polygon(to_node.x, to_node.y)
            min_clearance = min(min_clearance, clearance)
        
        # 靠近障碍物时增加代价
        if min_clearance < 1.0:
            penalty = 1.0 / (min_clearance + 0.1)  # 避免除零
            return base_cost * (1 + penalty)
        
        return base_cost
    
    return weighted_cost

def test_dijkstra():
    """测试Dijkstra算法"""
    try:
        from src.environment.scenario import create_scenario_1
        
        # 创建测试环境
        env = create_scenario_1()
        
        # 测试标准Dijkstra
        print("测试Dijkstra算法...")
        dijkstra_planner = DijkstraPlanner(resolution=0.4)
        path_dijkstra, metrics_dijkstra = dijkstra_planner.plan_with_metrics(env.start, env.goal, env)
        
        print(f"Dijkstra规划成功: {metrics_dijkstra.success}")
        if metrics_dijkstra.success:
            print(f"路径长度: {metrics_dijkstra.path_length:.2f}m")
            print(f"规划时间: {metrics_dijkstra.planning_time:.3f}s")
            print(f"探索节点: {metrics_dijkstra.num_nodes_explored}")
        
        # 测试加权UCS
        print("\n测试加权UCS算法...")
        weighted_cost_func = create_weighted_cost_function(env)
        ucs_planner = UniformCostSearch(resolution=0.4, cost_function=weighted_cost_func)
        path_ucs, metrics_ucs = ucs_planner.plan_with_metrics(env.start, env.goal, env)
        
        print(f"UCS规划成功: {metrics_ucs.success}")
        if metrics_ucs.success:
            print(f"路径长度: {metrics_ucs.path_length:.2f}m")
            print(f"规划时间: {metrics_ucs.planning_time:.3f}s")
            print(f"探索节点: {metrics_ucs.num_nodes_explored}")
        
        # 可视化对比
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Dijkstra结果
        env.visualize(ax=axes[0])
        if path_dijkstra:
            path_array = np.array(path_dijkstra)
            axes[0].plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='Dijkstra Path')
        dijkstra_planner.visualize_planning_process(env, env.start, env.goal, ax=axes[0])
        
        # UCS结果
        env.visualize(ax=axes[1])
        if path_ucs:
            path_array = np.array(path_ucs)
            axes[1].plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=2, label='UCS Path')
        ucs_planner.visualize_planning_process(env, env.start, env.goal, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
        
    except ImportError as e:
        print(f"Could not import required modules: {e}")

if __name__ == "__main__":
    test_dijkstra()
