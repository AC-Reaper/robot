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
        
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], 
             environment) -> List[Tuple[float, float]]:
        """
        Dijkstra路径规划主函数
        """
        # 清空历史记录
        self.visited_nodes.clear()
        self.current_distances.clear()
        
        # 将起点和终点对齐到网格
        start_x = round(start[0] / self.resolution) * self.resolution
        start_y = round(start[1] / self.resolution) * self.resolution
        goal_x = round(goal[0] / self.resolution) * self.resolution
        goal_y = round(goal[1] / self.resolution) * self.resolution
        
        # 创建起点和终点节点
        start_node = Node(start_x, start_y)
        goal_node = Node(goal_x, goal_y)
        
        # 检查起点和终点是否有效
        if environment.is_collision((start_x, start_y)):
            print("Start position is in collision!")
            return []
        
        if environment.is_collision((goal_x, goal_y)):
            print("Goal position is in collision!")
            return []
        
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
                # 重构路径
                path = self._reconstruct_path_dijkstra(current_node, previous)
                self.metrics.num_nodes_explored = len(visited)
                self.metrics.iterations = iteration
                return path
            
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
                
                # 检查碰撞
                if not environment.is_collision((new_x, new_y)):
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
