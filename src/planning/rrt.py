"""
RRT (Rapidly-exploring Random Tree) 路径规划算法
基于随机采样的快速探索树算法
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.planning.base_planner import BasePlanner, Node, smooth_path

class RRTNode(Node):
    """RRT专用节点类"""
    
    def __init__(self, x: float, y: float, parent=None):
        super().__init__(x, y, parent)
        self.children = []  # 子节点列表
        
    def add_child(self, child_node):
        """添加子节点"""
        self.children.append(child_node)
        child_node.parent = self

class RRTPlanner(BasePlanner):
    """RRT路径规划算法"""
    
    def __init__(self, max_iterations: int = 100000, step_size: float = 0.5, 
                 goal_sample_rate: float = 0.1, goal_tolerance: float = 0.5):
        """
        初始化RRT规划器
        
        Args:
            max_iterations: 最大迭代次数
            step_size: 步长
            goal_sample_rate: 目标点采样概率
            goal_tolerance: 目标容差
        """
        super().__init__("RRT")
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.goal_tolerance = goal_tolerance
        
        # 用于可视化的历史记录
        self.tree_nodes = []
        self.tree_edges = []
        self.sampled_points = []
        
    def plan(self, start: tuple, goal: tuple, environment) -> list:
        """
        RRT路径规划主函数
        """
        # 清空历史记录
        self.tree_nodes.clear()
        self.tree_edges.clear()
        self.sampled_points.clear()
        
        # 初始化树
        start_node = RRTNode(start[0], start[1])
        self.tree_nodes.append(start_node)
        
        goal_node = RRTNode(goal[0], goal[1])
        
        # 设置随机种子
        random.seed(42)  # 为了结果可重现
        
        for iteration in range(self.max_iterations):
            # 采样随机点
            if random.random() < self.goal_sample_rate:
                # 以一定概率采样目标点
                rand_point = goal
            else:
                # 随机采样
                rand_point = self._sample_random_point(environment)
            
            self.sampled_points.append(rand_point)
            
            # 找到树中距离采样点最近的节点
            nearest_node = self._find_nearest_node(rand_point)
            
            # 向采样点方向扩展
            new_node = self._extend_tree(nearest_node, rand_point, environment)
            
            if new_node is None:
                continue
            
            # 添加到树中
            nearest_node.add_child(new_node)
            self.tree_nodes.append(new_node)
            self.tree_edges.append((nearest_node, new_node))
            
            # 检查是否到达目标
            if self._distance(new_node, goal_node) <= self.goal_tolerance:
                # 连接到目标点
                if not environment.is_line_collision(
                    (new_node.x, new_node.y), goal, robot_radius=0.2
                ):
                    goal_node.parent = new_node
                    self.tree_nodes.append(goal_node)
                    self.tree_edges.append((new_node, goal_node))
                    
                    # 重构路径
                    path = self._reconstruct_path(goal_node)
                    
                    # 路径平滑
                    smoothed_path = smooth_path(path, environment)
                    
                    self.metrics.num_nodes_explored = len(self.tree_nodes)
                    self.metrics.iterations = iteration + 1
                    
                    return smoothed_path
        
        # 规划失败
        self.metrics.num_nodes_explored = len(self.tree_nodes)
        self.metrics.iterations = self.max_iterations
        print(f"RRT planning failed after {self.max_iterations} iterations")
        return []
    
    def _sample_random_point(self, environment) -> tuple:
        """随机采样点"""
        while True:
            x = random.uniform(0, environment.width)
            y = random.uniform(0, environment.height)
            
            # 检查采样点是否在自由空间内
            if not environment.is_collision((x, y)):
                return (x, y)
    
    def _find_nearest_node(self, point: tuple) -> RRTNode:
        """找到距离给定点最近的树节点"""
        min_distance = float('inf')
        nearest_node = None
        
        for node in self.tree_nodes:
            distance = math.sqrt((node.x - point[0])**2 + (node.y - point[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _extend_tree(self, from_node: RRTNode, to_point: tuple, environment) -> RRTNode:
        """向目标方向扩展树"""
        # 计算方向向量
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance == 0:
            return None
        
        # 归一化方向向量
        unit_dx = dx / distance
        unit_dy = dy / distance
        
        # 计算新节点位置
        step = min(self.step_size, distance)
        new_x = from_node.x + step * unit_dx
        new_y = from_node.y + step * unit_dy
        
        # 检查新节点是否有效
        if environment.is_collision((new_x, new_y)):
            return None
        
        # 检查从父节点到新节点的路径是否无碰撞
        if environment.is_line_collision((from_node.x, from_node.y), (new_x, new_y)):
            return None
        
        return RRTNode(new_x, new_y, from_node)
    
    def _distance(self, node1: RRTNode, node2: RRTNode) -> float:
        """计算两节点间距离"""
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def _reconstruct_path(self, goal_node: RRTNode) -> list:
        """重构路径"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        
        return path[::-1]  # 反转路径
    
    def _visualize_algorithm_specific(self, ax, step):
        """RRT算法特定的可视化"""
        if not self.tree_nodes:
            return
        
        # 绘制树的边
        for from_node, to_node in self.tree_edges:
            ax.plot([from_node.x, to_node.x], [from_node.y, to_node.y], 
                   'b-', linewidth=1, alpha=0.6)
        
        # 绘制树的节点
        if self.tree_nodes:
            node_x = [node.x for node in self.tree_nodes]
            node_y = [node.y for node in self.tree_nodes]
            ax.scatter(node_x, node_y, c='blue', s=20, alpha=0.8, label='RRT Tree Nodes')
        
        # 绘制采样点（只显示部分，避免太拥挤）
        if self.sampled_points and len(self.sampled_points) > 0:
            sample_step = max(1, len(self.sampled_points) // 50)  # 最多显示50个采样点
            sampled_subset = self.sampled_points[::sample_step]
            
            sample_x = [p[0] for p in sampled_subset]
            sample_y = [p[1] for p in sampled_subset]
            ax.scatter(sample_x, sample_y, c='red', s=10, alpha=0.5, 
                      marker='x', label='Sampled Points')
        
        ax.legend()
        ax.set_title(f'RRT Algorithm - {len(self.tree_nodes)} nodes explored')

class RRTStarPlanner(RRTPlanner):
    """RRT* 路径规划算法（RRT的优化版本）"""
    
    def __init__(self, max_iterations: int = 5000, step_size: float = 0.5, 
                 goal_sample_rate: float = 0.1, goal_tolerance: float = 0.5,
                 search_radius: float = 1.0):
        """
        初始化RRT*规划器
        
        Args:
            search_radius: 重连搜索半径
        """
        super().__init__(max_iterations, step_size, goal_sample_rate, goal_tolerance)
        self.name = "RRT*"
        self.search_radius = search_radius
    
    def _extend_tree(self, from_node: RRTNode, to_point: tuple, environment) -> RRTNode:
        """扩展树（RRT*版本，包含重连优化）"""
        # 首先按RRT方式创建新节点
        new_node = super()._extend_tree(from_node, to_point, environment)
        
        if new_node is None:
            return None
        
        # RRT* 优化：寻找更好的父节点
        new_node = self._choose_parent(new_node, environment)
        
        # RRT* 优化：重连附近节点
        self._rewire(new_node, environment)
        
        return new_node
    
    def _choose_parent(self, new_node: RRTNode, environment) -> RRTNode:
        """为新节点选择最佳父节点"""
        best_parent = new_node.parent
        best_cost = self._calculate_cost(new_node)
        
        # 搜索附近节点
        nearby_nodes = self._find_nearby_nodes(new_node)
        
        for nearby_node in nearby_nodes:
            # 检查连接是否无碰撞
            if not environment.is_line_collision(
                (nearby_node.x, nearby_node.y), (new_node.x, new_node.y)
            ):
                # 计算通过该节点的代价
                cost = self._calculate_cost(nearby_node) + self._distance(nearby_node, new_node)
                
                if cost < best_cost:
                    best_parent = nearby_node
                    best_cost = cost
        
        # 更新父节点
        new_node.parent = best_parent
        new_node.g_cost = best_cost
        
        return new_node
    
    def _rewire(self, new_node: RRTNode, environment):
        """重连附近节点以优化路径"""
        nearby_nodes = self._find_nearby_nodes(new_node)
        
        for nearby_node in nearby_nodes:
            if nearby_node == new_node.parent:
                continue
            
            # 检查连接是否无碰撞
            if not environment.is_line_collision(
                (new_node.x, new_node.y), (nearby_node.x, nearby_node.y)
            ):
                # 计算通过新节点的代价
                new_cost = new_node.g_cost + self._distance(new_node, nearby_node)
                
                if new_cost < self._calculate_cost(nearby_node):
                    # 重连
                    old_parent = nearby_node.parent
                    if old_parent and nearby_node in old_parent.children:
                        old_parent.children.remove(nearby_node)
                    
                    new_node.add_child(nearby_node)
                    nearby_node.g_cost = new_cost
    
    def _find_nearby_nodes(self, node: RRTNode) -> list:
        """找到节点附近的所有节点"""
        nearby_nodes = []
        
        for tree_node in self.tree_nodes:
            if tree_node == node:
                continue
            
            if self._distance(node, tree_node) <= self.search_radius:
                nearby_nodes.append(tree_node)
        
        return nearby_nodes
    
    def _calculate_cost(self, node: RRTNode) -> float:
        """计算从根节点到当前节点的代价"""
        cost = 0
        current = node
        
        while current.parent is not None:
            cost += self._distance(current, current.parent)
            current = current.parent
        
        return cost

def test_rrt():
    """测试RRT算法"""
    try:
        from src.environment.scenario import create_scenario_1
        
        # 创建测试环境
        env = create_scenario_1()
        
        # 测试RRT
        print("测试RRT算法...")
        rrt_planner = RRTPlanner(max_iterations=3000, step_size=0.5)
        path_rrt, metrics_rrt = rrt_planner.plan_with_metrics(env.start, env.goal, env)
        
        print(f"RRT规划成功: {metrics_rrt.success}")
        if metrics_rrt.success:
            print(f"路径长度: {metrics_rrt.path_length:.2f}m")
            print(f"规划时间: {metrics_rrt.planning_time:.3f}s")
            print(f"探索节点: {metrics_rrt.num_nodes_explored}")
        
        # 测试RRT*
        print("\n测试RRT*算法...")
        rrt_star_planner = RRTStarPlanner(max_iterations=3000, step_size=0.5)
        path_rrt_star, metrics_rrt_star = rrt_star_planner.plan_with_metrics(env.start, env.goal, env)
        
        print(f"RRT*规划成功: {metrics_rrt_star.success}")
        if metrics_rrt_star.success:
            print(f"路径长度: {metrics_rrt_star.path_length:.2f}m")
            print(f"规划时间: {metrics_rrt_star.planning_time:.3f}s")
            print(f"探索节点: {metrics_rrt_star.num_nodes_explored}")
        
        # 可视化对比
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RRT结果
        env.visualize(ax=axes[0])
        if path_rrt:
            path_array = np.array(path_rrt)
            axes[0].plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='RRT Path')
        rrt_planner.visualize_planning_process(env, env.start, env.goal, ax=axes[0])
        
        # RRT*结果
        env.visualize(ax=axes[1])
        if path_rrt_star:
            path_array = np.array(path_rrt_star)
            axes[1].plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=2, label='RRT* Path')
        rrt_star_planner.visualize_planning_process(env, env.start, env.goal, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
        
    except ImportError as e:
        print(f"Could not import required modules: {e}")

if __name__ == "__main__":
    test_rrt()
