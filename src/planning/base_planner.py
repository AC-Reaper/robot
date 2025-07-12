"""
路径规划算法基类
定义统一的接口和评估指标
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

class PlannerMetrics:
    """规划算法性能指标"""
    
    def __init__(self):
        self.planning_time = 0.0
        self.path_length = 0.0
        self.num_nodes_explored = 0
        self.success = False
        self.smoothness = 0.0
        self.clearance = 0.0
        self.iterations = 0
        
    def calculate_path_metrics(self, path: List[Tuple[float, float]], environment=None):
        """计算路径相关指标"""
        if len(path) < 2:
            return
            
        # 路径长度
        self.path_length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            self.path_length += np.sqrt(dx*dx + dy*dy)
        
        # 路径平滑度（曲率变化）
        if len(path) >= 3:
            curvatures = []
            for i in range(1, len(path) - 1):
                p1 = np.array(path[i-1])
                p2 = np.array(path[i])
                p3 = np.array(path[i+1])
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                # 计算曲率
                if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    curvatures.append(angle)
            
            if curvatures:
                self.smoothness = np.std(curvatures)  # 曲率变化的标准差
        
        # 安全间隙（如果提供了环境）
        if environment is not None:
            min_clearance = float('inf')
            for point in path:
                for obstacle in environment.obstacles:
                    dist = obstacle._distance_to_polygon(point[0], point[1])
                    min_clearance = min(min_clearance, dist)
            self.clearance = min_clearance if min_clearance != float('inf') else 0.0

class BasePlanner(ABC):
    """路径规划算法基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = PlannerMetrics()
        self.planning_history = []  # 存储规划过程中的中间状态
        
    @abstractmethod
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], 
             environment) -> List[Tuple[float, float]]:
        """
        路径规划主函数
        
        Args:
            start: 起点坐标
            goal: 终点坐标
            environment: 环境对象
            
        Returns:
            路径点列表
        """
        pass
    
    def plan_with_metrics(self, start: Tuple[float, float], goal: Tuple[float, float], 
                         environment) -> Tuple[List[Tuple[float, float]], PlannerMetrics]:
        """
        带性能统计的路径规划
        
        Returns:
            (路径, 性能指标)
        """
        # 重置指标
        self.metrics = PlannerMetrics()
        self.planning_history.clear()
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 执行规划
            path = self.plan(start, goal, environment)
            
            # 记录结束时间
            self.metrics.planning_time = time.time() - start_time
            
            # 计算路径指标
            if path and len(path) >= 2:
                self.metrics.success = True
                self.metrics.calculate_path_metrics(path, environment)
            else:
                self.metrics.success = False
                
        except Exception as e:
            print(f"Planning failed: {e}")
            self.metrics.planning_time = time.time() - start_time
            self.metrics.success = False
            path = []
        
        return path, self.metrics
    
    def visualize_planning_process(self, environment, start, goal, ax=None, step=-1):
        """
        可视化规划过程
        
        Args:
            environment: 环境对象
            start: 起点
            goal: 终点
            ax: matplotlib轴对象
            step: 显示到第几步(-1表示显示全部)
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制环境
        environment.visualize(ax=ax, show_grid=False)
        
        # 绘制规划过程（由子类实现具体的可视化逻辑）
        self._visualize_algorithm_specific(ax, step)
        
        ax.set_title(f'{self.name} Planning Process')
        
        return ax
    
    def _visualize_algorithm_specific(self, ax, step):
        """算法特定的可视化（由子类重写）"""
        pass
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """获取规划统计信息"""
        return {
            'algorithm': self.name,
            'success': self.metrics.success,
            'planning_time': self.metrics.planning_time,
            'path_length': self.metrics.path_length,
            'smoothness': self.metrics.smoothness,
            'clearance': self.metrics.clearance,
            'nodes_explored': self.metrics.num_nodes_explored,
            'iterations': self.metrics.iterations
        }

class Node:
    """路径规划中的节点类"""
    
    def __init__(self, x: float, y: float, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g_cost = 0.0  # 从起点到当前节点的代价
        self.h_cost = 0.0  # 从当前节点到终点的启发式代价
        self.f_cost = 0.0  # 总代价 f = g + h
        
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return abs(self.x - other.x) < 1e-6 and abs(self.y - other.y) < 1e-6
    
    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6)))
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def distance_to(self, other) -> float:
        """计算到另一个节点的距离"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def get_position(self) -> Tuple[float, float]:
        """获取节点位置"""
        return (self.x, self.y)

def reconstruct_path(node: Node) -> List[Tuple[float, float]]:
    """从终点节点重构路径"""
    path = []
    current = node
    
    while current is not None:
        path.append((current.x, current.y))
        current = current.parent
    
    return path[::-1]  # 反转路径

def smooth_path(path: List[Tuple[float, float]], environment, 
                max_iterations: int = 100) -> List[Tuple[float, float]]:
    """
    路径平滑处理
    使用简单的快捷方式方法移除不必要的中间点
    """
    if len(path) <= 2:
        return path
    
    smoothed_path = [path[0]]
    current_idx = 0
    
    while current_idx < len(path) - 1:
        # 寻找最远的可直达点
        farthest_idx = current_idx + 1
        
        for i in range(current_idx + 2, len(path)):
            if not environment.is_line_collision(path[current_idx], path[i]):
                farthest_idx = i
            else:
                break
        
        smoothed_path.append(path[farthest_idx])
        current_idx = farthest_idx
    
    return smoothed_path

if __name__ == "__main__":
    # 测试基础功能
    print("Testing base planner functionality...")
    
    # 测试节点
    node1 = Node(0, 0)
    node2 = Node(1, 1)
    node3 = Node(2, 2, node2)
    
    print(f"Distance between node1 and node2: {node1.distance_to(node2):.2f}")
    
    # 测试路径重构
    path = reconstruct_path(node3)
    print(f"Reconstructed path: {path}")
    
    # 测试指标计算
    metrics = PlannerMetrics()
    metrics.calculate_path_metrics(path)
    print(f"Path length: {metrics.path_length:.2f}")
    print(f"Path smoothness: {metrics.smoothness:.2f}")
