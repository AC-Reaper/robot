"""
环境和场景配置模块
定义不同的测试场景，包括起点、终点、障碍物等
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json

class Obstacle:
    """障碍物类"""
    def __init__(self, vertices: List[Tuple[float, float]]):
        self.vertices = np.array(vertices)
    
    def is_collision(self, point: Tuple[float, float], radius: float = 0.2) -> bool:
        """检查点是否与障碍物碰撞"""
        x, y = point
        # 使用点到多边形的距离判断
        return self._point_in_polygon(x, y) or self._distance_to_polygon(x, y) < radius
    
    def _point_in_polygon(self, x: float, y: float) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(self.vertices)
        inside = False
        
        p1x, p1y = self.vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _distance_to_polygon(self, x: float, y: float) -> float:
        """计算点到多边形边界的最小距离"""
        min_dist = float('inf')
        n = len(self.vertices)
        
        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]
            dist = self._distance_to_line(x, y, p1, p2)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _distance_to_line(self, x: float, y: float, p1: np.ndarray, p2: np.ndarray) -> float:
        """计算点到线段的距离"""
        A = x - p1[0]
        B = y - p1[1]
        C = p2[0] - p1[0]
        D = p2[1] - p1[1]
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return np.sqrt(A * A + B * B)
        
        param = dot / len_sq
        
        if param < 0:
            xx, yy = p1[0], p1[1]
        elif param > 1:
            xx, yy = p2[0], p2[1]
        else:
            xx = p1[0] + param * C
            yy = p1[1] + param * D
        
        dx = x - xx
        dy = y - yy
        return np.sqrt(dx * dx + dy * dy)

class Environment:
    """环境类，包含地图信息和障碍物"""
    
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.obstacles: List[Obstacle] = []
        self.start = (0, 0)
        self.goal = (width-1, height-1)
    
    def add_obstacle(self, vertices: List[Tuple[float, float]]):
        """添加障碍物"""
        self.obstacles.append(Obstacle(vertices))
    
    def set_start_goal(self, start: Tuple[float, float], goal: Tuple[float, float]):
        """设置起点和终点"""
        self.start = start
        self.goal = goal
    
    def is_collision(self, point: Tuple[float, float], radius: float = 0.2) -> bool:
        """检查点是否与任何障碍物碰撞"""
        x, y = point
        
        # 检查边界
        if x - radius < 0 or x + radius > self.width or y - radius < 0 or y + radius > self.height:
            return True
        
        # 检查障碍物
        for obstacle in self.obstacles:
            if obstacle.is_collision(point, radius):
                return True
        
        return False
    
    def is_line_collision(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                         robot_radius: float = 0.2, resolution: float = 0.1) -> bool:
        """检查线段是否与障碍物碰撞"""
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        num_points = int(distance / resolution) + 1
        
        for i in range(num_points):
            t = i / max(1, num_points - 1)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            if self.is_collision((x, y), robot_radius):
                return True
        
        return False
    
    def visualize(self, ax=None, show_grid=True):
        """可视化环境"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制边界
        ax.plot([0, self.width, self.width, 0, 0], 
                [0, 0, self.height, self.height, 0], 'k-', linewidth=2)
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            vertices = np.vstack([obstacle.vertices, obstacle.vertices[0]])
            ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=2)
            ax.fill(vertices[:, 0], vertices[:, 1], color='gray', alpha=0.7)
        
        # 绘制起点和终点
        ax.plot(self.start[0], self.start[1], 'go', markersize=12, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'ro', markersize=12, label='Goal')
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('Environment')
        
        return ax

def create_scenario_1() -> Environment:
    """创建场景1：简单环境"""
    env = Environment(12, 10)
    env.set_start_goal((1, 1), (11, 9))
    
    # 添加几个简单的矩形障碍物
    env.add_obstacle([(3, 2), (5, 2), (5, 4), (3, 4)])
    env.add_obstacle([(7, 6), (9, 6), (9, 8), (7, 8)])
    env.add_obstacle([(4, 6), (6, 6), (6, 7), (4, 7)])
    
    return env

def create_scenario_2() -> Environment:
    """创建场景2：复杂环境"""
    env = Environment(15, 12)
    env.set_start_goal((1, 1), (14, 11))
    
    # 添加更复杂的障碍物
    env.add_obstacle([(2, 3), (4, 3), (4, 6), (2, 6)])
    env.add_obstacle([(6, 2), (8, 2), (8, 5), (6, 5)])
    env.add_obstacle([(9, 7), (11, 7), (11, 9), (9, 9)])
    env.add_obstacle([(3, 8), (5, 8), (5, 10), (3, 10)])
    env.add_obstacle([(12, 3), (14, 3), (14, 6), (12, 6)])
    
    # 添加一个L形障碍物
    env.add_obstacle([(7, 8), (9, 8), (9, 10), (8, 10), (8, 9), (7, 9)])
    
    return env


def create_scenario_3() -> Environment:
    """创建场景3：有解的迷宫环境"""
    env = Environment(20, 20)
    env.set_start_goal((1, 1), (19, 19))
    
    # 创建迷宫样式的障碍物，确保有解
    obstacles = [
        # 第一层障碍物（靠近起点）
        [(0, 3), (15, 3), (15, 4), (0, 4)],  # 横向墙，留出右侧通道
        
        # 第二层障碍物
        [(5, 4), (6, 4), (6, 10), (5, 10)],  # 左侧竖墙
        [(10, 4), (11, 4), (11, 10), (10, 10)],  # 右侧竖墙
        
        # 第三层障碍物
        [(2, 7), (18, 7), (18, 8), (2, 8)],  # 横向墙，留出两侧通道
        
        # 第四层障碍物
        [(0, 10), (8, 10), (8, 11), (0, 11)],  # 左侧横墙
        [(12, 10), (20, 10), (20, 11), (12, 11)],  # 右侧横墙
        
        # 第五层障碍物
        [(3, 11), (4, 11), (4, 16), (3, 16)],  # 左侧竖墙
        [(16, 11), (17, 11), (17, 16), (16, 16)],  # 右侧竖墙
        
        # 第六层障碍物
        [(6, 13), (14, 13), (14, 14), (6, 14)],  # 中间横墙
        
        # 第七层障碍物
        [(0, 16), (10, 16), (10, 17), (0, 17)],  # 左侧横墙
        [(14, 16), (20, 16), (20, 17), (14, 17)],  # 右侧横墙
        
        # 额外的障碍物增加难度
        [(8, 8), (9, 8), (9, 10), (8, 10)],  # 中间小方块
        [(11, 14), (13, 14), (13, 16), (11, 16)],  # 下方小方块
    ]
    
    for obs in obstacles:
        env.add_obstacle(obs)
    
    return env

# 预定义场景字典
SCENARIOS = {
    'simple': create_scenario_1,
    'complex': create_scenario_2, 
    'maze': create_scenario_3
}

def get_scenario(name: str) -> Environment:
    """获取预定义场景"""
    if name in SCENARIOS:
        return SCENARIOS[name]()
    else:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")

if __name__ == "__main__":
    # 测试所有场景
    scenarios = ['simple', 'complex', 'maze']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, scenario_name in enumerate(scenarios):
        env = get_scenario(scenario_name)
        env.visualize(ax=axes[i])
        axes[i].set_title(f'Scenario {i+1}: {scenario_name.title()}')
    
    plt.tight_layout()
    plt.show()
