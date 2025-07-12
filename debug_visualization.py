#!/usr/bin/env python3
"""
调试路径可视化偏移问题
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.planning.a_star import AStarPlanner
from src.environment.scenario import get_scenario

def debug_coordinate_system():
    """调试坐标系统"""
    print("🔍 调试坐标系统和路径可视化")
    print("=" * 50)
    
    # 创建环境和算法
    env = get_scenario('simple')
    planner = AStarPlanner(resolution=0.4)
    
    # 执行路径规划
    path, metrics = planner.plan_with_metrics(env.start, env.goal, env)
    
    if not metrics.success:
        print("❌ 路径规划失败")
        return
    
    # 创建详细的可视化对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1: 环境本身
    ax1 = axes[0, 0]
    env.visualize(ax=ax1, show_grid=True)
    ax1.set_title('Environment Only', fontsize=12, fontweight='bold')
    
    # 图2: 路径点详细显示
    ax2 = axes[0, 1]
    env.visualize(ax=ax2, show_grid=True)
    
    # 显示原始起点终点
    ax2.plot(env.start[0], env.start[1], 'go', markersize=15, label='Original Start', zorder=10)
    ax2.plot(env.goal[0], env.goal[1], 'ro', markersize=15, label='Original Goal', zorder=10)
    
    # 显示实际路径起点终点
    ax2.plot(path[0][0], path[0][1], 'g^', markersize=12, label='Actual Start', zorder=9)
    ax2.plot(path[-1][0], path[-1][1], 'r^', markersize=12, label='Actual End', zorder=9)
    
    # 显示所有路径点
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax2.plot(path_x, path_y, 'b.', markersize=4, alpha=0.7, label='Path Points')
    
    ax2.legend()
    ax2.set_title('Coordinate Analysis', fontsize=12, fontweight='bold')
    
    # 图3: 路径连线
    ax3 = axes[1, 0]
    env.visualize(ax=ax3, show_grid=False)
    
    # 绘制路径
    path_array = np.array(path)
    ax3.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=3, alpha=0.8, label='A* Path')
    
    # 标注关键点
    ax3.plot(env.start[0], env.start[1], 'go', markersize=12, label='Target Start')
    ax3.plot(env.goal[0], env.goal[1], 'ro', markersize=12, label='Target Goal')
    ax3.plot(path[0][0], path[0][1], 'g^', markersize=10, label='Path Start')
    ax3.plot(path[-1][0], path[-1][1], 'r^', markersize=10, label='Path End')
    
    ax3.legend()
    ax3.set_title('Path Visualization', fontsize=12, fontweight='bold')
    
    # 图4: 误差分析
    ax4 = axes[1, 1]
    
    # 计算起点终点误差
    start_error = np.sqrt((path[0][0] - env.start[0])**2 + (path[0][1] - env.start[1])**2)
    goal_error = np.sqrt((path[-1][0] - env.goal[0])**2 + (path[-1][1] - env.goal[1])**2)
    
    # 显示误差信息
    error_text = f"""坐标偏移分析:
    
原始起点: ({env.start[0]}, {env.start[1]})
实际起点: ({path[0][0]:.2f}, {path[0][1]:.2f})
起点误差: {start_error:.3f}

原始终点: ({env.goal[0]}, {env.goal[1]})
实际终点: ({path[-1][0]:.2f}, {path[-1][1]:.2f})
终点误差: {goal_error:.3f}

网格分辨率: {planner.resolution}
路径点数: {len(path)}
路径长度: {metrics.path_length:.2f}m"""
    
    ax4.text(0.05, 0.95, error_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Error Analysis', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/images/coordinate_debug.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印详细信息
    print(f"\\n📊 详细分析:")
    print(f"原始目标: {env.start} -> {env.goal}")
    print(f"实际路径: {path[0]} -> {path[-1]}")
    print(f"起点偏移: {start_error:.3f}")
    print(f"终点偏移: {goal_error:.3f}")
    print(f"网格分辨率: {planner.resolution}")
    
    # 检查是否路径确实偏移
    if start_error > 0.5 or goal_error > 0.5:
        print("\\n⚠️  发现明显的坐标偏移问题!")
        return True
    else:
        print("\\n✅ 坐标偏移在合理范围内")
        return False

def test_different_resolutions():
    """测试不同分辨率下的路径偏移"""
    print("\\n🔬 测试不同分辨率下的偏移")
    print("=" * 40)
    
    env = get_scenario('simple')
    resolutions = [0.2, 0.4, 0.5, 1.0]
    
    results = []
    
    for res in resolutions:
        planner = AStarPlanner(resolution=res)
        path, metrics = planner.plan_with_metrics(env.start, env.goal, env)
        
        if metrics.success:
            start_error = np.sqrt((path[0][0] - env.start[0])**2 + (path[0][1] - env.start[1])**2)
            goal_error = np.sqrt((path[-1][0] - env.goal[0])**2 + (path[-1][1] - env.goal[1])**2)
            
            results.append({
                'resolution': res,
                'start_error': start_error,
                'goal_error': goal_error,
                'path_points': len(path),
                'path_length': metrics.path_length
            })
            
            print(f"分辨率 {res:0.1f}: 起点误差 {start_error:.3f}, 终点误差 {goal_error:.3f}")
    
    return results

if __name__ == "__main__":
    os.makedirs('results/images', exist_ok=True)
    
    # 调试坐标系统
    has_offset = debug_coordinate_system()
    
    # 测试不同分辨率
    test_different_resolutions()
    
    if has_offset:
        print("\\n🛠️  需要修复坐标偏移问题!")
    else:
        print("\\n✅ 坐标系统正常")