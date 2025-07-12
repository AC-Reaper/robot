#!/usr/bin/env python3
"""
测试动态路径规划可视化
创建简单的动画演示验证功能
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environment.scenario import get_scenario
from src.planning.a_star import AStarPlanner
from src.planning.dijkstra import DijkstraPlanner

def create_simple_animation():
    """创建简单的动画演示"""
    print("🎬 创建动态路径规划演示...")
    
    # 创建环境和算法
    env = get_scenario('simple')
    planner = AStarPlanner(resolution=0.4)
    
    print(f"场景: {env.start} -> {env.goal}")
    
    # 执行规划
    path, metrics = planner.plan_with_metrics(env.start, env.goal, env)
    
    if not metrics.success:
        print("❌ 路径规划失败")
        return
    
    print(f"✅ 路径规划成功: {len(path)}点, {metrics.planning_time:.3f}s")
    
    # 获取搜索历史
    search_history = []
    if hasattr(planner, 'open_set_history') and planner.open_set_history:
        for i in range(len(planner.open_set_history)):
            step_data = {
                'open_set': planner.open_set_history[i],
                'closed_set': planner.closed_set_history[i] if i < len(planner.closed_set_history) else set(),
                'current_node': planner.current_node_history[i] if i < len(planner.current_node_history) else None
            }
            search_history.append(step_data)
    
    if not search_history:
        print("❌ 没有搜索历史数据")
        return
    
    print(f"📊 搜索历史: {len(search_history)} 步")
    
    # 创建动画
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def animate(frame):
        ax.clear()
        
        # 绘制环境
        env.visualize(ax=ax, show_grid=False)
        
        # 绘制搜索状态
        if frame < len(search_history):
            step_data = search_history[frame]
            
            # 绘制关闭集
            if step_data['closed_set']:
                closed_x = [node.x for node in step_data['closed_set']]
                closed_y = [node.y for node in step_data['closed_set']]
                ax.scatter(closed_x, closed_y, c='lightblue', s=20, alpha=0.6, 
                          label='Explored Nodes', zorder=3)
            
            # 绘制开放集
            if step_data['open_set']:
                open_x = [node.x for node in step_data['open_set']]
                open_y = [node.y for node in step_data['open_set']]
                ax.scatter(open_x, open_y, c='yellow', s=30, alpha=0.8, 
                          label='Frontier Nodes', zorder=4)
            
            # 绘制当前节点
            if step_data['current_node']:
                current = step_data['current_node']
                ax.scatter(current.x, current.y, c='red', s=100, marker='*', 
                          label='Current Node', zorder=5)
        
        # 绘制最终路径（在后半段动画中）
        if frame >= len(search_history) // 2 and path:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'g-', linewidth=3, 
                   label='Final Path', alpha=0.8, zorder=10)
        
        # 设置标题和信息
        progress = min(frame + 1, len(search_history))
        ax.set_title(f'A* Algorithm - Step {progress}/{len(search_history)}', 
                    fontsize=14, fontweight='bold')
        
        info_text = f'Progress: {progress/len(search_history)*100:.1f}%\n'
        info_text += f'Target: {metrics.num_nodes_explored} nodes'
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        if frame >= len(search_history) - 5:
            ax.legend()
    
    # 创建动画
    frames = min(len(search_history), 50)  # 限制帧数
    anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                  interval=400, blit=False, repeat=True)
    
    # 保存为GIF
    os.makedirs('results/animations', exist_ok=True)
    gif_filename = 'results/animations/astar_demo.gif'
    print(f"💾 保存动画到 {gif_filename}...")
    
    try:
        anim.save(gif_filename, writer='pillow', fps=3)
        print("✅ 动画保存成功")
    except Exception as e:
        print(f"❌ 动画保存失败: {e}")
    
    # 显示静态结果
    plt.figure(figsize=(10, 8))
    env.visualize()
    if path:
        path_array = np.array(path)
        plt.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=3, label='A* Path')
    plt.title('A* Algorithm Final Result')
    plt.legend()
    plt.savefig('results/images/astar_result.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_all_algorithms():
    """测试所有算法的最终状态"""
    print("\n🎯 测试所有算法的完整功能")
    print("=" * 50)
    
    env = get_scenario('simple')
    
    algorithms = [
        ('A*', AStarPlanner(resolution=0.4)),
        ('Dijkstra', DijkstraPlanner(resolution=0.4))
    ]
    
    results = []
    
    for name, planner in algorithms:
        print(f"\n🔧 测试 {name} 算法...")
        
        start_time = time.time()
        path, metrics = planner.plan_with_metrics(env.start, env.goal, env)
        total_time = time.time() - start_time
        
        if metrics.success:
            print(f"✅ {name} 成功:")
            print(f"   路径点数: {len(path)}")
            print(f"   规划时间: {metrics.planning_time:.3f}s")
            print(f"   路径长度: {metrics.path_length:.2f}m")
            print(f"   探索节点: {metrics.num_nodes_explored}")
            
            results.append({
                'name': name,
                'success': True,
                'path': path,
                'metrics': metrics
            })
        else:
            print(f"❌ {name} 失败")
            results.append({
                'name': name,
                'success': False,
                'path': None,
                'metrics': metrics
            })
    
    # 创建对比图
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 6))
    if len(results) == 1:
        axes = [axes]
    
    for i, result in enumerate(results):
        ax = axes[i]
        env.visualize(ax=ax, show_grid=False)
        
        if result['success'] and result['path']:
            path_array = np.array(result['path'])
            colors = ['red', 'blue', 'green', 'orange']
            ax.plot(path_array[:, 0], path_array[:, 1], 
                   color=colors[i], linewidth=3, label=f"{result['name']} Path")
        
        success_text = "SUCCESS" if result['success'] else "FAILED"
        color = 'green' if result['success'] else 'red'
        ax.set_title(f"{result['name']} - {success_text}", 
                    color=color, fontsize=12, fontweight='bold')
        
        if result['success']:
            info = f"Time: {result['metrics'].planning_time:.3f}s\n"
            info += f"Length: {result['metrics'].path_length:.2f}m\n"
            info += f"Nodes: {result['metrics'].num_nodes_explored}"
            
            ax.text(0.02, 0.98, info, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/images/algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 测试完成!")
    print(f"📂 结果保存在:")
    print(f"   - results/images/astar_result.png")
    print(f"   - results/images/algorithm_comparison.png")
    print(f"   - results/animations/astar_demo.gif")

if __name__ == "__main__":
    # 创建结果目录
    os.makedirs('results/images', exist_ok=True)
    os.makedirs('results/animations', exist_ok=True)
    
    # 运行演示
    create_simple_animation()
    test_all_algorithms()