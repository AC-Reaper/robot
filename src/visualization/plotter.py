"""
可视化工具模块
提供统一的可视化接口和动画功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Polygon
import matplotlib.font_manager as fm
import seaborn as sns
from typing import List, Dict, Any, Tuple
import pandas as pd

# 设置字体，避免中文显示问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    """统一的可视化工具类"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_environment(self, environment, ax=None, title="Environment"):
        """绘制环境"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        environment.visualize(ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        return ax
    
    def plot_planning_comparison(self, planners_results: Dict[str, Any], 
                               environment, save_path=None):
        """绘制多个规划算法的对比结果"""
        n_planners = len(planners_results)
        fig, axes = plt.subplots(2, n_planners, figsize=(5*n_planners, 10))
        
        if n_planners == 1:
            axes = axes.reshape(2, 1)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        # 第一行：显示路径
        for i, (planner_name, result) in enumerate(planners_results.items()):
            ax = axes[0, i]
            
            # 绘制环境
            environment.visualize(ax=ax, show_grid=False)
            
            # 绘制路径
            path = result['path']
            if path and len(path) > 1:
                path_array = np.array(path)
                ax.plot(path_array[:, 0], path_array[:, 1], 
                       color=colors[i % len(colors)], linewidth=3, 
                       label=f'{planner_name} Path')
            
            ax.set_title(f'{planner_name}\nTime: {result["metrics"].planning_time:.3f}s, '
                        f'Length: {result["metrics"].path_length:.2f}', 
                        fontsize=12, fontweight='bold')
            ax.legend()
        
        # 第二行：显示规划过程
        for i, (planner_name, result) in enumerate(planners_results.items()):
            ax = axes[1, i]
            
            planner = result['planner']
            if hasattr(planner, 'visualize_planning_process'):
                planner.visualize_planning_process(environment, 
                                                 environment.start, 
                                                 environment.goal, 
                                                 ax=ax)
            else:
                # 简单显示最终路径
                environment.visualize(ax=ax, show_grid=False)
                path = result['path']
                if path and len(path) > 1:
                    path_array = np.array(path)
                    ax.plot(path_array[:, 0], path_array[:, 1], 
                           color=colors[i % len(colors)], linewidth=2)
                ax.set_title(f'{planner_name} - Final Result')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_control_comparison(self, controllers_results: Dict[str, Any], 
                              reference_path: List[Tuple[float, float]], 
                              environment=None, save_path=None):
        """绘制多个控制算法的对比结果"""
        n_controllers = len(controllers_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # 轨迹对比
        ax = axes[0, 0]
        if environment:
            environment.visualize(ax=ax, show_grid=False)
        
        # 绘制参考路径
        ref_array = np.array(reference_path)
        ax.plot(ref_array[:, 0], ref_array[:, 1], 'k--', 
               linewidth=2, label='Reference Path')
        
        # 绘制每个控制器的轨迹
        for i, (controller_name, result) in enumerate(controllers_results.items()):
            trajectory = result['trajectory']
            if trajectory:
                positions = [t['position'] for t in trajectory]
                traj_array = np.array(positions)
                ax.plot(traj_array[:, 0], traj_array[:, 1], 
                       color=colors[i % len(colors)], linewidth=2, 
                       label=f'{controller_name}')
        
        ax.set_title('Trajectory Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 横向误差对比
        ax = axes[0, 1]
        for i, (controller_name, result) in enumerate(controllers_results.items()):
            metrics = result['metrics']
            if metrics.lateral_errors:
                time_steps = np.arange(len(metrics.lateral_errors)) * 0.1
                ax.plot(time_steps, np.abs(metrics.lateral_errors), 
                       color=colors[i % len(colors)], linewidth=2, 
                       label=f'{controller_name}')
        
        ax.set_title('Lateral Error Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('|Lateral Error| (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 航向误差对比
        ax = axes[1, 0]
        for i, (controller_name, result) in enumerate(controllers_results.items()):
            metrics = result['metrics']
            if metrics.heading_errors:
                time_steps = np.arange(len(metrics.heading_errors)) * 0.1
                ax.plot(time_steps, np.abs(np.array(metrics.heading_errors)), 
                       color=colors[i % len(colors)], linewidth=2, 
                       label=f'{controller_name}')
        
        ax.set_title('Heading Error Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('|Heading Error| (rad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 控制输入对比
        ax = axes[1, 1]
        for i, (controller_name, result) in enumerate(controllers_results.items()):
            metrics = result['metrics']
            if metrics.control_efforts:
                time_steps = np.arange(len(metrics.control_efforts)) * 0.1
                ax.plot(time_steps, metrics.control_efforts, 
                       color=colors[i % len(colors)], linewidth=2, 
                       label=f'{controller_name}')
        
        ax.set_title('Control Effort Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Control Effort')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_performance_metrics(self, planning_stats: List[Dict], 
                               control_stats: List[Dict], 
                               save_path=None):
        """绘制性能指标对比图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 规划算法性能指标
        if planning_stats:
            # 规划时间对比
            ax = axes[0, 0]
            algorithms = [s['algorithm'] for s in planning_stats]
            times = [s['planning_time'] for s in planning_stats]
            bars = ax.bar(algorithms, times, color='skyblue', alpha=0.7)
            ax.set_title('Planning Time Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel('Time (s)')
            ax.tick_params(axis='x', rotation=45)
            
            # 在柱子上添加数值标签
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{time_val:.3f}s', ha='center', va='bottom', fontsize=10)
            
            # 路径长度对比
            ax = axes[0, 1]
            lengths = [s['path_length'] for s in planning_stats]
            bars = ax.bar(algorithms, lengths, color='lightcoral', alpha=0.7)
            ax.set_title('Path Length Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel('Length (m)')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, length in zip(bars, lengths):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{length:.2f}m', ha='center', va='bottom', fontsize=10)
            
            # 探索节点数对比
            ax = axes[0, 2]
            nodes = [s['nodes_explored'] for s in planning_stats]
            bars = ax.bar(algorithms, nodes, color='lightgreen', alpha=0.7)
            ax.set_title('Nodes Explored Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Nodes')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, node_count in zip(bars, nodes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{node_count}', ha='center', va='bottom', fontsize=10)
        
        # 控制算法性能指标
        if control_stats:
            # 最大横向误差对比
            ax = axes[1, 0]
            controllers = [s['controller'] for s in control_stats]
            max_errors = [s['max_lateral_error'] for s in control_stats]
            bars = ax.bar(controllers, max_errors, color='orange', alpha=0.7)
            ax.set_title('Max Lateral Error Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel('Error (m)')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, error in zip(bars, max_errors):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{error:.3f}m', ha='center', va='bottom', fontsize=10)
            
            # RMS横向误差对比
            ax = axes[1, 1]
            rms_errors = [s['rms_lateral_error'] for s in control_stats]
            bars = ax.bar(controllers, rms_errors, color='mediumpurple', alpha=0.7)
            ax.set_title('RMS Lateral Error Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel('RMS Error (m)')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, error in zip(bars, rms_errors):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{error:.3f}m', ha='center', va='bottom', fontsize=10)
            
            # 控制平滑度对比
            ax = axes[1, 2]
            smoothness = [s['control_smoothness'] for s in control_stats]
            bars = ax.bar(controllers, smoothness, color='gold', alpha=0.7)
            ax.set_title('Control Smoothness Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel('Smoothness')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, smooth in zip(bars, smoothness):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{smooth:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_performance_table(self, planning_stats: List[Dict], 
                               control_stats: List[Dict]) -> pd.DataFrame:
        """创建性能对比表格"""
        tables = {}
        
        if planning_stats:
            planning_df = pd.DataFrame(planning_stats)
            planning_df = planning_df.round(3)
            tables['Planning Algorithms'] = planning_df
        
        if control_stats:
            control_df = pd.DataFrame(control_stats)
            control_df = control_df.round(3)
            tables['Control Algorithms'] = control_df
        
        return tables

class AnimationVisualizer:
    """动画可视化工具"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        
    def create_planning_animation(self, planner, environment, 
                                start, goal, save_path=None):
        """创建规划过程动画"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        def animate(frame):
            ax.clear()
            planner.visualize_planning_process(environment, start, goal, ax=ax, step=frame)
            return ax.get_children()
        
        # 确定动画帧数
        if hasattr(planner, 'open_set_history'):
            frames = len(planner.open_set_history)
        else:
            frames = 50  # 默认帧数
        
        if frames > 0:
            anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                         interval=200, blit=False, repeat=True)
            
            if save_path:
                anim.save(save_path, writer='pillow', fps=5)
            
            return anim
        
        return None
    
    def create_tracking_animation(self, trajectory, reference_path, 
                                environment=None, robot_size=0.3, 
                                save_path=None):
        """创建路径跟踪动画"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 绘制静态元素
        if environment:
            environment.visualize(ax=ax, show_grid=False)
        
        # 绘制参考路径
        ref_array = np.array(reference_path)
        ax.plot(ref_array[:, 0], ref_array[:, 1], 'r--', 
               linewidth=2, label='Reference Path', alpha=0.7)
        
        # 初始化动态元素
        robot_circle = Circle((0, 0), robot_size, color='blue', alpha=0.7)
        ax.add_patch(robot_circle)
        
        trajectory_line, = ax.plot([], [], 'b-', linewidth=2, label='Robot Trajectory')
        
        ax.legend()
        ax.set_aspect('equal')
        
        def animate(frame):
            if frame < len(trajectory):
                pos = trajectory[frame]['position']
                
                # 更新机器人位置
                robot_circle.center = pos
                
                # 更新轨迹
                positions = [t['position'] for t in trajectory[:frame+1]]
                if positions:
                    pos_array = np.array(positions)
                    trajectory_line.set_data(pos_array[:, 0], pos_array[:, 1])
            
            return [robot_circle, trajectory_line]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(trajectory), 
                                     interval=100, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        
        return anim

if __name__ == "__main__":
    # 测试可视化功能
    print("Testing visualization tools...")
    
    # 创建测试数据
    planning_stats = [
        {'algorithm': 'A*', 'planning_time': 0.123, 'path_length': 15.67, 'nodes_explored': 245},
        {'algorithm': 'RRT', 'planning_time': 0.089, 'path_length': 18.32, 'nodes_explored': 156},
        {'algorithm': 'Dijkstra', 'planning_time': 0.234, 'path_length': 15.67, 'nodes_explored': 567}
    ]
    
    control_stats = [
        {'controller': 'Pure Pursuit', 'max_lateral_error': 0.234, 'rms_lateral_error': 0.123, 'control_smoothness': 0.456},
        {'controller': 'Stanley', 'max_lateral_error': 0.189, 'rms_lateral_error': 0.098, 'control_smoothness': 0.321},
        {'controller': 'PID', 'max_lateral_error': 0.267, 'rms_lateral_error': 0.145, 'control_smoothness': 0.234}
    ]
    
    # 测试可视化
    viz = Visualizer()
    fig = viz.plot_performance_metrics(planning_stats, control_stats)
    
    # 创建性能表格
    tables = viz.create_performance_table(planning_stats, control_stats)
    for table_name, df in tables.items():
        print(f"\n{table_name}:")
        print(df.to_string(index=False))
