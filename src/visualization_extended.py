"""
扩展可视化模块
提供MAHH和奖励塑形的可视化功能
"""
import numpy as np
import matplotlib.pyplot as plt
from src.config import (
    mahh_stats, reward_shaping_stats, 
    USE_MAHH, USE_REWARD_SHAPING
)
from typing import List, Optional

def plot_mahh_statistics():
    """绘制MAHH统计图，包括接受概率和停滞指标的变化"""
    if not USE_MAHH or len(mahh_stats['acceptance_probabilities']) == 0:
        print("没有MAHH统计数据可供可视化")
        return
    
    plt.figure(figsize=(14, 10))
    
    # 创建两个y轴
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()
    
    # 绘制接受概率变化
    x = range(len(mahh_stats['acceptance_probabilities']))
    ax1.plot(x, mahh_stats['acceptance_probabilities'], 'b-', linewidth=2, label='接受概率 (p_t)')
    ax1.set_xlabel('代数', fontsize=14)
    ax1.set_ylabel('接受概率', fontsize=14, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 绘制停滞指标变化
    ax2.plot(x, mahh_stats['stagnation_metrics'], 'r-', linewidth=2, label='停滞指标')
    ax2.set_ylabel('停滞指标', fontsize=14, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 添加标题和图例
    plt.title('MAHH接受概率和停滞指标变化', fontsize=16)
    
    # 创建合并的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('mahh_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 统计摘要
    print("\n=== MAHH 统计摘要 ===")
    print(f"总进化代数: {len(mahh_stats['acceptance_probabilities'])}")
    print(f"平均接受概率: {np.mean(mahh_stats['acceptance_probabilities']):.4f}")
    print(f"最终接受概率: {mahh_stats['acceptance_probabilities'][-1]:.4f}")
    print(f"平均停滞指标: {np.mean(mahh_stats['stagnation_metrics']):.4f}")

def plot_reward_shaping_statistics():
    """绘制奖励塑形统计图，包括原始奖励、塑形奖励和偏移量的变化"""
    if not USE_REWARD_SHAPING or len(reward_shaping_stats['raw_rewards']) == 0:
        print("没有奖励塑形统计数据可供可视化")
        return
    
    # 创建一个2x2的子图网格
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 原始奖励分布直方图
    axes[0, 0].hist(reward_shaping_stats['raw_rewards'], bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('原始奖励分布', fontsize=14)
    axes[0, 0].set_xlabel('奖励值', fontsize=12)
    axes[0, 0].set_ylabel('频率', fontsize=12)
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. 塑形奖励分布直方图
    axes[0, 1].hist(reward_shaping_stats['shaped_rewards'], bins=30, alpha=0.7, color='green')
    axes[0, 1].set_title('塑形奖励分布', fontsize=14)
    axes[0, 1].set_xlabel('奖励值', fontsize=12)
    axes[0, 1].set_ylabel('频率', fontsize=12)
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. 偏移量变化曲线
    x = range(len(reward_shaping_stats['bias_history']))
    axes[1, 0].plot(x, reward_shaping_stats['bias_history'], 'r-', linewidth=2)
    axes[1, 0].set_title('偏移量变化', fontsize=14)
    axes[1, 0].set_xlabel('样本索引', fontsize=12)
    axes[1, 0].set_ylabel('偏移量', fontsize=12)
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 4. 原始vs塑形奖励散点图
    # 为了避免点太多，采样部分数据点
    sample_size = min(1000, len(reward_shaping_stats['raw_rewards']))
    indices = np.random.choice(len(reward_shaping_stats['raw_rewards']), sample_size, replace=False)
    
    raw_samples = [reward_shaping_stats['raw_rewards'][i] for i in indices]
    shaped_samples = [reward_shaping_stats['shaped_rewards'][i] for i in indices]
    
    axes[1, 1].scatter(raw_samples, shaped_samples, alpha=0.5, s=20)
    axes[1, 1].set_title('原始奖励 vs 塑形奖励', fontsize=14)
    axes[1, 1].set_xlabel('原始奖励', fontsize=12)
    axes[1, 1].set_ylabel('塑形奖励', fontsize=12)
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    # 添加参考线
    if raw_samples and shaped_samples:
        max_raw = max(raw_samples)
        min_raw = min(raw_samples)
        max_shaped = max(shaped_samples)
        min_shaped = min(shaped_samples)
        
        # 绘制平均值参考线
        raw_mean = np.mean(raw_samples)
        shaped_mean = np.mean(shaped_samples)
        axes[1, 1].axvline(x=raw_mean, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=shaped_mean, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('reward_shaping_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 统计摘要
    print("\n=== 奖励塑形统计摘要 ===")
    print(f"总样本数: {len(reward_shaping_stats['raw_rewards'])}")
    print(f"原始奖励均值: {np.mean(reward_shaping_stats['raw_rewards']):.4f}")
    print(f"塑形奖励均值: {np.mean(reward_shaping_stats['shaped_rewards']):.4f}")
    print(f"最终偏移量: {reward_shaping_stats['bias_history'][-1] if reward_shaping_stats['bias_history'] else 0:.4f}")

def plot_hybrid_results(returns: List[float], p_values: List[float], title: str):
    """绘制混合代理的训练结果
    
    Args:
        returns (List[float]): 每个episode的回报
        p_values (List[float]): 每个episode的p值
        title (str): 图表标题
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 绘制回报曲线
    ax1.plot(returns, label='Episode Return')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('Training Returns')
    ax1.grid(True)
    
    # 计算移动平均
    window_size = 100
    if len(returns) >= window_size:
        moving_avg = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(returns)), moving_avg, 
                label=f'{window_size}-Episode Moving Average', color='red')
    ax1.legend()
    
    # 绘制p值曲线
    ax2.plot(p_values, label='Q-learning Probability (p)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Probability')
    ax2.set_title('Q-learning vs NEAT Selection Probability')
    ax2.grid(True)
    
    # 添加水平线表示0.5
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('hybrid_training_results.png')
    plt.close()

def plot_ablation_study_results(results: dict, confidence: float = 0.95):
    """绘制消融实验结果
    
    Args:
        results (dict): 不同配置的结果
        confidence (float): 置信度
    """
    # 计算每个配置的统计量
    means = []
    stds = []
    labels = []
    
    for name, rewards in results.items():
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
        labels.append(name)
    
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x, means, width, yerr=stds, capsize=5)
    plt.xlabel('Configuration')
    plt.ylabel('Average Reward')
    plt.title('Ablation Study Results')
    plt.xticks(x, labels, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(means):
        plt.text(i, v + stds[i], f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('ablation_study_results.png')
    plt.close()

def plot_combined_learning_curves(baseline_meta_data, baseline_vanilla_data,
                                combined_meta_data, combined_vanilla_data,
                                confidence: float = 0.95):
    """绘制组合学习曲线
    
    Args:
        baseline_meta_data: 基线Meta-NEAT数据
        baseline_vanilla_data: 基线Vanilla NEAT数据
        combined_meta_data: 组合Meta-NEAT数据
        combined_vanilla_data: 组合Vanilla NEAT数据
        confidence (float): 置信度
    """
    plt.figure(figsize=(12, 6))
    
    # 计算移动平均
    window_size = 10
    
    def plot_curve(data, label, color):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        x = np.arange(len(mean))
        
        # 计算置信区间
        ci = std * 1.96 / np.sqrt(len(data))
        
        plt.plot(x, mean, label=label, color=color)
        plt.fill_between(x, mean-ci, mean+ci, color=color, alpha=0.2)
    
    # 绘制四条曲线
    plot_curve(baseline_meta_data, 'Baseline Meta-NEAT', 'blue')
    plot_curve(baseline_vanilla_data, 'Baseline Vanilla NEAT', 'red')
    plot_curve(combined_meta_data, 'Combined Meta-NEAT', 'green')
    plot_curve(combined_vanilla_data, 'Combined Vanilla NEAT', 'purple')
    
    plt.xlabel('Generation')
    plt.ylabel('Average Reward')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('combined_learning_curves.png')
    plt.close()
