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

def plot_combined_learning_curves(meta_data, vanilla_data, meta_mahh_data=None, vanilla_mahh_data=None, confidence=0.95):
    """
    绘制结合了原始方法和改进方法的学习曲线
    
    Args:
        meta_data (list): Meta-NEAT训练数据
        vanilla_data (list): Vanilla NEAT训练数据
        meta_mahh_data (list, optional): 使用MAHH的Meta-NEAT训练数据
        vanilla_mahh_data (list, optional): 使用MAHH的Vanilla NEAT训练数据
        confidence (float, optional): 置信度. Defaults to 0.95.
    """
    from scipy import stats
    
    def calculate_confidence_interval(data, confidence=0.95):
        if len(data) <= 1:
            return 0
        n = len(data)
        m = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return h
    
    plt.figure(figsize=(16, 10))
    
    # 确定最大代数
    max_gens = max([
        max([len(data) for data in meta_data]) if meta_data else 0,
        max([len(data) for data in vanilla_data]) if vanilla_data else 0,
        max([len(data) for data in meta_mahh_data]) if meta_mahh_data else 0,
        max([len(data) for data in vanilla_mahh_data]) if vanilla_mahh_data else 0
    ])
    
    x = list(range(1, max_gens + 1))
    
    # 设置颜色和标记
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', 's', '^', 'x']
    labels = ['Meta-NEAT', 'Vanilla NEAT', 'Meta-NEAT+MAHH', 'Vanilla NEAT+MAHH']
    data_sets = [meta_data, vanilla_data, meta_mahh_data, vanilla_mahh_data]
    
    # 为每个数据集绘制学习曲线
    for i, dataset in enumerate(data_sets):
        if not dataset:
            continue
            
        # 计算每代的平均值和置信区间
        y_values = []
        ci_values = []
        
        for gen in range(max_gens):
            gen_data = [run_data[gen] if gen < len(run_data) else None for run_data in dataset]
            gen_data = [d for d in gen_data if d is not None]
            
            if gen_data:
                y_values.append(np.mean(gen_data))
                ci_values.append(calculate_confidence_interval(gen_data, confidence))
            else:
                y_values.append(0)
                ci_values.append(0)
        
        # 绘制带置信区间的曲线
        plt.plot(x, y_values, marker=markers[i], color=colors[i], label=labels[i], linewidth=2)
        plt.fill_between(
            x, 
            np.array(y_values) - np.array(ci_values),
            np.array(y_values) + np.array(ci_values),
            color=colors[i], alpha=0.2
        )
    
    plt.xlabel('进化代数', fontsize=16)
    plt.ylabel('适应度', fontsize=16)
    plt.title(f'NEAT变体性能对比 ({confidence*100:.0f}%置信区间)', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    
    plt.savefig('neat_variants_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_ablation_study_results(results, confidence=0.95):
    """
    绘制消融实验结果
    
    Args:
        results (dict): 包含不同配置结果的字典
        confidence (float, optional): 置信度. Defaults to 0.95.
    """
    from scipy import stats
    
    def calculate_confidence_interval(data, confidence=0.95):
        if len(data) <= 1:
            return 0
        n = len(data)
        m = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return h
    
    # 期望的格式：
    # results = {
    #    "Baseline": [rewards_list],
    #    "MAHH": [rewards_list],
    #    "RewardShaping": [rewards_list],
    #    "MAHH+RewardShaping": [rewards_list]
    # }
    
    plt.figure(figsize=(12, 8))
    
    # 计算每个配置的平均值和置信区间
    means = []
    errors = []
    labels = []
    
    for label, rewards in results.items():
        means.append(np.mean(rewards))
        errors.append(calculate_confidence_interval(rewards, confidence))
        labels.append(label)
    
    # 绘制条形图
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=errors, align='center', alpha=0.7, capsize=10, color=['gray', 'blue', 'green', 'purple'])
    
    plt.ylabel('平均适应度', fontsize=14)
    plt.title(f'消融实验结果 ({confidence*100:.0f}%置信区间)', fontsize=16)
    plt.xticks(x, labels, fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, v in enumerate(means):
        plt.text(i, v + errors[i] + 0.05 * max(means), f'{v:.2f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('ablation_study_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n=== 消融实验统计摘要 ===")
    for label, rewards in results.items():
        ci = calculate_confidence_interval(rewards, confidence)
        print(f"{label}: 平均值 = {np.mean(rewards):.2f} ± {ci:.2f}")
