"""
可视化模块
包含绘制性能比较图、训练过程图和局部搜索统计图的函数
"""
import numpy as np
import matplotlib.pyplot as plt
from src.utils import calculate_confidence_interval
from src.config import local_search_stats, MAX_LOCAL_TRIALS

def plot_performance_comparison(generation_settings, meta_data, vanilla_data, confidence=0.95):
    """绘制性能比较图函数
    
    Args:
        generation_settings (list): 代数设置列表
        meta_data (list): Meta-NEAT数据 [(代数, 奖励列表), ...]
        vanilla_data (list): Vanilla NEAT数据 [(代数, 奖励列表), ...]
        confidence (float, optional): 置信度. Defaults to 0.95.
    """
    plt.figure(figsize=(16, 10))
    
    # 提取每代的平均值和置信区间
    x = generation_settings
    
    # Meta-NEAT数据
    y_meta = [np.mean(rewards) for _, rewards in meta_data]
    ci_meta = [calculate_confidence_interval(rewards, confidence) for _, rewards in meta_data]
    std_meta = [np.std(rewards) for _, rewards in meta_data]
    
    # Vanilla NEAT数据
    y_vanilla = [np.mean(rewards) for _, rewards in vanilla_data]
    ci_vanilla = [calculate_confidence_interval(rewards, confidence) for _, rewards in vanilla_data]
    std_vanilla = [np.std(rewards) for _, rewards in vanilla_data]
    
    # 绘制带置信区间的填充区域
    plt.plot(x, y_meta, '-o', markersize=8, label='Meta-NEAT', color='blue', linewidth=2.5)
    plt.fill_between(x, 
                     np.array(y_meta) - np.array(ci_meta), 
                     np.array(y_meta) + np.array(ci_meta), 
                     alpha=0.2, color='blue')
    
    plt.plot(x, y_vanilla, '-s', markersize=8, label='Vanilla NEAT', color='red', linewidth=2.5)
    plt.fill_between(x, 
                     np.array(y_vanilla) - np.array(ci_vanilla), 
                     np.array(y_vanilla) + np.array(ci_vanilla), 
                     alpha=0.2, color='red')
    
    plt.xlabel('进化代数', fontsize=16)
    plt.ylabel('平均奖励', fontsize=16)
    plt.title(f'Meta-NEAT vs Vanilla NEAT 性能对比 ({confidence*100:.0f}%置信区间)', fontsize=18)
    
    # 设置x轴刻度和网格更细致
    plt.xticks(generation_settings)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 保存图像
    plt.savefig('neat_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_progression(training_data, title="Training Progression"):
    """绘制训练进度
    
    Args:
        training_data (list): 训练数据列表
        title (str): 图表标题
    """
    plt.figure(figsize=(10, 6))
    
    # 计算移动平均
    window_size = min(100, len(training_data))
    if window_size > 0:
        moving_avg = np.convolve(training_data, 
                                np.ones(window_size)/window_size, 
                                mode='valid')
        plt.plot(moving_avg, label='Moving Average', color='red')
    
    # 绘制原始数据
    plt.plot(training_data, label='Episode Returns', alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    plt.savefig('training_progression.png')
    plt.close()

def plot_local_search_statistics():
    """绘制局部搜索统计信息图函数"""
    from src.config import collect_statistics
    
    if not collect_statistics or len(local_search_stats['trials']) == 0:
        print("没有收集到局部搜索统计信息")
        return
    
    plt.figure(figsize=(16, 15))
    
    # 绘制局部搜索次数分布
    plt.subplot(3, 1, 1)
    plt.hist(local_search_stats['trials'], bins=range(1, MAX_LOCAL_TRIALS+2), alpha=0.7, 
             color='blue', edgecolor='black')
    plt.xlabel('局部搜索次数', fontsize=14)
    plt.ylabel('频率', fontsize=14)
    plt.title('局部搜索次数分布 (泊松分布)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 绘制改进次数分布
    plt.subplot(3, 1, 2)
    max_improvements = max(local_search_stats['improvements']) + 1
    plt.hist(local_search_stats['improvements'], bins=range(max_improvements+1), alpha=0.7,
             color='green', edgecolor='black')
    plt.xlabel('有效改进次数', fontsize=14)
    plt.ylabel('频率', fontsize=14)
    plt.title('局部搜索有效改进次数分布', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 绘制改进率随时间变化
    plt.subplot(3, 1, 3)
    window_size = min(50, len(local_search_stats['improvement_ratios']))
    if window_size > 0:
        moving_avg = []
        for i in range(len(local_search_stats['improvement_ratios']) - window_size + 1):
            moving_avg.append(np.mean(local_search_stats['improvement_ratios'][i:i+window_size]))
        
        plt.plot(range(window_size-1, len(local_search_stats['improvement_ratios'])), 
                moving_avg, color='red', linewidth=2)
        plt.xlabel('局部搜索索引', fontsize=14)
        plt.ylabel('改进率 (滑动平均)', fontsize=14)
        plt.title(f'局部搜索改进率变化 (窗口大小={window_size})', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('local_search_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计摘要
    trials_mean = np.mean(local_search_stats['trials'])
    trials_std = np.std(local_search_stats['trials'])
    improvements_mean = np.mean(local_search_stats['improvements'])
    improvement_ratio_mean = np.mean(local_search_stats['improvement_ratios'])
    
    print(f"\n=== 局部搜索统计摘要 ===")
    print(f"总局部搜索次数: {len(local_search_stats['trials'])}")
    print(f"平均每次尝试的搜索次数: {trials_mean:.2f} ± {trials_std:.2f}")
    print(f"平均每次改进次数: {improvements_mean:.2f}")
    print(f"平均改进率: {improvement_ratio_mean:.2f}")
