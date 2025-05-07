"""
统计分析模块
包含数据分析和结果比较的函数
"""
import numpy as np
from scipy import stats
from src.utils import calculate_confidence_interval

def perform_wilcoxon_test(meta_rewards, vanilla_rewards):
    """执行Wilcoxon秩和检验
    
    Args:
        meta_rewards (list): Meta-NEAT的奖励列表
        vanilla_rewards (list): Vanilla NEAT的奖励列表
        
    Returns:
        tuple: (统计量, p值)
    """
    statistic, p_value = stats.ranksums(meta_rewards, vanilla_rewards)
    return statistic, p_value

def compare_results(generation_settings, meta_results, vanilla_results, confidence=0.95):
    """比较Meta-NEAT和Vanilla NEAT的结果，输出统计分析
    
    Args:
        generation_settings (list): 代数设置列表
        meta_results (list): Meta-NEAT结果 [(代数, 奖励列表), ...]
        vanilla_results (list): Vanilla NEAT结果 [(代数, 奖励列表), ...]
        confidence (float, optional): 置信度. Defaults to 0.95.
    """
    print("\n=== Summary of Results ===")
    for gens, rewards in meta_results:
        ci = calculate_confidence_interval(rewards, confidence)
        print(f"Meta-NEAT (Generations = {gens}): Mean Reward = {np.mean(rewards):.2f} ± {ci:.2f} ({confidence*100:.0f}% CI), Std = {np.std(rewards):.2f}")
    
    for gens, rewards in vanilla_results:
        ci = calculate_confidence_interval(rewards, confidence)
        print(f"Vanilla NEAT (Generations = {gens}): Mean Reward = {np.mean(rewards):.2f} ± {ci:.2f} ({confidence*100:.0f}% CI), Std = {np.std(rewards):.2f}")
    
    # 比较结论（基于平均奖励、置信区间和Wilcoxon检验）
    print("\n=== Wilcoxon Rank-Sum Test Results ===")
    for gens, m_rewards in meta_results:
        for g_v, v_rewards in vanilla_results:
            if gens == g_v:
                m_mean = np.mean(m_rewards)
                v_mean = np.mean(v_rewards)
                m_ci = calculate_confidence_interval(m_rewards, confidence)
                v_ci = calculate_confidence_interval(v_rewards, confidence)
                
                # 执行Wilcoxon秩和检验
                statistic, p_value = perform_wilcoxon_test(m_rewards, v_rewards)
                
                print(f"\nAt {gens} generations:")
                print(f"Meta-NEAT: {m_mean:.2f} ± {m_ci:.2f}")
                print(f"Vanilla NEAT: {v_mean:.2f} ± {v_ci:.2f}")
                print(f"Wilcoxon Rank-Sum Test: statistic = {statistic:.4f}, p-value = {p_value:.4f}")
                
                # 基于p值的结论
                alpha = 0.05
                if p_value < alpha:
                    if m_mean > v_mean:
                        print("结论: Meta-NEAT显著优于Vanilla NEAT (p < 0.05)")
                    else:
                        print("结论: Vanilla NEAT显著优于Meta-NEAT (p < 0.05)")
                else:
                    print("结论: 两种算法之间没有显著差异 (p >= 0.05)")
