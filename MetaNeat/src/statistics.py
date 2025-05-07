"""
统计分析模块
包含数据分析和结果比较的函数
"""
import numpy as np
from src.utils import calculate_confidence_interval

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
    
    # 比较结论（基于平均奖励和置信区间）
    for gens, m_rewards in meta_results:
        for g_v, v_rewards in vanilla_results:
            if gens == g_v:
                m_mean = np.mean(m_rewards)
                v_mean = np.mean(v_rewards)
                m_ci = calculate_confidence_interval(m_rewards, confidence)
                v_ci = calculate_confidence_interval(v_rewards, confidence)
                
                if m_mean - m_ci > v_mean + v_ci:
                    print(f"\nAt {gens} generations: Meta-NEAT significantly outperforms Vanilla NEAT")
                    print(f"Meta-NEAT: {m_mean:.2f} ± {m_ci:.2f} vs Vanilla NEAT: {v_mean:.2f} ± {v_ci:.2f}")
                elif v_mean - v_ci > m_mean + m_ci:
                    print(f"\nAt {gens} generations: Vanilla NEAT significantly outperforms Meta-NEAT")
                    print(f"Vanilla NEAT: {v_mean:.2f} ± {v_ci:.2f} vs Meta-NEAT: {m_mean:.2f} ± {m_ci:.2f}")
                else:
                    print(f"\nAt {gens} generations: No significant difference between Meta-NEAT and Vanilla NEAT")
                    print(f"Meta-NEAT: {m_mean:.2f} ± {m_ci:.2f} vs Vanilla NEAT: {v_mean:.2f} ± {v_ci:.2f}")
