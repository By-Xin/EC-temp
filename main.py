"""
主程序
Meta-NEAT与Vanilla NEAT对比实验，集成MAHH和奖励塑形改进
"""
import time
import random
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.config import (
    NUM_RUNS, 
    NUM_GENERATIONS, 
    SEED_BASE, 
    CONFIDENCE, 
    collect_statistics, 
    USE_MAHH,
    USE_REWARD_SHAPING
)
from src.neat_algorithms import (
    run_training, 
    eval_genomes_meta, 
    eval_genomes_vanilla
)
from src.evaluation import evaluate_genome
from src.visualization import (
    plot_performance_comparison, 
    plot_training_progression, 
    plot_local_search_statistics
)
from src.visualization_extended import (
    plot_mahh_statistics,
    plot_reward_shaping_statistics,
    plot_combined_learning_curves,
    plot_ablation_study_results
)
from src.statistics import compare_results

def run_experiment(use_mahh=False, use_reward_shaping=False, generation_settings=None, num_runs=None):
    """
    运行实验，可设置是否启用MAHH和奖励塑形
    
    Args:
        use_mahh (bool, optional): 是否启用MAHH. Defaults to False.
        use_reward_shaping (bool, optional): 是否启用奖励塑形. Defaults to False.
        generation_settings (list, optional): 代数设置列表. Defaults to None.
        num_runs (int, optional): 每种设置的运行次数. Defaults to None.
        
    Returns:
        tuple: (meta_results, vanilla_results, meta_training_data, vanilla_training_data)
    """
    # 导入配置模块并修改全局设置
    import src.config as config
    
    # 保存原始设置
    original_mahh = config.USE_MAHH
    original_reward_shaping = config.USE_REWARD_SHAPING
    
    # 应用新设置
    config.USE_MAHH = use_mahh
    config.USE_REWARD_SHAPING = use_reward_shaping
    
    # 设置实验参数
    if generation_settings is None:
        generation_settings = list(range(2, 51, 8))  # 增加步长以减少不同代数设置的数量
    
    if num_runs is None:
        num_runs = config.NUM_RUNS
    
    # 构建实验名称
    experiment_name = "Baseline"
    if use_mahh:
        experiment_name += "+MAHH"
    if use_reward_shaping:
        experiment_name += "+RS"
    
    print(f"\n=== 运行实验: {experiment_name} ===")
    
    meta_results = []     # 存储Meta-NEAT每组最终评估奖励
    vanilla_results = []  # 存储Vanilla NEAT每组最终评估奖励
    
    # 存储每次运行的进化过程数据
    all_meta_training_data = []
    all_vanilla_training_data = []

    # 使用tqdm包装generation_settings迭代，显示整体进度
    for gens in tqdm(generation_settings, desc="进化代数设置", leave=True):
        print(f"\n=== Generation Setting: {gens} generations ===")
        meta_run_rewards = []
        vanilla_run_rewards = []
        
        # 存储当前代数设置下的所有运行数据
        meta_training_data = []
        vanilla_training_data = []

        # 使用tqdm包装run迭代
        for run in tqdm(range(1, num_runs + 1), desc=f"运行 (代数={gens})", leave=True):
            # 设置此次运行的随机种子
            run_seed = SEED_BASE + run
            
            # 构建算法名称
            meta_name = "Meta-NEAT"
            vanilla_name = "Vanilla NEAT"
            
            if use_mahh:
                meta_name += "+MAHH"
                vanilla_name += "+MAHH"
            if use_reward_shaping:
                meta_name += "+RS"
                vanilla_name += "+RS"
            
            print(f"\n[{meta_name}] Run {run}/{num_runs} with {gens} generations (Seed: {run_seed})")
            start_time = time.time()
            winner_meta, config_meta, _, gen_fitnesses_meta = run_training(gens, eval_genomes_meta, run_seed, meta_name)
            elapsed_meta = time.time() - start_time
            avg_reward_meta = evaluate_genome(winner_meta, config_meta, num_eval_episodes=5)
            print(f"{meta_name} Run {run}: Average Reward = {avg_reward_meta:.2f}, Training Time = {elapsed_meta:.2f} seconds")
            meta_run_rewards.append(avg_reward_meta)
            meta_training_data.append(gen_fitnesses_meta)

            print(f"\n[{vanilla_name}] Run {run}/{num_runs} with {gens} generations (Seed: {run_seed})")
            start_time = time.time()
            winner_vanilla, config_vanilla, _, gen_fitnesses_vanilla = run_training(gens, eval_genomes_vanilla, run_seed, vanilla_name)
            elapsed_vanilla = time.time() - start_time
            avg_reward_vanilla = evaluate_genome(winner_vanilla, config_vanilla, num_eval_episodes=5)
            print(f"{vanilla_name} Run {run}: Average Reward = {avg_reward_vanilla:.2f}, Training Time = {elapsed_vanilla:.2f} seconds")
            vanilla_run_rewards.append(avg_reward_vanilla)
            vanilla_training_data.append(gen_fitnesses_vanilla)

        meta_results.append((gens, meta_run_rewards))
        vanilla_results.append((gens, vanilla_run_rewards))
        
        # 如果是最大代数的运行，保存进化过程数据用于绘制训练过程图
        if gens == max(generation_settings):
            all_meta_training_data = meta_training_data
            all_vanilla_training_data = vanilla_training_data

    # 恢复原始设置
    config.USE_MAHH = original_mahh
    config.USE_REWARD_SHAPING = original_reward_shaping
    
    return meta_results, vanilla_results, all_meta_training_data, all_vanilla_training_data

def main():
    """主函数：分别对Meta-NEAT和Vanilla NEAT进行训练对比，并生成对比图"""
    # 是否运行完整实验或简化实验
    run_full_experiment = False  # 设置为True以运行完整实验，包括所有代数设置
    
    if run_full_experiment:
        # 完整实验 - 使用多个代数设置
        generation_settings = list(range(2, 51, 8))
        num_runs = NUM_RUNS
    else:
        # 简化实验 - 仅使用几个代数设置进行快速验证
        generation_settings = [10, 20]
        num_runs = 3
    
    # 是否运行消融实验
    run_ablation_study = True
    
    if run_ablation_study:
        print("\n=== 运行消融实验 ===")
        print("这将测试不同组件的独立贡献...")
        
        # 使用固定的代数设置进行消融实验
        ablation_generations = 20
        ablation_runs = 5 if run_full_experiment else 3
        
        # 结果字典
        ablation_results = {}
        
        # 1. 基线实验 (不使用MAHH和奖励塑形)
        print("\n=== 基线实验 (Baseline) ===")
        _, _, baseline_meta_data, baseline_vanilla_data = run_experiment(
            use_mahh=False, 
            use_reward_shaping=False,
            generation_settings=[ablation_generations],
            num_runs=ablation_runs
        )
        
        # 收集最终适应度结果
        baseline_meta_rewards = [data[-1] for data in baseline_meta_data]
        baseline_vanilla_rewards = [data[-1] for data in baseline_vanilla_data]
        ablation_results["Baseline (Meta)"] = baseline_meta_rewards
        ablation_results["Baseline (Vanilla)"] = baseline_vanilla_rewards
        
        # 2. 仅使用MAHH
        print("\n=== 仅使用MAHH ===")
        _, _, mahh_meta_data, mahh_vanilla_data = run_experiment(
            use_mahh=True, 
            use_reward_shaping=False,
            generation_settings=[ablation_generations],
            num_runs=ablation_runs
        )
        
        # 收集最终适应度结果
        mahh_meta_rewards = [data[-1] for data in mahh_meta_data]
        mahh_vanilla_rewards = [data[-1] for data in mahh_vanilla_data]
        ablation_results["MAHH (Meta)"] = mahh_meta_rewards
        ablation_results["MAHH (Vanilla)"] = mahh_vanilla_rewards
        
        # 3. 仅使用奖励塑形
        print("\n=== 仅使用奖励塑形 ===")
        _, _, rs_meta_data, rs_vanilla_data = run_experiment(
            use_mahh=False, 
            use_reward_shaping=True,
            generation_settings=[ablation_generations],
            num_runs=ablation_runs
        )
        
        # 收集最终适应度结果
        rs_meta_rewards = [data[-1] for data in rs_meta_data]
        rs_vanilla_rewards = [data[-1] for data in rs_vanilla_data]
        ablation_results["RS (Meta)"] = rs_meta_rewards
        ablation_results["RS (Vanilla)"] = rs_vanilla_rewards
        
        # 4. 同时使用MAHH和奖励塑形
        print("\n=== 同时使用MAHH和奖励塑形 ===")
        _, _, combined_meta_data, combined_vanilla_data = run_experiment(
            use_mahh=True, 
            use_reward_shaping=True,
            generation_settings=[ablation_generations],
            num_runs=ablation_runs
        )
        
        # 收集最终适应度结果
        combined_meta_rewards = [data[-1] for data in combined_meta_data]
        combined_vanilla_rewards = [data[-1] for data in combined_vanilla_data]
        ablation_results["MAHH+RS (Meta)"] = combined_meta_rewards
        ablation_results["MAHH+RS (Vanilla)"] = combined_vanilla_rewards
        
        # 绘制消融实验结果
        plot_ablation_study_results(ablation_results, CONFIDENCE)
        
        # 绘制学习曲线比较
        print("\n=== 绘制学习曲线比较 ===")
        plot_combined_learning_curves(
            baseline_meta_data, 
            baseline_vanilla_data, 
            combined_meta_data, 
            combined_vanilla_data, 
            CONFIDENCE
        )
        
        # 如果启用了MAHH或奖励塑形，绘制相关统计图
        if USE_MAHH:
            plot_mahh_statistics()
        
        if USE_REWARD_SHAPING:
            plot_reward_shaping_statistics()
    else:
        # 运行标准实验
        meta_results, vanilla_results, all_meta_training_data, all_vanilla_training_data = run_experiment(
            use_mahh=USE_MAHH, 
            use_reward_shaping=USE_REWARD_SHAPING,
            generation_settings=generation_settings,
            num_runs=num_runs
        )
        
        # 输出各代数下的对比结果与统计分析
        compare_results(generation_settings, meta_results, vanilla_results, CONFIDENCE)
        
        # 绘制性能对比图
        plot_performance_comparison(generation_settings, meta_results, vanilla_results, CONFIDENCE)
        
        # 绘制训练过程对比图（使用最大代数设置的数据）
        plot_training_progression(all_meta_training_data, all_vanilla_training_data, CONFIDENCE)
        
        # 绘制局部搜索统计信息
        if collect_statistics:
            plot_local_search_statistics()
        
        # 如果启用了MAHH或奖励塑形，绘制相关统计图
        if USE_MAHH:
            plot_mahh_statistics()
        
        if USE_REWARD_SHAPING:
            plot_reward_shaping_statistics()

if __name__ == "__main__":
    main()
