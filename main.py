"""
主程序
Meta-NEAT与Vanilla NEAT对比实验
"""
import time
import random
import numpy as np
from tqdm.auto import tqdm

from src.config import (
    NUM_RUNS, 
    NUM_GENERATIONS, 
    SEED_BASE, 
    CONFIDENCE, 
    collect_statistics
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
from src.statistics import compare_results

def main():
    """主函数：分别对Meta-NEAT和Vanilla NEAT进行训练对比，并生成对比图"""
    meta_results = []     # 存储Meta-NEAT每组最终评估奖励
    vanilla_results = []  # 存储Vanilla NEAT每组最终评估奖励
    
    # 存储每次运行的进化过程数据
    all_meta_training_data = []
    all_vanilla_training_data = []

    # 测试更多不同的进化代数设置，减少颗粒度以节省时间
    generation_settings = list(range(2, 51, 8))  # 增加步长以减少不同代数设置的数量
    
    # 使用tqdm包装generation_settings迭代，显示整体进度
    for gens in tqdm(generation_settings, desc="进化代数设置", leave=True):
        print(f"\n=== Generation Setting: {gens} generations ===")
        meta_run_rewards = []
        vanilla_run_rewards = []
        
        # 存储当前代数设置下的所有运行数据
        meta_training_data = []
        vanilla_training_data = []

        # 使用tqdm包装run迭代
        for run in tqdm(range(1, NUM_RUNS + 1), desc=f"运行 (代数={gens})", leave=True):
            # 设置此次运行的随机种子
            run_seed = SEED_BASE + run
            
            print(f"\n[Meta-NEAT] Run {run}/{NUM_RUNS} with {gens} generations (Seed: {run_seed})")
            start_time = time.time()
            winner_meta, config_meta, _, gen_fitnesses_meta = run_training(gens, eval_genomes_meta, run_seed, "Meta-NEAT")
            elapsed_meta = time.time() - start_time
            avg_reward_meta = evaluate_genome(winner_meta, config_meta, num_eval_episodes=5)
            print(f"Meta-NEAT Run {run}: Average Reward = {avg_reward_meta:.2f}, Training Time = {elapsed_meta:.2f} seconds")
            meta_run_rewards.append(avg_reward_meta)
            meta_training_data.append(gen_fitnesses_meta)

            print(f"\n[Vanilla NEAT] Run {run}/{NUM_RUNS} with {gens} generations (Seed: {run_seed})")
            start_time = time.time()
            winner_vanilla, config_vanilla, _, gen_fitnesses_vanilla = run_training(gens, eval_genomes_vanilla, run_seed, "Vanilla NEAT")
            elapsed_vanilla = time.time() - start_time
            avg_reward_vanilla = evaluate_genome(winner_vanilla, config_vanilla, num_eval_episodes=5)
            print(f"Vanilla NEAT Run {run}: Average Reward = {avg_reward_vanilla:.2f}, Training Time = {elapsed_vanilla:.2f} seconds")
            vanilla_run_rewards.append(avg_reward_vanilla)
            vanilla_training_data.append(gen_fitnesses_vanilla)

        meta_results.append((gens, meta_run_rewards))
        vanilla_results.append((gens, vanilla_run_rewards))
        
        # 如果是最大代数的运行，保存进化过程数据用于绘制训练过程图
        if gens == max(generation_settings):
            all_meta_training_data = meta_training_data
            all_vanilla_training_data = vanilla_training_data

    # 输出各代数下的对比结果与统计分析
    compare_results(generation_settings, meta_results, vanilla_results, CONFIDENCE)
    
    # 绘制性能对比图
    plot_performance_comparison(generation_settings, meta_results, vanilla_results, CONFIDENCE)
    
    # 绘制训练过程对比图（使用最大代数设置的数据）
    plot_training_progression(all_meta_training_data, all_vanilla_training_data, CONFIDENCE)
    
    # 绘制局部搜索统计信息
    if collect_statistics:
        plot_local_search_statistics()

if __name__ == "__main__":
    main()
