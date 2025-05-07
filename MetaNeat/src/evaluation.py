"""
评估模块
包含基因组评估相关的函数
"""
import neat
import copy
import random
import numpy as np
from tqdm.auto import tqdm

from src.config import (
    EP_STEP, 
    BASE_LOCAL_TRIALS, 
    MAX_LOCAL_TRIALS, 
    local_search_stats, 
    collect_statistics,
    VERBOSE)
from src.utils import log_print
from src.environment import make_env, evaluate_single_genome
import gym
def mutate_genome(genome, config):
    """基因组轻度变异函数：用于内循环局部适应
    
    Args:
        genome (neat.DefaultGenome): 要变异的基因组
        config (neat.Config): NEAT配置
    """
    for cg in genome.connections.values():
        if random.random() < 0.6:
            cg.weight += random.gauss(0, 0.1)

def local_adaptation(genome, config, env, gen_progress=0.5):
    """内循环：对单个genome进行局部适应（局部搜索）
    
    对给定genome在单一环境中进行泊松分布随机次数的轻微变异尝试，
    并返回表现最佳的变异版本及其奖励。
    
    Args:
        genome (neat.DefaultGenome): 要进行局部适应的基因组
        config (neat.Config): NEAT配置
        env (gym.Env): 环境
        gen_progress (float, optional): 当前进化的进度 (0-1)，用于自适应调整lambda参数. Defaults to 0.5.
    
    Returns:
        tuple: (最佳基因组, 最佳奖励)
    """
    log_print("开始局部适应")
    best_genome = copy.deepcopy(genome)
    best_reward = evaluate_single_genome(best_genome, config, env)
    log_print(f"基准奖励: {best_reward}")
    
    # 根据当前进化阶段调整lambda值
    if gen_progress < 0.3:
        lambda_param = BASE_LOCAL_TRIALS * 1.2  # 早期阶段增加局部搜索
    elif gen_progress > 0.7:
        lambda_param = BASE_LOCAL_TRIALS * 0.8  # 后期阶段减少局部搜索
    else:
        lambda_param = BASE_LOCAL_TRIALS
    
    # 使用泊松分布生成局部搜索次数
    local_trials = np.random.poisson(lambda_param)
    
    # 设置上下限，避免极端值
    local_trials = max(1, min(local_trials, MAX_LOCAL_TRIALS))
    
    log_print(f"当前局部搜索次数: {local_trials} (lambda={lambda_param:.2f})")
    
    # 统计信息
    improvements = 0
    
    # 使用tqdm进度条，只在VERBOSE=True时显示
    trials_iterator = range(local_trials)
    if VERBOSE:
        trials_iterator = tqdm(trials_iterator, desc="局部适应试验", leave=False)
    
    for i in trials_iterator:
        log_print(f"局部适应试验 {i+1}/{local_trials}")
        mutated = copy.deepcopy(genome)
        mutate_genome(mutated, config)
        reward = evaluate_single_genome(mutated, config, env)
        log_print(f"变异后奖励: {reward}")
        if reward > best_reward:
            best_reward = reward
            best_genome = mutated
            improvements += 1
            log_print(f"更新最佳奖励: {best_reward}")
    
    # 收集统计信息
    if collect_statistics:
        local_search_stats['trials'].append(local_trials)
        local_search_stats['improvements'].append(improvements)
        local_search_stats['improvement_ratios'].append(improvements / local_trials if local_trials > 0 else 0)
    
    log_print(f"局部适应完成。总尝试次数: {local_trials}, 有效改进次数: {improvements}")
    return best_genome, best_reward

def evaluate_genome(winner, config, num_eval_episodes=5):
    """最终评估函数：在标准环境下对最佳genome进行评估
    
    Args:
        winner (neat.DefaultGenome): 要评估的最优基因组
        config (neat.Config): NEAT配置
        num_eval_episodes (int, optional): 评估轮数. Defaults to 5.
    
    Returns:
        float: 平均奖励
    """
    from src.config import GAME, VERBOSE
    env = gym.make(GAME)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    rewards = []
    
    # 使用tqdm包装评估迭代
    episodes_iterator = range(num_eval_episodes)
    if not VERBOSE:
        episodes_iterator = tqdm(episodes_iterator, desc="最终评估", leave=False)
    
    for _ in episodes_iterator:
        observation, _ = env.reset()
        total_reward = 0.0
        while True:
            action_values = net.activate(observation)
            action = np.argmax(action_values)  # 使用贪婪策略选择动作
            observation, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        rewards.append(total_reward)
    env.close()
    return np.mean(rewards)
