"""
比较NEAT和混合代理的性能
"""
import os
import sys
import logging
import numpy as np
import neat
import torch
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.hybrid_agent import HybridAgent
from src.rl.q_learning import QAgent
from src.config import get_config

def setup_logging():
    """设置日志"""
    logger = logging.getLogger('AgentComparison')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    os.makedirs('logs', exist_ok=True)
    fh = logging.FileHandler('logs/agent_comparison.log')
    fh.setLevel(logging.INFO)
    
    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def evaluate_agent(agent, env, num_episodes):
    """评估代理性能
    
    Args:
        agent: 代理（NEAT或混合代理）
        env: 环境
        num_episodes (int): 评估episode数
        
    Returns:
        list: 每个episode的回报
    """
    returns = []
    
    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0.0
        
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_return += reward
            state = next_state
            
        returns.append(episode_return)
        
    return returns

def plot_comparison(results, title="Agent Performance Comparison"):
    """绘制性能比较图
    
    Args:
        results (dict): 各代理的回报结果
        title (str): 图表标题
    """
    plt.figure(figsize=(12, 6))
    
    colors = {
        'NEAT': 'blue',
        'Q-learning': 'red',
        'Hybrid': 'green'
    }
    
    # 计算移动平均
    window_size = min(100, len(next(iter(results.values()))))
    if window_size > 0:
        for name, returns in results.items():
            moving_avg = np.convolve(returns, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            plt.plot(moving_avg, label=f'{name} (Moving Average)', 
                    color=colors[name], linestyle='-')
    
    # 绘制原始数据
    for name, returns in results.items():
        plt.plot(returns, label=name, alpha=0.3, color=colors[name])
    
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    plt.savefig('agent_comparison.png')
    plt.close()

def init_neat_population(env, config_path):
    """初始化NEAT种群
    
    Args:
        env: 环境
        config_path (str): NEAT配置文件路径
        
    Returns:
        neat.Population: 初始化后的NEAT种群
    """
    # 加载NEAT配置
    neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    
    # 创建种群
    population = neat.Population(neat_config)
    
    # 添加统计报告
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # 定义评估函数
    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            state, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            
            while not (done or truncated):
                action_values = net.activate(state)
                action = np.argmax(action_values)
                next_state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                state = next_state
                
            genome.fitness = total_reward
    
    # 运行一代以初始化
    population.run(eval_genomes, 1)
    
    return population

def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    logger.info("Starting agent comparison")
    
    # 创建环境
    env = gym.make('CartPole-v1')
    logger.info(f"Environment: {env.spec.id}")
    
    # 加载配置
    config = get_config()
    
    # 初始化NEAT种群
    neat_pop = init_neat_population(env, 'config/neat_config.txt')
    
    # 创建Q-learning代理
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    q_agent = QAgent(state_size, action_size, config)
    
    # 创建不同模式的代理
    neat_agent = HybridAgent(config, neat_pop, q_agent, mode='neat')
    q_learning_agent = HybridAgent(config, neat_pop, q_agent, mode='q')
    hybrid_agent = HybridAgent(config, neat_pop, q_agent, mode='hybrid')
    
    # 评估参数
    num_episodes = 500
    
    # 评估各个代理
    results = {}
    
    logger.info("Evaluating NEAT agent...")
    results['NEAT'] = evaluate_agent(neat_agent, env, num_episodes)
    logger.info(f"NEAT average return: {np.mean(results['NEAT']):.2f} ± {np.std(results['NEAT']):.2f}")
    
    logger.info("Evaluating Q-learning agent...")
    results['Q-learning'] = evaluate_agent(q_learning_agent, env, num_episodes)
    logger.info(f"Q-learning average return: {np.mean(results['Q-learning']):.2f} ± {np.std(results['Q-learning']):.2f}")
    
    logger.info("Evaluating Hybrid agent...")
    results['Hybrid'] = evaluate_agent(hybrid_agent, env, num_episodes)
    logger.info(f"Hybrid average return: {np.mean(results['Hybrid']):.2f} ± {np.std(results['Hybrid']):.2f}")
    
    # 绘制比较图
    plot_comparison(results)
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main() 