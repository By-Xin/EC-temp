"""
混合代理训练脚本
"""
import os
import sys
import logging
import numpy as np
from tqdm import tqdm
import neat
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    NUM_RUNS, NUM_GENERATIONS, SEED_BASE, CONFIDENCE,
    P_INIT, BETA_SIGMOID, NEAT_EVAL_PERIOD, POPULATION_SIZE,
    REPLAY_CAPACITY, BATCH_SIZE, Q_HIDDEN_DIMS, LR_Q, GAMMA,
    ALPHA_MIXED_REWARD, LOG_DIR, MODEL_DIR
)
from src.environment import make_env
from src.rl.q_learning import QAgent
from src.hybrid_agent import HybridAgent
from src.evaluation import evaluate_genome
from src.visualization import plot_training_progression
from src.visualization_extended import plot_hybrid_results

# 获取配置文件路径
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'neat_config.txt')

def setup_logging(log_dir):
    """设置日志
    
    Args:
        log_dir (str): 日志目录
    """
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'hybrid_training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("HybridTraining")

def train_hybrid_agent(env, agent, num_episodes, logger):
    """训练混合代理
    
    Args:
        env: 环境
        agent: 混合代理
        num_episodes (int): 训练episode数
        logger: 日志记录器
    """
    returns = []
    p_values = []
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0.0
        
        while not (done or truncated):
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            
            # 可选：使用Q值进行奖励塑形
            if ALPHA_MIXED_REWARD > 0:
                q_val = agent.q_agent.estimate_value(state, action)
                reward = reward + ALPHA_MIXED_REWARD * q_val
            
            # 观察和学习
            agent.observe(state, action, reward, next_state, done or truncated)
            agent.learn(done or truncated)
            
            state = next_state
            episode_return += reward
            
        # 记录结果
        returns.append(episode_return)
        p_values.append(agent.p)
        
        # 定期更新目标网络
        if episode % 10 == 0:
            agent.q_agent.update_target_network()
            
        # 记录日志
        if episode % 100 == 0:
            logger.info(f"Episode {episode}: Return = {episode_return:.2f}, p = {agent.p:.3f}")
            
    return returns, p_values

def main(args):
    """主函数
    
    Args:
        args: 命令行参数
    """
    # 设置日志
    logger = setup_logging(args.log_dir)
    logger.info("Starting hybrid agent training")
    logger.info(f"Arguments: {args}")
    
    # 创建环境
    env = make_env()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 打印环境信息
    logger.info(f"Environment: {env.unwrapped.spec.id}")
    logger.info(f"State space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    
    # 创建NEAT种群
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        CONFIG_PATH)
    neat_pop = neat.Population(config)
    
    # 创建Q-learning代理
    q_agent = QAgent(state_size, action_size, {
        'Q_HIDDEN_DIMS': Q_HIDDEN_DIMS,
        'LR_Q': LR_Q,
        'GAMMA': GAMMA,
        'REPLAY_CAPACITY': REPLAY_CAPACITY,
        'BATCH_SIZE': BATCH_SIZE
    })
    
    # 创建混合代理
    agent = HybridAgent({
        'P_INIT': args.p_init,
        'BETA_SIGMOID': args.beta_sigmoid,
        'NEAT_EVAL_PERIOD': args.neat_eval_period,
        'REPLAY_CAPACITY': REPLAY_CAPACITY
    }, neat_pop, q_agent)
    
    # 训练
    returns, p_values = train_hybrid_agent(env, agent, args.num_episodes, logger)
    
    # 保存模型
    os.makedirs(args.model_dir, exist_ok=True)
    agent.save(os.path.join(args.model_dir, 'hybrid_agent'))
    
    # 绘制训练结果
    plot_training_progression(returns, "Hybrid Agent Training")
    plot_hybrid_results(returns, p_values, "Hybrid Agent Results")
    
    # 与普通NEAT进行对比
    logger.info("Comparing with vanilla NEAT...")
    hybrid_mean, neat_mean = agent.compare_with_neat(env, num_episodes=args.eval_episodes)
    logger.info(f"Hybrid Agent Mean Return: {hybrid_mean:.2f}")
    logger.info(f"Vanilla NEAT Mean Return: {neat_mean:.2f}")
    
    # 绘制对比结果
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.bar(['Hybrid Agent', 'Vanilla NEAT'], [hybrid_mean, neat_mean])
    plt.title('Performance Comparison')
    plt.ylabel('Mean Return')
    plt.savefig(os.path.join(args.log_dir, 'performance_comparison.png'))
    plt.close()
    
    logger.info("Training and comparison completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='混合代理训练脚本')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=100,
                      help='训练轮数 (默认: 100)')
    parser.add_argument('--eval_episodes', type=int, default=20,
                      help='评估轮数 (默认: 20)')
    
    # 混合代理参数
    parser.add_argument('--p_init', type=float, default=P_INIT,
                      help='初始Q-learning概率 (默认: 0.5)')
    parser.add_argument('--beta_sigmoid', type=float, default=BETA_SIGMOID,
                      help='sigmoid函数温度参数 (默认: 0.1)')
    parser.add_argument('--neat_eval_period', type=int, default=NEAT_EVAL_PERIOD,
                      help='NEAT评估周期 (默认: 10)')
    
    # 路径参数
    parser.add_argument('--log_dir', type=str, default=LOG_DIR,
                      help='日志保存目录')
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR,
                      help='模型保存目录')
    
    args = parser.parse_args()
    main(args) 