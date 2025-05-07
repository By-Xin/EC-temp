"""
混合代理训练脚本
"""
import os
import sys
import numpy as np
import neat
import torch
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid_agent import HybridAgent
from src.rl.q_learning import QAgent
from src.environment import create_environment
from src.config import get_config

def train_hybrid(num_episodes=1000):
    """训练混合代理
    
    Args:
        num_episodes (int): 训练的总episode数
    """
    # 获取配置
    config = get_config()
    
    # 创建环境
    env = create_environment()
    
    # 创建NEAT种群
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        'config/neat_config.txt'
    )
    neat_pop = neat.Population(neat_config)
    
    # 创建Q-learning代理
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    q_agent = QAgent(state_size, action_size, config)
    
    # 创建混合代理
    agent = HybridAgent(config, neat_pop, q_agent, mode='hybrid')
    
    # 训练循环
    returns = []
    for episode in tqdm(range(num_episodes), desc="Training Hybrid Agent"):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            
            # 更新代理
            agent.update(state, action, reward, next_state, done)
            
            # 更新状态和总奖励
            state = next_state
            total_reward += reward
        
        returns.append(total_reward)
        
        # 每100个episode保存一次模型
        if (episode + 1) % 100 == 0:
            agent.save(f"models/hybrid_agent_episode_{episode+1}")
    
    # 保存最终模型
    agent.save("models/hybrid_agent_final")
    
    return returns

if __name__ == "__main__":
    returns = train_hybrid()
    print(f"Training completed. Final average return: {np.mean(returns[-100:]):.2f}") 