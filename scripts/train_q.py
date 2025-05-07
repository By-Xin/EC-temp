"""
Q-learning代理训练脚本
"""
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.q_learning import QAgent
from src.environment import create_environment
from src.config import get_config

def train_q(num_episodes=1000):
    """训练Q-learning代理
    
    Args:
        num_episodes (int): 训练的总episode数
    """
    # 获取配置
    config = get_config()
    
    # 创建环境
    env = create_environment()
    
    # 创建Q-learning代理
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = QAgent(state_size, action_size, config)
    
    # 训练循环
    returns = []
    for episode in tqdm(range(num_episodes), desc="Training Q-learning Agent"):
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
            agent.learn((state, action, reward, next_state, done))
            
            # 更新状态和总奖励
            state = next_state
            total_reward += reward
        
        returns.append(total_reward)
        
        # 每100个episode保存一次模型
        if (episode + 1) % 100 == 0:
            agent.save(f"models/q_agent_episode_{episode+1}.pth")
    
    # 保存最终模型
    agent.save("models/q_agent_final.pth")
    
    return returns

if __name__ == "__main__":
    returns = train_q()
    print(f"Training completed. Final average return: {np.mean(returns[-100:]):.2f}") 