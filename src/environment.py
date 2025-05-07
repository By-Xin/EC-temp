"""
环境相关模块
包含环境创建和交互的函数
"""
import gym
import neat
import numpy as np
from src.config import EP_STEP, USE_SOFTMAX
from src.utils import softmax

def make_env(seed=None):
    """创建单一环境，使用不同的随机种子
    
    Args:
        seed (int, optional): 随机种子. Defaults to None.
    
    Returns:
        gym.Env: 创建的环境
    """
    from src.config import GAME
    env = gym.make(GAME)
    if seed is not None:
        env.reset(seed=seed)
    return env

def evaluate_single_genome(genome, config, env):
    """单次评估函数：在给定环境中运行一个episode，返回累计奖励
    
    Args:
        genome (neat.DefaultGenome): 要评估的基因组
        config (neat.Config): NEAT配置
        env (gym.Env): 评估环境
    
    Returns:
        float: 累计奖励
    """
    from src.config import TAU
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    observation, _ = env.reset()
    total_reward = 0.0
    done = False
    truncated = False
    
    for _ in range(EP_STEP):
        action_values = net.activate(observation)
        if USE_SOFTMAX:
            probs = softmax(action_values, TAU)
            action = np.random.choice(len(probs), p=probs)
        else:
            action = np.argmax(action_values)
            
        observation, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        if done or truncated:
            break
            
    return total_reward
