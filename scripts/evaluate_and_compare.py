"""
评估和对比脚本
用于加载训练好的模型并进行性能对比
"""
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import neat
import torch
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid_agent import HybridAgent
from src.rl.q_learning import QAgent
from src.environment import create_environment
from src.config import get_config

def evaluate_agent(agent, env, num_episodes=100):
    """评估代理性能
    
    Args:
        agent: 要评估的代理
        env: 环境
        num_episodes (int): 评估的episode数
        
    Returns:
        list: 每个episode的回报
    """
    returns = []
    
    for _ in tqdm(range(num_episodes), desc="Evaluating agent"):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action = agent.act(state)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        
        returns.append(total_reward)
    
    return returns

def load_and_evaluate_all():
    """加载所有模型并评估性能"""
    # 获取配置
    config = get_config()
    
    # 创建环境
    env = create_environment()
    
    # 加载并评估混合代理
    print("\nEvaluating Hybrid Agent...")
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        'config/neat_config.txt'
    )
    with open('models/hybrid_agent_final_neat.pkl', 'rb') as f:
        neat_pop = pickle.load(f)
    q_agent = QAgent(env.observation_space.shape[0], env.action_space.n, config)
    q_agent.load('models/hybrid_agent_final_q.pth')
    hybrid_agent = HybridAgent(config, neat_pop, q_agent, mode='hybrid')
    hybrid_agent.load('models/hybrid_agent_final')
    hybrid_returns = evaluate_agent(hybrid_agent, env)
    
    # 加载并评估Q-learning代理
    print("\nEvaluating Q-learning Agent...")
    q_agent = QAgent(env.observation_space.shape[0], env.action_space.n, config)
    q_agent.load('models/q_agent_final.pth')
    q_returns = evaluate_agent(q_agent, env)
    
    # 加载并评估NEAT代理
    print("\nEvaluating NEAT Agent...")
    with open('models/neat_best_genome.pkl', 'rb') as f:
        best_genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(best_genome, neat_config)
    
    def neat_act(state):
        action_values = net.activate(state)
        return np.argmax(action_values)
    
    neat_returns = evaluate_agent(type('obj', (object,), {'act': neat_act}), env)
    
    return hybrid_returns, q_returns, neat_returns

def plot_comparison(hybrid_returns, q_returns, neat_returns):
    """绘制性能对比图
    
    Args:
        hybrid_returns (list): 混合代理的回报
        q_returns (list): Q-learning代理的回报
        neat_returns (list): NEAT代理的回报
    """
    plt.figure(figsize=(10, 6))
    
    # 计算移动平均
    window = 10
    hybrid_ma = np.convolve(hybrid_returns, np.ones(window)/window, mode='valid')
    q_ma = np.convolve(q_returns, np.ones(window)/window, mode='valid')
    neat_ma = np.convolve(neat_returns, np.ones(window)/window, mode='valid')
    
    # 绘制移动平均线
    plt.plot(hybrid_ma, label='Hybrid', color='blue')
    plt.plot(q_ma, label='Q-learning', color='red')
    plt.plot(neat_ma, label='NEAT', color='green')
    
    # 添加图例和标签
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Agent Performance Comparison')
    
    # 保存图片
    plt.savefig('agent_comparison.png')
    plt.close()

def main():
    """主函数"""
    # 加载并评估所有代理
    hybrid_returns, q_returns, neat_returns = load_and_evaluate_all()
    
    # 打印平均回报
    print("\nAverage Returns:")
    print(f"Hybrid: {np.mean(hybrid_returns):.2f} ± {np.std(hybrid_returns):.2f}")
    print(f"Q-learning: {np.mean(q_returns):.2f} ± {np.std(q_returns):.2f}")
    print(f"NEAT: {np.mean(neat_returns):.2f} ± {np.std(neat_returns):.2f}")
    
    # 绘制对比图
    plot_comparison(hybrid_returns, q_returns, neat_returns)
    print("\nComparison plot saved as 'agent_comparison.png'")

if __name__ == "__main__":
    main() 