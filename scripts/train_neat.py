"""
NEAT代理训练脚本
"""
import os
import sys
import pickle
import numpy as np
import neat
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import create_environment
from src.config import get_config

def eval_genomes(genomes, config):
    """评估基因组
    
    Args:
        genomes (list): 基因组列表
        config (neat.Config): NEAT配置
    """
    env = create_environment()
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0
        
        # 运行5个episode取平均
        for _ in range(5):
            state, _ = env.reset()
            done = False
            truncated = False
            
            while not (done or truncated):
                # 获取动作
                action_values = net.activate(state)
                action = np.argmax(action_values)
                
                # 执行动作
                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
        
        # 设置适应度
        genome.fitness = total_reward / 5

def train_neat(num_generations=100):
    """训练NEAT代理
    
    Args:
        num_generations (int): 训练的代数
    """
    # 加载NEAT配置
    config_path = os.path.join('config', 'neat_config.txt')
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    # 创建种群
    population = neat.Population(config)
    
    # 添加统计报告
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # 训练
    winner = population.run(eval_genomes, num_generations)
    
    # 保存最佳基因组
    os.makedirs('models', exist_ok=True)
    with open('models/neat_best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    return winner, stats

if __name__ == "__main__":
    winner, stats = train_neat()
    print(f"Training completed. Best genome fitness: {winner.fitness:.2f}") 