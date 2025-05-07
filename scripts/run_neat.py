"""
NEAT代理训练脚本
"""
import os
import sys
import logging
import numpy as np
import neat
from tqdm import tqdm
import gymnasium as gym
from src.visualization import plot_training_progression

def setup_logging():
    """设置日志"""
    logger = logging.getLogger('NEATTraining')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    os.makedirs('logs', exist_ok=True)
    fh = logging.FileHandler('logs/neat_training.log')
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

def create_eval_function(env):
    """创建评估函数
    
    Args:
        env: 环境
        
    Returns:
        function: 评估函数
    """
    def eval_genomes(genomes, config):
        """评估基因组种群
        
        Args:
            genomes (list): 基因组列表
            config (neat.Config): NEAT配置
        """
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            state, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            
            while not (done or truncated):
                # 获取动作
                action_values = net.activate(state)
                action = np.argmax(action_values)
                
                # 执行动作
                next_state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                state = next_state
                
            genome.fitness = total_reward
            
    return eval_genomes

def train_neat_agent(env, config_path, num_episodes, logger):
    """训练NEAT代理
    
    Args:
        env: 环境
        config_path (str): NEAT配置文件路径
        num_episodes (int): 训练episode数
        logger: 日志记录器
        
    Returns:
        list: 每代的平均回报
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
    
    # 创建评估函数
    eval_function = create_eval_function(env)
    
    # 训练循环
    returns = []
    best_genome = None
    best_fitness = float('-inf')
    
    for generation in tqdm(range(num_episodes), desc="Training"):
        # 评估所有基因组
        eval_function(list(population.population.items()), neat_config)
        
        # 更新最佳基因组
        for genome_id, genome in population.population.items():
            if genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_genome = genome
        
        # 记录平均回报
        avg_return = np.mean([genome.fitness for genome in population.population.values()])
        returns.append(avg_return)
        
        # 记录日志
        logger.info(f"Generation {generation}: Average Return = {avg_return:.2f}, Best Return = {best_fitness:.2f}")
        
        # 进化一代
        population.run(eval_function, 1)
    
    return returns

def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    logger.info("Starting NEAT agent training")
    
    # 创建环境
    env = gym.make('CartPole-v1')
    logger.info(f"Environment: {env.spec.id}")
    logger.info(f"State space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    
    # 训练参数
    num_episodes = 200
    config_path = 'config/neat_config.txt'
    
    # 训练代理
    returns = train_neat_agent(env, config_path, num_episodes, logger)
    
    # 绘制训练进度
    plot_training_progression(returns, "NEAT Agent Training")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main() 