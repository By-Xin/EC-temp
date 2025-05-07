"""
混合代理模块
结合NEAT和Q-learning的混合代理实现
"""
import os
import pickle
import numpy as np
import torch
import neat
from src.rl.q_learning import QAgent
from src.rl.replay_buffer import ReplayBuffer
from src.config import P_INIT, BETA_SIGMOID, NEAT_EVAL_PERIOD, REPLAY_CAPACITY
import logging

class HybridAgent:
    """混合代理，结合NEAT和Q-learning"""
    
    def __init__(self, config, neat_pop, q_agent, mode='hybrid'):
        """初始化混合代理
        
        Args:
            config (dict): 配置参数
            neat_pop (neat.Population): NEAT种群
            q_agent (QAgent): Q-learning代理
            mode (str): 运行模式，可选值：
                - 'hybrid': 混合模式，动态调整NEAT和Q-learning的使用比例
                - 'neat': 纯NEAT模式
                - 'q': 纯Q-learning模式
        """
        self.config = config
        self.neat_pop = neat_pop
        self.q_agent = q_agent
        self.mode = mode
        self.episode_count = 0
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(config['REPLAY_CAPACITY'])
        
        # 初始化NEAT评估计数器
        self.neat_eval_counter = 0
        
        # 创建日志目录
        os.makedirs(config['LOG_DIR'], exist_ok=True)
        os.makedirs(config['MODEL_DIR'], exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger('HybridAgent')
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        fh = logging.FileHandler(os.path.join(config['LOG_DIR'], 'hybrid_agent.log'))
        fh.setLevel(logging.INFO)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info("Hybrid agent initialized")
        self.logger.info(f"Mode: {mode}")
        
    def act(self, state):
        """选择动作
        
        Args:
            state (np.ndarray): 当前状态
            
        Returns:
            int: 选择的动作
        """
        if self.mode == 'neat':
            # 纯NEAT模式
            best_genome = self.neat_pop.best_genome
            net = neat.nn.FeedForwardNetwork.create(best_genome, self.neat_pop.config)
            action_values = net.activate(state)
            return np.argmax(action_values)
            
        elif self.mode == 'q':
            # 纯Q-learning模式
            return self.q_agent.act(state)
            
        else:  # hybrid模式
            # 计算NEAT的使用概率
            p = self._calculate_neat_probability()
            
            # 根据概率选择使用NEAT还是Q-learning
            if np.random.random() < p:
                # 使用NEAT
                best_genome = self.neat_pop.best_genome
                net = neat.nn.FeedForwardNetwork.create(best_genome, self.neat_pop.config)
                action_values = net.activate(state)
                return np.argmax(action_values)
            else:
                # 使用Q-learning
                return self.q_agent.act(state)
    
    def _calculate_neat_probability(self):
        """计算使用NEAT的概率
        
        Returns:
            float: NEAT的使用概率
        """
        # 使用sigmoid函数计算概率
        x = self.episode_count - self.config['P_INIT']
        return 1 / (1 + np.exp(-self.config['BETA_SIGMOID'] * x))
    
    def update(self, state, action, reward, next_state, done):
        """更新代理
        
        Args:
            state (np.ndarray): 当前状态
            action (int): 执行的动作
            reward (float): 获得的奖励
            next_state (np.ndarray): 下一个状态
            done (bool): 是否结束
        """
        # 更新episode计数
        if done:
            self.episode_count += 1
        
        # 存储经验
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # 更新Q-learning代理
        if len(self.replay_buffer) >= self.config['BATCH_SIZE']:
            batch = self.replay_buffer.sample(self.config['BATCH_SIZE'])
            self.q_agent.learn(batch)
        
        # 定期评估NEAT种群
        if self.mode in ['hybrid', 'neat']:
            self.neat_eval_counter += 1
            if self.neat_eval_counter >= self.config['NEAT_EVAL_PERIOD']:
                self._evaluate_neat_population()
                self.neat_eval_counter = 0
    
    def _evaluate_neat_population(self):
        """评估NEAT种群"""
        self.logger.info("Evaluating NEAT population...")
        
        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                # 从经验回放缓冲区采样状态
                if len(self.replay_buffer) >= self.config['BATCH_SIZE']:
                    batch = self.replay_buffer.sample(self.config['BATCH_SIZE'])
                    states = batch[0]  # 只使用状态
                    
                    # 计算基因组的总奖励
                    total_reward = 0.0
                    for state in states:
                        net = neat.nn.FeedForwardNetwork.create(genome, config)
                        action_values = net.activate(state)
                        action = np.argmax(action_values)
                        # 使用Q-learning代理的奖励估计
                        q_value = self.q_agent.get_q_value(state, action)
                        total_reward += q_value
                    
                    genome.fitness = total_reward
                else:
                    genome.fitness = 0.0
        
        # 运行一代
        self.neat_pop.run(eval_genomes, 1)
        
        # 记录最佳基因组
        best_genome = self.neat_pop.best_genome
        self.logger.info(f"Best genome fitness: {best_genome.fitness:.2f}")
        
        # 保存最佳基因组
        if best_genome.fitness > 0:
            self._save_best_genome(best_genome)
    
    def _save_best_genome(self, genome):
        """保存最佳基因组
        
        Args:
            genome (neat.DefaultGenome): 要保存的基因组
        """
        # 创建保存目录
        os.makedirs(self.config['MODEL_DIR'], exist_ok=True)
        
        # 保存基因组
        with open(os.path.join(self.config['MODEL_DIR'], 'best_genome.pkl'), 'wb') as f:
            pickle.dump(genome, f)
        
        self.logger.info("Best genome saved")
        
    def save(self, path):
        """保存模型
        
        Args:
            path (str): 保存路径
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存NEAT种群
        with open(f"{path}_neat.pkl", 'wb') as f:
            pickle.dump(self.neat_pop, f)
            
        # 保存Q-learning模型
        self.q_agent.save(f"{path}_q.pth")
        
        # 保存混合代理状态
        state = {
            'p': self.p,
            'beta': self.beta,
            'episode_count': self.episode_count,
            'neat_fitness': self.neat_fitness,
            'q_fitness': self.q_fitness
        }
        with open(f"{path}_state.pkl", 'wb') as f:
            pickle.dump(state, f)
            
    def load(self, path):
        """加载模型
        
        Args:
            path (str): 加载路径
        """
        # 加载NEAT种群
        with open(f"{path}_neat.pkl", 'rb') as f:
            self.neat_pop = pickle.load(f)
            
        # 加载Q-learning模型
        self.q_agent.load(f"{path}_q.pth")
        
        # 加载混合代理状态
        with open(f"{path}_state.pkl", 'rb') as f:
            state = pickle.load(f)
            self.p = state['p']
            self.beta = state['beta']
            self.episode_count = state['episode_count']
            self.neat_fitness = state['neat_fitness']
            self.q_fitness = state['q_fitness']
            
    def compare_with_neat(self, env, num_episodes=100):
        """与普通NEAT进行对比
        
        Args:
            env: 环境
            num_episodes (int): 评估的episode数
            
        Returns:
            tuple: (混合代理平均回报, NEAT平均回报)
        """
        # 评估混合代理
        hybrid_returns = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_return = 0.0
            
            while not (done or truncated):
                action = self.act(state)
                next_state, reward, done, truncated, _ = env.step(action)
                episode_return += reward
                state = next_state
                
            hybrid_returns.append(episode_return)
            
        # 评估普通NEAT
        neat_returns = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_return = 0.0
            
            while not (done or truncated):
                best_genome = self.neat_pop.best_genome
                net = neat.nn.FeedForwardNetwork.create(best_genome, self.neat_pop.config)
                action_values = net.activate(state)
                action = np.argmax(action_values)
                next_state, reward, done, truncated, _ = env.step(action)
                episode_return += reward
                state = next_state
                
            neat_returns.append(episode_return)
            
        return np.mean(hybrid_returns), np.mean(neat_returns) 