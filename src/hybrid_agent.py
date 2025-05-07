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

class HybridAgent:
    """混合代理类，结合NEAT和Q-learning"""
    
    def __init__(self, config, neat_pop, q_agent):
        """初始化混合代理
        
        Args:
            config (dict): 配置参数字典
            neat_pop (neat.Population): NEAT种群
            q_agent (QAgent): Q-learning代理
        """
        self.config = config
        self.neat_pop = neat_pop
        self.q_agent = q_agent
        self.replay_buffer = ReplayBuffer(config['REPLAY_CAPACITY'])
        
        # 混合参数
        self.p = config['P_INIT']  # Q-learning的概率
        self.beta = config['BETA_SIGMOID']  # sigmoid函数的温度参数
        self.neat_eval_period = config['NEAT_EVAL_PERIOD']
        
        # 训练统计
        self.episode_count = 0
        self.neat_fitness = []
        self.q_fitness = []
        
        # 初始化NEAT种群
        self._initialize_neat_population()
        
    def _initialize_neat_population(self):
        """初始化NEAT种群"""
        # 为每个基因组设置初始适应度
        for genome in self.neat_pop.population.values():
            genome.fitness = 0.0
            
        # 运行一代以初始化种群
        self.neat_pop.run(self._evaluate_genome, 1)
        
    def _get_best_genome(self):
        """获取最佳基因组
        
        Returns:
            neat.DefaultGenome: 最佳基因组
        """
        if not hasattr(self.neat_pop, 'best_genome') or self.neat_pop.best_genome is None:
            # 如果没有最佳基因组，返回种群中适应度最高的基因组
            return max(self.neat_pop.population.values(), 
                      key=lambda x: x.fitness if x.fitness is not None else float('-inf'))
        return self.neat_pop.best_genome
        
    def act(self, state):
        """选择动作
        
        Args:
            state (np.ndarray): 当前状态
            
        Returns:
            int: 选择的动作
        """
        # 使用sigmoid函数动态调整p值
        p = 1 / (1 + np.exp(-self.beta * (self.episode_count - self.neat_eval_period)))
        
        if np.random.random() < p:
            # 使用Q-learning
            return self.q_agent.act(state)
        else:
            # 使用NEAT
            best_genome = self._get_best_genome()
            net = neat.nn.FeedForwardNetwork.create(best_genome, self.neat_pop.config)
            action_values = net.activate(state)
            return np.argmax(action_values)
            
    def observe(self, state, action, reward, next_state, done):
        """观察环境转换
        
        Args:
            state (np.ndarray): 当前状态
            action (int): 执行的动作
            reward (float): 获得的奖励
            next_state (np.ndarray): 下一个状态
            done (bool): 是否结束
        """
        # 存储到经验回放缓冲区
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # 更新episode计数
        if done:
            self.episode_count += 1
            
    def learn(self, done):
        """从经验中学习
        
        Args:
            done (bool): 是否结束
        """
        # Q-learning学习
        if len(self.replay_buffer) >= self.q_agent.batch_size:
            batch = self.replay_buffer.sample(self.q_agent.batch_size)
            self.q_agent.learn(batch)
            
        # NEAT学习
        if done and self.episode_count % self.neat_eval_period == 0:
            # 评估当前种群
            for genome in self.neat_pop.population.values():
                genome.fitness = self._evaluate_genome(genome)
                
            # 进化一代
            self.neat_pop.run(self._evaluate_genome, 1)
            
    def _evaluate_genome(self, genome):
        """评估基因组
        
        Args:
            genome (neat.DefaultGenome): 要评估的基因组
            
        Returns:
            float: 适应度值
        """
        net = neat.nn.FeedForwardNetwork.create(genome, self.neat_pop.config)
        total_reward = 0.0
        
        # 使用最近的几个episode进行评估
        if len(self.replay_buffer) > 0:
            batch = self.replay_buffer.sample(min(10, len(self.replay_buffer)))
            for state, _, _, _, _ in batch:
                action_values = net.activate(state)
                action = np.argmax(action_values)
                q_value = self.q_agent.estimate_value(state, action)
                total_reward += q_value
                
        return total_reward
        
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
                best_genome = self._get_best_genome()
                net = neat.nn.FeedForwardNetwork.create(best_genome, self.neat_pop.config)
                action_values = net.activate(state)
                action = np.argmax(action_values)
                next_state, reward, done, truncated, _ = env.step(action)
                episode_return += reward
                state = next_state
                
            neat_returns.append(episode_return)
            
        return np.mean(hybrid_returns), np.mean(neat_returns) 