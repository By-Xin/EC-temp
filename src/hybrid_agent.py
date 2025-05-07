"""
混合代理实现
结合Q-learning和NEAT的混合学习系统
"""
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
import logging
import neat

from src.rl.replay_buffer import ReplayBuffer
from src.rl.q_learning import QAgent

class HybridAgent:
    def __init__(self, config, neat_pop: neat.Population, q_agent: QAgent):
        """初始化混合代理
        
        Args:
            config: 配置对象
            neat_pop (neat.Population): NEAT种群
            q_agent (QAgent): Q-learning代理
        """
        self.config = config
        self.neat_pop = neat_pop
        self.q_agent = q_agent
        
        # 混合参数
        self.p = config['P_INIT']  # 初始Q-learning使用概率
        self.beta = config['BETA_SIGMOID']  # sigmoid函数参数
        self.neat_eval_period = config['NEAT_EVAL_PERIOD']  # NEAT评估周期
        
        # 经验回放
        self.replay = ReplayBuffer(config['REPLAY_CAPACITY'])
        
        # 训练状态
        self.episode_counter = 0
        self.returns = deque(maxlen=100)  # 记录最近100个episode的回报
        
        # 日志
        self.logger = logging.getLogger("HybridAgent")
        
    def act(self, state: np.ndarray) -> int:
        """选择动作
        
        Args:
            state (np.ndarray): 当前状态
            
        Returns:
            int: 选择的动作
        """
        if np.random.rand() < self.p:
            # 使用Q-learning
            return self.q_agent.act(state)
        else:
            # 使用NEAT
            best_genome = max(self.neat_pop.population.values(), key=lambda x: x.fitness if x.fitness is not None else float('-inf'))
            return self._get_action_from_genome(best_genome, state)
            
    def _get_action_from_genome(self, genome: neat.DefaultGenome, state: np.ndarray) -> int:
        """从基因组获取动作
        
        Args:
            genome (neat.DefaultGenome): NEAT基因组
            state (np.ndarray): 当前状态
            
        Returns:
            int: 选择的动作
        """
        net = neat.nn.FeedForwardNetwork.create(genome, self.neat_pop.config)
        output = net.activate(state)
        return np.argmax(output)
            
    def observe(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """观察环境反馈
        
        Args:
            state (np.ndarray): 当前状态
            action (int): 执行的动作
            reward (float): 获得的奖励
            next_state (np.ndarray): 下一个状态
            done (bool): 是否结束
        """
        # 存储经验
        self.replay.add(state, action, reward, next_state, done)
        self.q_agent.store(state, action, reward, next_state, done)
        
        # 更新回报记录
        if done:
            self.returns.append(reward)
            
    def learn(self, done: bool) -> None:
        """执行学习
        
        Args:
            done (bool): 当前episode是否结束
        """
        # Q-learning学习
        self.q_agent.learn_step(self.replay.sample)
        
        if done:
            self.episode_counter += 1
            
            # 定期评估NEAT种群
            if self.episode_counter % self.neat_eval_period == 0:
                self._evaluate_neat_population()
                
            # 更新混合概率
            self._update_p()
            
    def _evaluate_neat_population(self) -> None:
        """评估NEAT种群"""
        # 为每个基因组计算适应度
        for genome_id, genome in self.neat_pop.population.items():
            total_r = 0.0
            n_evals = 5  # 每个个体评估5次
            
            for _ in range(n_evals):
                # 随机采样一个episode
                traj = self.replay.random_episode()
                if not traj:
                    continue
                    
                # 使用个体执行episode
                state = traj[0][0]  # 初始状态
                episode_return = 0.0
                
                for state, action, reward, next_state, done in traj:
                    # 使用个体选择动作
                    indiv_action = self._get_action_from_genome(genome, state)
                    if indiv_action == action:  # 如果动作匹配
                        episode_return += reward
                    state = next_state
                    if done:
                        break
                        
                total_r += episode_return
                
            genome.fitness = total_r / n_evals
            
        # 进化一代
        self.neat_pop.run(lambda genomes, config: None, 1)
        
    def _update_p(self) -> None:
        """更新混合概率p"""
        if len(self.q_agent.returns) < 50:
            return
            
        # 计算最近50个episode的平均回报
        R_q = np.mean(list(self.q_agent.returns)[-50:])
        
        # 获取NEAT种群的最佳适应度
        best_genome = max(self.neat_pop.population.values(), key=lambda x: x.fitness if x.fitness is not None else float('-inf'))
        R_e = best_genome.fitness if best_genome.fitness is not None else 0.0
        
        # 使用sigmoid函数更新p
        delta = R_q - R_e
        new_p = 1.0 / (1.0 + np.exp(-self.beta * delta))
        
        # 平滑更新
        self.p = 0.9 * self.p + 0.1 * new_p
        
        # 限制在合理范围内
        self.p = np.clip(self.p, 0.05, 0.95)
        
        self.logger.info(f"Updated p: {self.p:.3f} (Q: {R_q:.2f}, NEAT: {R_e:.2f})")
        
    def save(self, path: str) -> None:
        """保存模型
        
        Args:
            path (str): 保存路径
        """
        # 保存Q-learning模型
        self.q_agent.save(f"{path}_q.pth")
        
        # 保存NEAT种群
        self.neat_pop.save(f"{path}_neat.pkl")
        
        # 保存混合参数
        np.save(f"{path}_hybrid.npy", {
            'p': self.p,
            'episode_counter': self.episode_counter,
            'returns': list(self.returns)
        })
        
    def load(self, path: str) -> None:
        """加载模型
        
        Args:
            path (str): 加载路径
        """
        # 加载Q-learning模型
        self.q_agent.load(f"{path}_q.pth")
        
        # 加载NEAT种群
        self.neat_pop.load(f"{path}_neat.pkl")
        
        # 加载混合参数
        params = np.load(f"{path}_hybrid.npy", allow_pickle=True).item()
        self.p = params['p']
        self.episode_counter = params['episode_counter']
        self.returns = deque(params['returns'], maxlen=100) 