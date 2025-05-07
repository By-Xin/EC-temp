"""
Q-learning代理实现
包含神经网络和训练逻辑
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Tuple, Optional

class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_dims: List[int]):
        """初始化Q网络
        
        Args:
            state_size (int): 状态空间维度
            action_size (int): 动作空间维度
            hidden_dims (List[int]): 隐藏层维度列表
        """
        super(QNetwork, self).__init__()
        
        # 构建网络层
        layers = []
        prev_dim = state_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            state (torch.Tensor): 输入状态
            
        Returns:
            torch.Tensor: Q值
        """
        return self.network(state)

class QAgent:
    def __init__(self, state_size: int, action_size: int, config):
        """初始化Q-learning代理
        
        Args:
            state_size (int): 状态空间维度
            action_size (int): 动作空间维度
            config: 配置对象
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # 初始化Q网络
        self.q_network = QNetwork(state_size, action_size, config.Q_HIDDEN_DIMS)
        self.target_network = QNetwork(state_size, action_size, config.Q_HIDDEN_DIMS)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LR_Q)
        
        # 经验回放
        self.memory = deque(maxlen=config.REPLAY_CAPACITY)
        
        # 训练参数
        self.gamma = config.GAMMA
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = config.BATCH_SIZE
        
        # 记录回报
        self.returns = deque(maxlen=100)
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作
        
        Args:
            state (np.ndarray): 当前状态
            training (bool): 是否处于训练模式
            
        Returns:
            int: 选择的动作
        """
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
            
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax().item()
            
    def store(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool) -> None:
        """存储经验
        
        Args:
            state (np.ndarray): 当前状态
            action (int): 执行的动作
            reward (float): 获得的奖励
            next_state (np.ndarray): 下一个状态
            done (bool): 是否结束
        """
        self.memory.append((state, action, reward, next_state, done))
        
    def learn_step(self, sample_fn) -> None:
        """执行一步学习
        
        Args:
            sample_fn: 采样函数，返回(states, actions, rewards, next_states, dones)
        """
        if len(self.memory) < self.batch_size:
            return
            
        # 采样经验
        states, actions, rewards, next_states, dones = sample_fn(self.batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self) -> None:
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def estimate_value(self, state: np.ndarray, action: int) -> float:
        """估计状态-动作值
        
        Args:
            state (np.ndarray): 状态
            action (int): 动作
            
        Returns:
            float: 估计的Q值
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values[0][action].item()
            
    def save(self, path: str) -> None:
        """保存模型
        
        Args:
            path (str): 保存路径
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load(self, path: str) -> None:
        """加载模型
        
        Args:
            path (str): 加载路径
        """
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon'] 