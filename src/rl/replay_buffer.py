"""
经验回放缓冲区实现
用于存储和采样训练数据
"""
import numpy as np
from collections import deque
from typing import Tuple, List, Optional

class ReplayBuffer:
    def __init__(self, capacity: int):
        """初始化经验回放缓冲区
        
        Args:
            capacity (int): 缓冲区最大容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
    def add(self, state, action, reward, next_state, done) -> None:
        """添加一条经验到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        """随机采样一批经验
        
        Args:
            batch_size (int): 采样数量
            
        Returns:
            Tuple: (states, actions, rewards, next_states, dones)
        """
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def random_episode(self) -> List:
        """随机采样一个完整的episode
        
        Returns:
            List: episode中的经验列表
        """
        if len(self.buffer) == 0:
            return []
            
        # 随机选择一个起始点
        start_idx = np.random.randint(0, len(self.buffer))
        episode = []
        
        # 收集直到done或到达缓冲区末尾
        for i in range(start_idx, len(self.buffer)):
            state, action, reward, next_state, done = self.buffer[i]
            episode.append((state, action, reward, next_state, done))
            if done:
                break
                
        return episode
    
    def __len__(self) -> int:
        """返回当前缓冲区中的经验数量"""
        return len(self.buffer) 