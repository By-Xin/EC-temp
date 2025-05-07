"""
奖励塑形模块

提供对环境回报的缩放与自适应偏移，优化NEAT的适应度计算。
"""

import numpy as np
from collections import deque

class RewardShaper:
    """
    实现奖励缩放与自适应偏移
    
    通过奖励塑形解决强化学习中常见的三类问题：
    1. 奖励规模不一致 - 通过归一化解决
    2. 奖励偏移问题 - 通过自适应偏移解决
    3. 评估成本 - 通过成本惩罚解决
    """
    
    def __init__(self, r_max=300.0, alpha=0.02, beta=0.3, target_mean=0.2, window_size=1000):
        """
        初始化奖励塑形器
        
        Args:
            r_max (float, optional): 估计的环境最大回报. Defaults to 300.0.
            alpha (float, optional): 成本惩罚系数. Defaults to 0.02.
            beta (float, optional): 偏移调整系数. Defaults to 0.3.
            target_mean (float, optional): 目标奖励均值. Defaults to 0.2.
            window_size (int, optional): 滑动均值窗口大小. Defaults to 1000.
        """
        self.r_max = r_max
        self.alpha = alpha
        self.beta = beta
        self.target_mean = target_mean
        
        # 用于计算滑动平均的队列
        self.reward_history = deque(maxlen=window_size)
        
        # 初始化偏移量为0
        self.bias = 0.0
        
        # 记录统计信息
        self.stats = {
            "raw_rewards": [],
            "shaped_rewards": [],
            "bias_history": []
        }
        
    def shape(self, reward, eval_cost=0.0):
        """
        对原始奖励进行塑形
        
        Args:
            reward (float): 原始奖励值
            eval_cost (float, optional): 评估成本. Defaults to 0.0.
        
        Returns:
            float: 塑形后的奖励
        """
        # 保存原始奖励用于统计
        self.stats["raw_rewards"].append(reward)
        
        # 1. 奖励缩放 - 归一化到[0,1]范围
        normalized_reward = reward / self.r_max
        
        # 2. 评估成本惩罚 - 可以是时间、计算次数等
        # 使用对数缩放避免惩罚过重
        if eval_cost > 0:
            cost_penalty = self.alpha * np.log(1 + eval_cost)
        else:
            cost_penalty = 0.0
            
        # 3. 应用当前偏移量
        shaped_reward = normalized_reward - cost_penalty + self.bias
        
        # 更新奖励历史
        self.reward_history.append(shaped_reward)
        
        # 计算当前均值并更新偏移量
        if len(self.reward_history) > 0:
            current_mean = np.mean(self.reward_history)
            mean_error = self.target_mean - current_mean
            
            # 通过beta系数平滑更新偏移量
            self.bias += self.beta * mean_error
            
            # 限制偏移量范围，避免过度偏移
            self.bias = max(-0.5, min(0.5, self.bias))
        
        # 记录偏移量历史
        self.stats["bias_history"].append(self.bias)
        self.stats["shaped_rewards"].append(shaped_reward)
        
        return shaped_reward
    
    def reset(self):
        """重置奖励塑形器状态"""
        self.reward_history.clear()
        self.bias = 0.0
    
    def update_r_max(self, r_max):
        """
        更新最大奖励估计值
        
        Args:
            r_max (float): 新的最大奖励估计值
        """
        # 当观察到更高的回报时，更新r_max
        if r_max > self.r_max:
            self.r_max = r_max
            
    def get_stats(self):
        """
        获取奖励塑形器的统计信息
        
        Returns:
            dict: 包含原始奖励、塑形奖励和偏移量历史的字典
        """
        stats = {
            "raw_rewards_mean": np.mean(self.stats["raw_rewards"]) if self.stats["raw_rewards"] else 0,
            "shaped_rewards_mean": np.mean(self.stats["shaped_rewards"]) if self.stats["shaped_rewards"] else 0,
            "current_bias": self.bias,
            "r_max": self.r_max
        }
        return stats
