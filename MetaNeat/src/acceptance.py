"""
接受管理器模块

实现Move-Acceptance Hyper-Heuristic (MAHH)，负责在评估阶段决定是否接受劣后个体。
"""

import random
import math
import numpy as np
from collections import deque

class AcceptanceManager:
    """
    实现Move-Acceptance Hyper-Heuristic (MAHH)
    
    通过动态调整接受概率p_t，决定是否接受劣后个体，
    帮助算法跳出局部最优，增强全局搜索能力。
    """
    
    def __init__(self, pop_size, window=20, kappa=0.7, min_p_factor=None, max_p_factor=None):
        """
        初始化接受管理器
        
        Args:
            pop_size (int): 种群大小
            window (int, optional): 计算停滞指标的窗口大小. Defaults to 20.
            kappa (float, optional): 调节系数. Defaults to 0.7.
            min_p_factor (float, optional): 最小概率因子. None则使用log(log N)/N.
            max_p_factor (float, optional): 最大概率因子. None则使用sqrt(N*log N)/N.
        """
        self.pop_size = pop_size
        self.window = window
        self.kappa = kappa
        
        # 确定概率上下界
        if min_p_factor is None:
            self.min_p = math.log(math.log(max(pop_size, 3))) / pop_size
        else:
            self.min_p = min_p_factor / pop_size
            
        if max_p_factor is None:
            self.max_p = math.sqrt(pop_size * math.log(pop_size)) / pop_size
        else:
            self.max_p = max_p_factor / pop_size
            
        # 初始化当前概率为最小值
        self.p_t = self.min_p
        
        # 记录每代的最佳适应度改进情况
        self.improvements = deque(maxlen=window)
        self.improvements.extend([0] * window)  # 初始化为无改进
        
        # 记录上一代的最佳适应度
        self.last_best_fitness = None
        
    def update_stagnation_metric(self, current_best_fitness):
        """
        更新停滞指标
        
        Args:
            current_best_fitness (float): 当前代的最佳适应度
        """
        # 检查是否有改进
        if self.last_best_fitness is not None:
            improvement = max(0, current_best_fitness - self.last_best_fitness)
            self.improvements.append(1 if improvement > 0 else 0)
        
        self.last_best_fitness = current_best_fitness
        
        # 计算停滞指标：无改进的代数比例
        stagnation_metric = 1.0 - (sum(self.improvements) / self.window)
        
        # 更新接受概率
        new_p_t = self.min_p + self.kappa * stagnation_metric * (self.max_p - self.min_p)
        
        # 限制变化速度，避免震荡
        if self.p_t > 0:
            max_change = 0.3 * self.p_t
            if new_p_t > self.p_t + max_change:
                new_p_t = self.p_t + max_change
            elif new_p_t < self.p_t - max_change:
                new_p_t = self.p_t - max_change
        
        # 确保概率在合理范围内
        self.p_t = max(self.min_p, min(self.max_p, new_p_t))
        
    def should_accept(self, worse_fitness, better_fitness):
        """
        决定是否接受劣后个体
        
        Args:
            worse_fitness (float): 劣后个体的适应度
            better_fitness (float): 更好个体的适应度
            
        Returns:
            bool: 是否接受劣后个体
        """
        # 如果劣后个体实际更好或相等，直接接受
        if worse_fitness >= better_fitness:
            return True
        
        # 否则根据当前概率决定是否接受
        return random.random() < self.p_t
    
    def get_current_probability(self):
        """
        获取当前接受概率
        
        Returns:
            float: 当前接受概率p_t
        """
        return self.p_t
    
    def get_stats(self):
        """
        获取接受管理器的统计信息
        
        Returns:
            dict: 包含当前概率和停滞指标的字典
        """
        stagnation_metric = 1.0 - (sum(self.improvements) / self.window)
        return {
            "acceptance_probability": self.p_t,
            "stagnation_metric": stagnation_metric,
            "min_probability": self.min_p,
            "max_probability": self.max_p
        }
