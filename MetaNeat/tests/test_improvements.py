"""
测试改进组件模块
测试MAHH和奖励塑形组件的功能
"""
import unittest
import numpy as np
import random
import os
import sys

# 将项目根目录添加到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.acceptance import AcceptanceManager
from src.reward_shaping import RewardShaper

class TestAcceptanceManager(unittest.TestCase):
    """测试AcceptanceManager组件"""
    
    def setUp(self):
        """初始化测试环境"""
        self.pop_size = 50
        random.seed(42)  # 设置随机种子确保可重复性
        
    def test_initialization(self):
        """测试初始化功能"""
        # 创建管理器实例
        manager = AcceptanceManager(self.pop_size)
        
        # 检查默认属性
        self.assertEqual(manager.pop_size, self.pop_size)
        self.assertEqual(manager.window, 20)  # 默认窗口大小
        self.assertEqual(manager.kappa, 0.7)  # 默认kappa值
        
        # 检查计算的概率界限
        expected_min_p = np.log(np.log(self.pop_size)) / self.pop_size
        expected_max_p = np.sqrt(self.pop_size * np.log(self.pop_size)) / self.pop_size
        
        self.assertAlmostEqual(manager.min_p, expected_min_p, places=6)
        self.assertAlmostEqual(manager.max_p, expected_max_p, places=6)
        
        # 检查初始p_t是否等于min_p
        self.assertEqual(manager.p_t, manager.min_p)
    
    def test_should_accept_better_fitness(self):
        """测试更好的适应度总是被接受"""
        manager = AcceptanceManager(self.pop_size)
        
        # 子代适应度高于父代
        self.assertTrue(manager.should_accept(0.8, 0.7))
        
        # 子代适应度等于父代
        self.assertTrue(manager.should_accept(0.7, 0.7))
    
    def test_should_accept_worse_fitness(self):
        """测试更差的适应度根据概率接受"""
        manager = AcceptanceManager(self.pop_size)
        
        # 设置一个已知的概率值用于测试
        manager.p_t = 1.0  # 100%接受概率
        
        # 此时即使适应度更差也应该接受
        self.assertTrue(manager.should_accept(0.6, 0.7))
        
        # 设置为0概率
        manager.p_t = 0.0
        
        # 此时更差的适应度应该被拒绝
        self.assertFalse(manager.should_accept(0.6, 0.7))
    
    def test_update_stagnation_metric(self):
        """测试停滞指标更新功能"""
        manager = AcceptanceManager(self.pop_size)
        
        # 初始化时所有改进值都是0
        self.assertEqual(sum(manager.improvements), 0)
        
        # 更新一次，没有改进
        manager.update_stagnation_metric(0.5)
        manager.update_stagnation_metric(0.5)  # 没有改进
        
        # 停滞指标应该接近1.0
        stagnation_metric = 1.0 - (sum(manager.improvements) / manager.window)
        self.assertAlmostEqual(stagnation_metric, 1.0, places=2)
        
        # 连续更新几次，有改进
        manager.update_stagnation_metric(0.6)  # 有改进
        manager.update_stagnation_metric(0.7)  # 有改进
        manager.update_stagnation_metric(0.8)  # 有改进
        
        # 现在停滞指标应该更小
        stagnation_metric = 1.0 - (sum(manager.improvements) / manager.window)
        self.assertLess(stagnation_metric, 1.0)
        
        # 概率应该随着停滞指标的变化而变化
        self.assertNotEqual(manager.p_t, manager.min_p)
    
    def test_probability_bounds(self):
        """测试概率上下限约束"""
        manager = AcceptanceManager(self.pop_size)
        
        # 尝试手动将概率设置到超出界限的值
        manager.p_t = manager.max_p * 2
        manager.update_stagnation_metric(0.5)
        
        # 概率应该被限制在最大值
        self.assertEqual(manager.p_t, manager.max_p)
        
        # 尝试设置低于下限的值
        manager.p_t = manager.min_p / 2
        manager.update_stagnation_metric(0.5)
        
        # 概率应该被限制在最小值
        self.assertEqual(manager.p_t, manager.min_p)
    
    def test_get_stats(self):
        """测试获取统计信息功能"""
        manager = AcceptanceManager(self.pop_size)
        stats = manager.get_stats()
        
        # 检查返回的字典是否包含所有必要的键
        expected_keys = ['acceptance_probability', 'stagnation_metric', 'min_probability', 'max_probability']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # 检查数值是否正确
        self.assertEqual(stats['acceptance_probability'], manager.p_t)
        self.assertEqual(stats['min_probability'], manager.min_p)
        self.assertEqual(stats['max_probability'], manager.max_p)


class TestRewardShaper(unittest.TestCase):
    """测试RewardShaper组件"""
    
    def setUp(self):
        """初始化测试环境"""
        self.r_max = 300.0
        np.random.seed(42)
        
    def test_initialization(self):
        """测试初始化功能"""
        shaper = RewardShaper(r_max=self.r_max)
        
        # 检查默认属性
        self.assertEqual(shaper.r_max, self.r_max)
        self.assertEqual(shaper.alpha, 0.02)  # 默认alpha值
        self.assertEqual(shaper.beta, 0.3)    # 默认beta值
        self.assertEqual(shaper.target_mean, 0.2)  # 默认目标均值
        self.assertEqual(shaper.bias, 0.0)    # 初始偏移量
        
        # 检查统计信息初始化
        self.assertEqual(len(shaper.reward_history), 0)
        self.assertIn('raw_rewards', shaper.stats)
        self.assertIn('shaped_rewards', shaper.stats)
        self.assertIn('bias_history', shaper.stats)
    
    def test_reward_scaling(self):
        """测试奖励缩放功能"""
        shaper = RewardShaper(r_max=100.0, alpha=0.0, beta=0.0)
        
        # 不考虑偏移和惩罚，应该只进行归一化
        shaped_value = shaper.shape(50.0)
        self.assertEqual(shaped_value, 0.5)  # 50/100 = 0.5
        
        # 超过最大值的情况
        shaped_value = shaper.shape(150.0)
        self.assertEqual(shaped_value, 1.5)  # 150/100 = 1.5
    
    def test_cost_penalty(self):
        """测试成本惩罚功能"""
        shaper = RewardShaper(r_max=100.0, alpha=0.1, beta=0.0)
        
        # 计算预期惩罚
        eval_cost = 10.0
        expected_penalty = 0.1 * np.log(1 + eval_cost)
        
        # 考虑惩罚，不考虑偏移
        shaped_value = shaper.shape(50.0, eval_cost)
        self.assertAlmostEqual(shaped_value, 0.5 - expected_penalty, places=6)
    
    def test_bias_adjustment(self):
        """测试偏移调整功能"""
        shaper = RewardShaper(r_max=100.0, alpha=0.0, beta=0.5, target_mean=0.5)
        
        # 添加一系列低于目标均值的奖励
        for _ in range(10):
            shaper.shape(20.0)  # 归一化后为0.2，低于目标0.5
        
        # 偏移量应该是正的，以提高奖励
        self.assertGreater(shaper.bias, 0.0)
        
        # 重置并添加一系列高于目标均值的奖励
        shaper.reset()
        for _ in range(10):
            shaper.shape(80.0)  # 归一化后为0.8，高于目标0.5
        
        # 偏移量应该是负的，以降低奖励
        self.assertLess(shaper.bias, 0.0)
    
    def test_r_max_update(self):
        """测试最大奖励更新功能"""
        shaper = RewardShaper(r_max=100.0)
        
        # 使用低于r_max的值
        shaper.shape(80.0)
        self.assertEqual(shaper.r_max, 100.0)  # 不应改变
        
        # 使用高于r_max的值
        shaper.update_r_max(120.0)
        self.assertEqual(shaper.r_max, 120.0)  # 应该更新
        
        # 再次使用低于新r_max的值
        shaper.update_r_max(110.0)
        self.assertEqual(shaper.r_max, 120.0)  # 不应改变
    
    def test_get_stats(self):
        """测试获取统计信息功能"""
        shaper = RewardShaper(r_max=100.0)
        
        # 添加一些奖励
        shaper.shape(50.0)
        shaper.shape(60.0)
        
        stats = shaper.get_stats()
        
        # 检查返回的字典是否包含所有必要的键
        expected_keys = ['raw_rewards_mean', 'shaped_rewards_mean', 'current_bias', 'r_max']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # 检查数值是否正确
        self.assertEqual(stats['r_max'], shaper.r_max)
        self.assertEqual(stats['current_bias'], shaper.bias)
        self.assertAlmostEqual(stats['raw_rewards_mean'], 55.0, places=6)  # (50+60)/2


if __name__ == '__main__':
    unittest.main()
