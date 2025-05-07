"""
全局参数配置模块
包含了Meta-NEAT和Vanilla NEAT的所有配置参数
"""
import numpy as np
import os

# ======================
# 全局参数配置
# ======================
GAME = 'CartPole-v1'           # 游戏环境名称
CONFIG_PATH = "./config/neat_config.txt"  # NEAT配置文件路径
EP_STEP = 300                  # 每个episode最大步数
NUM_GENERATIONS = 50           # 进化代数 (可根据需要调整)
BASE_LOCAL_TRIALS = 5          # 内循环（局部适应）尝试次数的基准值 (Meta-NEAT)
MAX_LOCAL_TRIALS = 15          # 内循环尝试次数的上限
NUM_RUNS = 30                  # 每种方法重复实验次数，以获得更可靠的统计显著性
CONFIDENCE = 0.95              # 置信区间的置信度

USE_SOFTMAX = True             # 是否使用Softmax策略选动作
TAU = 1.0                      # Softmax温度参数
C_UCB = 0.5                    # UCB常数 (本例中未使用)

SEED_BASE = 42                 # 环境随机种子基数
VERBOSE = False                # 是否打印详细日志，设为False以减少输出

# ======================
# MAHH 相关配置
# ======================
USE_MAHH = True                # 是否启用Move-Acceptance Hyper-Heuristic
MAHH_KAPPA = 0.7               # 调节系数κ
MAHH_WINDOW = 20               # 过去k代的窗口大小
MIN_P_FACTOR = None            # 最小概率因子，None则使用默认公式
MAX_P_FACTOR = None            # 最大概率因子，None则使用默认公式

# ======================
# 奖励塑形相关配置
# ======================
USE_REWARD_SHAPING = True      # 是否启用奖励塑形
REWARD_ALPHA = 0.02            # 成本惩罚系数
REWARD_BETA = 0.3              # 偏移调整系数
REWARD_TARGET_MEAN = 0.2       # 目标奖励均值
REWARD_WINDOW_SIZE = 1000      # 滑动均值窗口大小
REWARD_MAX_ESTIMATE = 300.0    # 估计的环境最大回报
# 种群大小配置
POPULATION_SIZE = 50  # 根据您的需求设置，通常与NEAT配置文件中的pop_size相同

# 局部搜索统计信息
collect_statistics = True
local_search_stats = {
    'trials': [],
    'improvements': [],
    'improvement_ratios': []
}

# 新增统计信息
mahh_stats = {
    'acceptance_probabilities': [],
    'stagnation_metrics': []
}

reward_shaping_stats = {
    'raw_rewards': [],
    'shaped_rewards': [],
    'bias_history': []
}

# 在src/config.py中设置
USE_MAHH = True  # 启用/禁用MAHH
USE_REWARD_SHAPING = True  # 启用/禁用奖励塑形

# MAHH相关参数
MAHH_KAPPA = 0.7  # 调节系数κ
MAHH_WINDOW = 20  # 停滞指标窗口大小

# 奖励塑形相关参数
REWARD_ALPHA = 0.02  # 成本惩罚系数
REWARD_BETA = 0.3  # 偏移调整系数
REWARD_TARGET_MEAN = 0.2  # 目标奖励均值:wq

# 混合代理配置
P_INIT = 0.5                # 初始Q-learning使用概率
BETA_SIGMOID = 5.0          # sigmoid函数参数
NEAT_EVAL_PERIOD = 5        # NEAT评估周期
REPLAY_CAPACITY = 100_000   # 经验回放缓冲区大小
BATCH_SIZE = 128           # Q-learning批次大小
Q_HIDDEN_DIMS = [128, 128] # Q网络隐藏层维度
LR_Q = 1e-3                # Q-learning学习率
GAMMA = 0.99               # 折扣因子
ALPHA_MIXED_REWARD = 0.1   # 混合奖励系数

# 日志配置
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 模型保存配置
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def collect_statistics():
    """收集统计信息"""
    return {
        "num_runs": NUM_RUNS,
        "num_generations": NUM_GENERATIONS,
        "use_mahh": USE_MAHH,
        "use_reward_shaping": USE_REWARD_SHAPING,
        "p_init": P_INIT,
        "beta_sigmoid": BETA_SIGMOID,
        "neat_eval_period": NEAT_EVAL_PERIOD,
        "population_size": POPULATION_SIZE,
        "replay_capacity": REPLAY_CAPACITY,
        "batch_size": BATCH_SIZE,
        "q_hidden_dims": Q_HIDDEN_DIMS,
        "lr_q": LR_Q,
        "gamma": GAMMA,
        "alpha_mixed_reward": ALPHA_MIXED_REWARD
    }

def get_config():
    """获取配置参数字典
    
    Returns:
        dict: 配置参数字典
    """
    return {
        'P_INIT': P_INIT,
        'BETA_SIGMOID': BETA_SIGMOID,
        'NEAT_EVAL_PERIOD': NEAT_EVAL_PERIOD,
        'REPLAY_CAPACITY': REPLAY_CAPACITY,
        'BATCH_SIZE': BATCH_SIZE,
        'Q_HIDDEN_DIMS': Q_HIDDEN_DIMS,
        'LR_Q': LR_Q,
        'GAMMA': GAMMA,
        'ALPHA_MIXED_REWARD': ALPHA_MIXED_REWARD,
        'LOG_DIR': LOG_DIR,
        'MODEL_DIR': MODEL_DIR
    }

