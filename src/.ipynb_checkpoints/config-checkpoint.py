"""
全局参数配置模块
包含了Meta-NEAT和Vanilla NEAT的所有配置参数
"""
import numpy as np

# ======================
# 全局参数配置
# ======================
GAME = 'CartPole-v1'           # 游戏环境名称
CONFIG_PATH = "./config/neat_config.txt"  # NEAT配置文件路径
EP_STEP = 300                  # 每个episode最大步数
NUM_GENERATIONS = 10           # 进化代数 (可根据需要调整)
BASE_LOCAL_TRIALS = 5          # 内循环（局部适应）尝试次数的基准值 (Meta-NEAT)
MAX_LOCAL_TRIALS = 15          # 内循环尝试次数的上限
NUM_RUNS = 15                  # 每种方法重复实验次数，以获得更可靠的统计显著性
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

