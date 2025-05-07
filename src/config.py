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
NUM_RUNS = 30                  # 每种方法重复实验次数，以获得更可靠的统计显著性
CONFIDENCE = 0.95              # 置信区间的置信度

USE_SOFTMAX = True             # 是否使用Softmax策略选动作
TAU = 1.0                      # Softmax温度参数
C_UCB = 0.5                    # UCB常数 (本例中未使用)

SEED_BASE = 42                 # 环境随机种子基数
VERBOSE = False                # 是否打印详细日志，设为False以减少输出

# 局部搜索统计信息
collect_statistics = True
local_search_stats = {
    'trials': [],
    'improvements': [],
    'improvement_ratios': []
}
