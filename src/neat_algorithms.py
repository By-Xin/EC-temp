"""
NEAT算法实现模块
包含Meta-NEAT和Vanilla NEAT的评估函数和训练函数
"""
import neat
import random
import numpy as np
from tqdm.auto import tqdm

from src.config import EP_STEP, NUM_GENERATIONS, VERBOSE
from src.utils import log_print
from src.environment import make_env, evaluate_single_genome
from src.evaluation import local_adaptation

def eval_genomes_meta(genomes, config, gen=0, total_gens=NUM_GENERATIONS):
    """Meta-NEAT评估函数（外循环）
    
    对每个genome，在单一环境中先进行局部适应（内循环），再将局部适应后的奖励归一化后作为fitness。
    
    Args:
        genomes (list): (genome_id, genome)对的列表
        config (neat.Config): NEAT配置
        gen (int, optional): 当前代数. Defaults to 0.
        total_gens (int, optional): 总代数. Defaults to NUM_GENERATIONS.
    """
    env = make_env()
    gen_progress = gen / total_gens  # 计算当前进化进度 (0-1)
    
    log_print(f"== 评估第 {gen+1}/{total_gens} 代 (进度: {gen_progress:.2f}) ==")
    
    # 使用tqdm包装genomes迭代
    genomes_iterator = genomes
    if not VERBOSE:  # 仅在非详细模式下显示tqdm
        genomes_iterator = tqdm(genomes, desc=f"评估第 {gen+1}/{total_gens} 代", leave=False)
    
    for genome_id, genome in genomes_iterator:
        # 对当前genome进行局部适应，传入当前进化进度
        _, best_reward = local_adaptation(genome, config, env, gen_progress)
        # 将局部适应后的奖励归一化作为fitness
        genome.fitness = best_reward / float(EP_STEP)
    env.close()

def eval_genomes_vanilla(genomes, config):
    """Vanilla NEAT评估函数
    
    对每个genome，在单一环境中直接评估，不进行局部适应
    
    Args:
        genomes (list): (genome_id, genome)对的列表
        config (neat.Config): NEAT配置
    """
    env = make_env()
    
    # 使用tqdm包装genomes迭代
    genomes_iterator = genomes
    if not VERBOSE:  # 仅在非详细模式下显示tqdm
        genomes_iterator = tqdm(genomes, desc="评估Vanilla NEAT", leave=False)
    
    for genome_id, genome in genomes_iterator:
        reward = evaluate_single_genome(genome, config, env)
        genome.fitness = reward / float(EP_STEP)
    env.close()

def run_training(num_generations, eval_fn, run_seed, algorithm_name=""):
    """训练函数：根据传入评估函数运行NEAT进化过程，记录每一代的最佳适应度
    
    Args:
        num_generations (int): 进化代数
        eval_fn (function): 评估函数
        run_seed (int): 随机种子
        algorithm_name (str, optional): 算法名称，用于显示. Defaults to "".
    
    Returns:
        tuple: (最优基因组, 配置, 统计信息, 每代最佳适应度)
    """
    from src.config import CONFIG_PATH
    
    # 设置随机种子，确保每次运行的可重复性
    random.seed(run_seed)
    np.random.seed(run_seed)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)
    pop = neat.Population(config)
    
    # 添加静默模式的Reporter，减少输出量
    pop.add_reporter(neat.StdOutReporter(VERBOSE))
    
    # 添加统计报告器
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    # 创建一个定制的评估函数，传递额外的代数参数
    # 检查评估函数是否是meta版本，只有meta版本才需要额外参数
    if eval_fn == eval_genomes_meta:
        def eval_wrapper(genomes, config):
            current_gen = pop.generation
            return eval_fn(genomes, config, current_gen, num_generations)
        run_func = eval_wrapper
    else:
        # 对于vanilla版本，直接使用原始的eval_fn
        run_func = eval_fn
    
    # 创建进度条来跟踪代数进度
    pbar_desc = f"训练 {algorithm_name}" if algorithm_name else "训练进度"
    with tqdm(total=num_generations, desc=pbar_desc, leave=True) as pbar:
        class ProgressReporter(neat.reporting.BaseReporter):
            def end_generation(self, config, population, species_set):
                pbar.update(1)
        
        # 添加进度条报告器
        progress_reporter = ProgressReporter()
        pop.add_reporter(progress_reporter)
        
        # 运行进化过程
        winner = pop.run(run_func, num_generations)
    
    # 记录每代的最佳适应度
    generation_fitnesses = []
    for gen in range(len(stats.most_fit_genomes)):
        generation_fitnesses.append(stats.most_fit_genomes[gen].fitness * EP_STEP)  # 反归一化得到原始奖励
    
    return winner, config, stats, generation_fitnesses
