"""
NEAT算法实现模块
包含Meta-NEAT和Vanilla NEAT的评估函数和训练函数
"""
import neat
import random
import numpy as np
from tqdm.auto import tqdm

from src.config import (
    EP_STEP, NUM_GENERATIONS, VERBOSE, 
    USE_MAHH, MAHH_KAPPA, MAHH_WINDOW, MIN_P_FACTOR, MAX_P_FACTOR,
    USE_REWARD_SHAPING, REWARD_ALPHA, REWARD_BETA, REWARD_TARGET_MEAN,
    REWARD_WINDOW_SIZE, REWARD_MAX_ESTIMATE, mahh_stats, reward_shaping_stats
)
from src.utils import log_print
from src.environment import make_env, evaluate_single_genome
from src.evaluation import local_adaptation
from src.acceptance import AcceptanceManager
from src.reward_shaping import RewardShaper

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
    
    # 如果启用奖励塑形，初始化RewardShaper
    reward_shaper = None
    if USE_REWARD_SHAPING:
        reward_shaper = RewardShaper(
            r_max=REWARD_MAX_ESTIMATE,
            alpha=REWARD_ALPHA,
            beta=REWARD_BETA,
            target_mean=REWARD_TARGET_MEAN,
            window_size=REWARD_WINDOW_SIZE
        )
    
    log_print(f"== 评估第 {gen+1}/{total_gens} 代 (进度: {gen_progress:.2f}) ==")
    
    # 使用tqdm包装genomes迭代
    genomes_iterator = genomes
    if not VERBOSE:  # 仅在非详细模式下显示tqdm
        genomes_iterator = tqdm(genomes, desc=f"评估第 {gen+1}/{total_gens} 代", leave=False)
    
    for genome_id, genome in genomes_iterator:
        # 对当前genome进行局部适应，传入当前进化进度
        _, best_reward = local_adaptation(genome, config, env, gen_progress)
        
        # 如果启用奖励塑形，应用塑形
        if reward_shaper:
            # 更新最大奖励估计
            reward_shaper.update_r_max(best_reward)
            
            # 应用奖励塑形 (使用局部搜索次数作为评估成本)
            from src.config import local_search_stats
            eval_cost = local_search_stats['trials'][-1] if local_search_stats['trials'] else 1.0
            shaped_reward = reward_shaper.shape(best_reward, eval_cost)
            
            # 记录统计信息
            genome.fitness = shaped_reward  # 直接使用塑形后的值作为适应度
        else:
            # 将局部适应后的奖励归一化作为fitness
            genome.fitness = best_reward / float(EP_STEP)
    
    # 如果使用奖励塑形，收集统计信息
    if reward_shaper:
        reward_shaping_stats['raw_rewards'].extend(reward_shaper.stats["raw_rewards"])
        reward_shaping_stats['shaped_rewards'].extend(reward_shaper.stats["shaped_rewards"])
        reward_shaping_stats['bias_history'].extend(reward_shaper.stats["bias_history"])
    
    env.close()

def eval_genomes_vanilla(genomes, config):
    """Vanilla NEAT评估函数
    
    对每个genome，在单一环境中直接评估，不进行局部适应
    
    Args:
        genomes (list): (genome_id, genome)对的列表
        config (neat.Config): NEAT配置
    """
    env = make_env()
    
    # 如果启用奖励塑形，初始化RewardShaper
    reward_shaper = None
    if USE_REWARD_SHAPING:
        reward_shaper = RewardShaper(
            r_max=REWARD_MAX_ESTIMATE,
            alpha=REWARD_ALPHA,
            beta=REWARD_BETA,
            target_mean=REWARD_TARGET_MEAN,
            window_size=REWARD_WINDOW_SIZE
        )
    
    # 使用tqdm包装genomes迭代
    genomes_iterator = genomes
    if not VERBOSE:  # 仅在非详细模式下显示tqdm
        genomes_iterator = tqdm(genomes, desc="评估Vanilla NEAT", leave=False)
    
    for genome_id, genome in genomes_iterator:
        reward = evaluate_single_genome(genome, config, env)
        
        # 如果启用奖励塑形，应用塑形
        if reward_shaper:
            # 更新最大奖励估计
            reward_shaper.update_r_max(reward)
            
            # 应用奖励塑形 (Vanilla NEAT无局部搜索，成本为1.0)
            shaped_reward = reward_shaper.shape(reward, 1.0)
            
            # 使用塑形后的值作为适应度
            genome.fitness = shaped_reward
        else:
            # 将原始奖励归一化作为fitness
            genome.fitness = reward / float(EP_STEP)
    
    # 如果使用奖励塑形，收集统计信息
    if reward_shaper:
        reward_shaping_stats['raw_rewards'].extend(reward_shaper.stats["raw_rewards"])
        reward_shaping_stats['shaped_rewards'].extend(reward_shaper.stats["shaped_rewards"])
        reward_shaping_stats['bias_history'].extend(reward_shaper.stats["bias_history"])
    
    env.close()

class CustomReproduction(neat.DefaultReproduction):
    """
    定制的繁殖器，集成了MAHH接受策略
    """
    def __init__(self, config, reporters, stagnation):
        super().__init__(config, reporters, stagnation)
        
        # 创建接受管理器
        self.acceptance_manager = AcceptanceManager(
            pop_size=config.pop_size,
            window=MAHH_WINDOW,
            kappa=MAHH_KAPPA,
            min_p_factor=MIN_P_FACTOR,
            max_p_factor=MAX_P_FACTOR
        )
        
        # 当前代的最佳适应度
        self.current_best_fitness = 0.0
    
    def reproduce(self, config, species, pop_size, generation):
        """
        实现集成MAHH的繁殖过程
        
        重写DefaultReproduction的reproduce方法，集成MAHH接受决策。
        对于精英策略，保持不变；对于非精英，使用MAHH决定是否接受劣后个体。
        
        Args:
            config: 配置对象
            species: 物种集合
            pop_size: 种群大小
            generation: 当前代数
            
        Returns:
            dict: (genome_id, genome)对的字典
        """
        # 找出当前最佳适应度
        all_fitnesses = []
        for sid, s in species.species.items():
            if s.fitness is not None:
                all_fitnesses.append(s.fitness)
        
        if all_fitnesses:
            self.current_best_fitness = max(all_fitnesses)
            # 更新接受管理器的停滞指标
            self.acceptance_manager.update_stagnation_metric(self.current_best_fitness)
        
        # 记录统计信息
        p_t = self.acceptance_manager.get_current_probability()
        mahh_stats['acceptance_probabilities'].append(p_t)
        stats = self.acceptance_manager.get_stats()
        mahh_stats['stagnation_metrics'].append(stats["stagnation_metric"])
        
        if not USE_MAHH:
            # 如果未启用MAHH，调用原始实现
            return super().reproduce(config, species, pop_size, generation)
        
        # 以下是修改的繁殖过程
        new_population = {}
        
        # 首先为每个物种分配后代数量
        species.compute_spawn_amounts(pop_size, generation)
        
        # 为每个物种产生后代
        for sid, s in species.species.items():
            # 如果这个物种应该灭绝，跳过
            if s.spawn_amount <= 0:
                continue
                
            # 获取精英数量
            elitism = max(self.elitism, self.elitism_percentage * len(s.members))
            
            # 精英个体直接保留
            if elitism > 0:
                elites = self.elitism_function(s.members, elitism)
                for elite in elites:
                    new_population[elite.key] = elite
                    
                # 更新剩余需生成的后代数量    
                spawn = s.spawn_amount - len(elites)
            else:
                spawn = s.spawn_amount
            
            # 如果没有剩余后代要产生，继续下一个物种
            if spawn <= 0:
                continue
                
            # 获取这个物种的所有基因组
            old_members = list(s.members)
            
            # 生成新后代
            parents = []
            while spawn > 0:
                # 如果父代不足，再次使用所有成员
                if not parents:
                    parents = old_members.copy()
                
                # 随机选择父代
                parent = random.choice(parents)
                parents.remove(parent)
                
                # 创建子代 (突变)
                child = parent.duplicate(generation)
                child.mutate(config.genome_config)
                
                # 主要区别：使用MAHH决定是否接受子代
                # 即使子代适应度低于父代，也有可能被接受
                if hasattr(parent, 'fitness') and hasattr(child, 'fitness'):
                    parent_fitness = parent.fitness
                    child_fitness = child.fitness
                    
                    # 使用接受管理器决定是否接受
                    if self.acceptance_manager.should_accept(child_fitness, parent_fitness):
                        new_population[child.key] = child
                        spawn -= 1
                    else:
                        # 如果不接受，保留父代
                        if parent.key not in new_population:
                            new_population[parent.key] = parent
                            spawn -= 1
                else:
                    # 如果适应度未知，直接接受子代
                    new_population[child.key] = child
                    spawn -= 1
        
        # 如果新种群未达到目标大小，添加随机基因组填充
        if len(new_population) < pop_size:
            log_print(f"种群大小不足 ({len(new_population)}/{pop_size})，添加随机个体")
            # 找出适应度最高的个体
            max_fitness = 0.0
            best_genome = None
            for genome in new_population.values():
                if hasattr(genome, 'fitness') and genome.fitness is not None:
                    if genome.fitness > max_fitness:
                        max_fitness = genome.fitness
                        best_genome = genome
            
            # 基于最佳个体创建填充个体
            while len(new_population) < pop_size:
                if best_genome is not None:
                    # 复制并强烈变异最佳个体
                    child = best_genome.duplicate(generation)
                    # 增加变异率进行强探索
                    child.mutate(config.genome_config)
                else:
                    # 如果没有最佳个体，创建全新个体
                    child = config.genome_type(generation)
                    child.configure_new(config.genome_config)
                
                new_population[child.key] = child
        
        return new_population

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
    
    # 创建NEAT配置
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        CONFIG_PATH)
    
    # 创建种群
    pop = neat.Population(config)
    
    # 如果启用MAHH，创建接受管理器
    acceptance_manager = None
    if USE_MAHH:
        from src.config import POPULATION_SIZE  # 导入全局定义的种群大小
        acceptance_manager = AcceptanceManager(
            pop_size=POPULATION_SIZE,
            window=MAHH_WINDOW,
            kappa=MAHH_KAPPA,
            min_p_factor=MIN_P_FACTOR,
            max_p_factor=MAX_P_FACTOR
        )
    
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
                # 如果启用了MAHH，在每代结束时更新接受概率
                if acceptance_manager is not None:
                    # 找出当前最佳适应度
                    current_best_fitness = 0.0
                    for g in population.values():
                        if g.fitness is not None and g.fitness > current_best_fitness:
                            current_best_fitness = g.fitness
                    
                    # 更新接受管理器
                    acceptance_manager.update_stagnation_metric(current_best_fitness)
                    
                    # 记录统计信息
                    p_t = acceptance_manager.get_current_probability()
                    mahh_stats['acceptance_probabilities'].append(p_t)
                    stats = acceptance_manager.get_stats()
                    mahh_stats['stagnation_metrics'].append(stats["stagnation_metric"])
                
                pbar.update(1)
                
        # 添加进度条报告器
        progress_reporter = ProgressReporter()
        pop.add_reporter(progress_reporter)
        
        # 运行进化过程 - 标准NEAT实现
        winner = pop.run(run_func, num_generations)
    
    # 记录每代的最佳适应度
    generation_fitnesses = []
    for gen in range(len(stats.most_fit_genomes)):
        # 如果使用了奖励塑形，需要特殊处理
        if USE_REWARD_SHAPING:
            # 直接使用已记录的适应度
            generation_fitnesses.append(stats.most_fit_genomes[gen].fitness)
        else:
            # 否则反归一化得到原始奖励
            generation_fitnesses.append(stats.most_fit_genomes[gen].fitness * EP_STEP)
    
    return winner, config, stats, generation_fitnesses
