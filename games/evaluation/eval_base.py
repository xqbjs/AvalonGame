# -*- coding: utf-8 -*-
"""Base evaluation framework for all games."""
import copy
import statistics
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_task_configs(
    base_config: Dict[str, Any], 
    num_games: int,
    experiment_name: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """Build list of task configurations for parallel execution.
    
    Args:
        base_config: Base configuration dictionary
        num_games: Number of games to run
        experiment_name: Optional experiment name for organizing logs
        **kwargs: Additional configuration overrides
        
    Returns:
        List of configuration dictionaries, one per game
    """
    configs = []
    for game_id in range(num_games):
        config = copy.deepcopy(base_config)
        
        # Add experiment_name if provided
        if experiment_name:
            config['experiment_name'] = experiment_name
        
        # Apply any additional overrides from kwargs
        for key, value in kwargs.items():
            if value is not None:
                if '.' in key:
                    # Handle nested keys like 'formatter.max_model_len'
                    keys = key.split('.')
                    current = config
                    for k in keys[:-1]:
                        if k not in current:
                            current[k] = {}
                        current = current[k]
                    current[keys[-1]] = value
                else:
                    # Handle flat keys
                    if key not in config:
                        config[key] = {}
                    if isinstance(config[key], dict) and isinstance(value, dict):
                        config[key].update(value)
                    else:
                        config[key] = value
        
        configs.append(config)
    
    return configs


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate game results, automatically handling different field types."""
    
    def compute_stats(values: List[float]) -> Dict[str, float]:
        return {"mean": round(statistics.mean(values), 2), "max": max(values), "min": min(values)}
    
    def aggregate_fields(data_list: List[Dict], group_by: str = None) -> Dict:
        if not data_list:
            return {}
        
        if group_by:
            grouped = defaultdict(lambda: defaultdict(list))
            for item in data_list:
                if isinstance(item, dict) and group_by in item:
                    for k, v in item.items():
                        if k != group_by and isinstance(v, (int, float)):
                            grouped[item[group_by]][k].append(v)
            return {g: {m: compute_stats(v) for m, v in sorted(metrics.items())} 
                    for g, metrics in sorted(grouped.items())}
        
        all_keys = {k for d in data_list for k in d.keys()}
        return {k: compute_stats([d[k] for d in data_list if k in d and isinstance(d[k], (int, float))])
                for k in sorted(all_keys) if any(k in d and isinstance(d[k], (int, float)) for d in data_list)}
    
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return {"error": "All games failed"}
    
    aggregated = {}
    game_results = [r["game_result"] for r in valid_results if "game_result" in r and isinstance(r["game_result"], dict)]
    if game_results:
        aggregated["game_result"] = aggregate_fields(game_results)
    
    roles_list = [role for r in valid_results for role in r.get("roles", []) if isinstance(role, dict)]
    if roles_list:
        aggregated["roles"] = aggregate_fields(roles_list, group_by="role_name")
    
    return aggregated


def run_evaluation(
    game_name: str,
    config_dict: Dict[str, Any],
    num_games: int,
    max_workers: int = 1,
    experiment_name: Optional[str] = None,
    run_single_game_fn: Optional[Callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run evaluation for a game.
    
    Args:
        game_name: Name of the game (e.g., 'avalon', 'diplomacy')
        config_dict: Base configuration dictionary
        num_games: Number of games to run
        max_workers: Maximum number of parallel workers
        experiment_name: Optional experiment name for organizing logs
        run_single_game_fn: Function to run a single game. 
            Signature: (config_dict: Dict[str, Any], game_id: int) -> Dict[str, Any]
        **kwargs: Additional configuration overrides
        
    Returns:
        Aggregated results dictionary
    """
    if run_single_game_fn is None:
        raise ValueError(f"No run_single_game function provided for game: {game_name}")
    
    # Step 1: Build task list (configurations)
    print(f"[{game_name}] Building {num_games} game configurations...")
    task_configs = build_task_configs(
        config_dict,
        num_games,
        experiment_name=experiment_name,
        **kwargs
    )
    
    # Step 2: Execute games in parallel
    print(f"[{game_name}] Running {num_games} games (max_workers={max_workers})...")
    results = []
    
    if num_games == 1:
        # Single game execution
        result = run_single_game_fn(task_configs[0], 0)
        results = [result]
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_game_fn, task_configs[game_id], game_id): game_id
                for game_id in range(num_games)
            }
            
            for future in as_completed(futures):
                game_id = futures[future]
                result = future.result()
                results.append(result)
                if result is not None:
                    print(f"[{game_name}] Game {game_id} completed")
    
    # Step 3: Aggregate results
    aggregated = aggregate_results(results)
    
    return aggregated

