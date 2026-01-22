#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run arena evaluation and generate leaderboard."""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Callable
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from games.utils import load_config
from games.evaluation.eval_base import build_task_configs
from games.evaluation.leaderboard.arena_workflow import create_arena_workflow
from games.evaluation.leaderboard.leaderboard_db import LeaderboardDB
from games.evaluation.leaderboard.leaderboard import generate_leaderboard_from_db
from games.evaluation.leaderboard.rate_limiter import set_global_rate_limiter, apply_rate_limiting_to_openai_model
from concurrent.futures import ThreadPoolExecutor, as_completed


def create_arena_evaluator(game_name: str, leaderboard_db: LeaderboardDB):
    """Create evaluator function for arena (reuses pattern from run_eval.py).
    
    Args:
        game_name: Name of the game (e.g., 'avalon', 'diplomacy')
        leaderboard_db: Leaderboard database to get game counts for fairness
    
    Returns:
        Function to run a single game
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def run_single_game(config_dict: Dict[str, Any], game_id: int) -> Dict[str, Any]:
        """Run a single arena game."""
        try:
            config_dict = config_dict.copy()
            config_dict['game_id'] = game_id
            config_dict['evaluation_timestamp'] = timestamp
            
            # Use game_id as seed offset for reproducibility
            arena_config = config_dict.get('arena', {})
            base_seed = arena_config.get('seed')
            if base_seed is not None:
                arena_config['seed'] = base_seed + game_id
                config_dict['arena'] = arena_config
            
            # Pass game counts for fair model assignment
            # Update counts before each game to ensure fairness
            # Pass arena models to ensure all are included (new models get count 0)
            arena_models = config_dict.get('arena', {}).get('models', [])
            config_dict['_model_game_counts'] = leaderboard_db.get_model_game_counts(arena_models)
            
            workflow = create_arena_workflow(game_name, config_dict)
            return workflow.execute()
        except Exception as e:
            print(f"[arena] Game {game_id} failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return None
    
    return run_single_game


def run_arena_with_db_update(
    config_dict: Dict[str, Any],
    num_games: int,
    max_workers: int,
    leaderboard_db: LeaderboardDB,
    run_single_game_fn: Callable,
    update_counts_interval: int = 10,
) -> list:
    """Run arena games and update leaderboard incrementally.
    
    Args:
        config_dict: Configuration dictionary
        num_games: Number of games to run
        max_workers: Maximum number of parallel workers
        leaderboard_db: Leaderboard database instance
        run_single_game_fn: Function to run a single game
    
    Returns:
        List of game results
    """
    # Build task configs
    task_configs = build_task_configs(
        config_dict,
        num_games,
        experiment_name=config_dict.get('experiment_name', 'arena_leaderboard'),
    )
    
    # Run games and collect results
    results = []
    if num_games == 1:
        result = run_single_game_fn(task_configs[0], 0)
        if result is not None:
            results.append(result)
            # Update leaderboard incrementally
            leaderboard_db.update_from_game_results([result])
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_single_game_fn, task_configs[game_id], game_id): game_id
                for game_id in range(num_games)
            }
            
            completed_count = 0
            for future in as_completed(futures):
                game_id = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        # Update leaderboard incrementally after each game
                        leaderboard_db.update_from_game_results([result])
                        completed_count += 1
                        
                        # Periodically show balance stats
                        if completed_count % update_counts_interval == 0:
                            balance = leaderboard_db.get_game_count_balance()
                            print(f"[arena] Game {game_id} completed | "
                                  f"Balance: {balance['balance_ratio']:.1%} "
                                  f"(min={balance['min']}, max={balance['max']})")
                        else:
                            print(f"[arena] Game {game_id} completed")
                    else:
                        print(f"[arena] Game {game_id} failed")
                except Exception as e:
                    print(f"[arena] Game {game_id} error: {e}", file=sys.stderr)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run arena evaluation and generate leaderboard for any game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 200 Avalon arena games
  python games/evaluation/leaderboard/run_arena.py \\
      --game avalon \\
      --config games/games/avalon/configs/arena_config.yaml \\
      --num-games 200 \\
      --max-workers 10
  
  # Run Diplomacy arena games
  python games/evaluation/leaderboard/run_arena.py \\
      --game diplomacy \\
      --config games/games/diplomacy/configs/arena_config.yaml \\
      --num-games 100 \\
      --max-workers 10
        """
    )
    
    # Supported games (avalon is always available, diplomacy is lazy-loaded)
    SUPPORTED_GAMES = ["avalon", "diplomacy"]
    
    parser.add_argument(
        "--game",
        "-g",
        type=str,
        required=True,
        choices=SUPPORTED_GAMES,
        help=f"Game to evaluate. Choices: {', '.join(SUPPORTED_GAMES)}",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to arena config YAML file",
    )
    parser.add_argument(
        "--num-games",
        "-n",
        type=int,
        default=200,
        help="Number of games to run (default: 200)",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=10,
        help="Maximum number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for organizing logs",
    )
    parser.add_argument(
        "--leaderboard-db",
        type=str,
        default=None,
        help="Path to leaderboard database file (default: games/evaluation/leaderboard/leaderboard_{game_name}.json)",
    )
    parser.add_argument(
        "--api-call-interval",
        type=float,
        default=0.0,
        help="Minimum seconds between API calls to prevent rate limiting (default: 0.0, no limit). Recommended: 0.1-0.5 seconds for high concurrency.",
    )
    
    args = parser.parse_args()
    
    # Set default leaderboard path based on game name if not specified
    if args.leaderboard_db is None:
        args.leaderboard_db = f"games/evaluation/leaderboard/leaderboard_{args.game}.json"
    
    # Initialize rate limiter if specified
    if args.api_call_interval > 0.0:
        set_global_rate_limiter(args.api_call_interval)
        apply_rate_limiting_to_openai_model()
        print(f"[arena] Rate limiting enabled: {args.api_call_interval}s between API calls")
    
    # Resolve config file path (reuses logic from run_eval.py)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        if not config_path.exists():
            # Try relative to game directory
            game_config_path = Path(__file__).parent.parent.parent / "games" / args.game / "configs" / args.config
            if game_config_path.exists():
                config_path = game_config_path
        config_path = config_path.resolve()
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load config file
    try:
        config_dict = load_config(config_path)
        if not isinstance(config_dict, dict):
            print(f"Error: Config must be a dictionary, got {type(config_dict)}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate arena config
    if 'arena' not in config_dict:
        print("Error: Config must contain 'arena' key with 'models' list", file=sys.stderr)
        sys.exit(1)
    
    arena_config = config_dict.get('arena', {})
    if 'models' not in arena_config or not arena_config['models']:
        print("Error: arena.models must be a non-empty list", file=sys.stderr)
        sys.exit(1)
    
    # Initialize leaderboard database (automatically loads existing data if available)
    # Default path is set in argument parsing: games/evaluation/leaderboard/leaderboard_{game_name}.json
    leaderboard_db = LeaderboardDB(args.leaderboard_db)
    
    # Update Elo settings from config if provided
    if 'elo_initial' in arena_config:
        leaderboard_db.data['elo_initial'] = arena_config['elo_initial']
    if 'elo_k' in arena_config:
        leaderboard_db.data['elo_k'] = arena_config['elo_k']
    
    # Always load existing leaderboard data and add any new models
    existing_models = leaderboard_db.get_all_models()
    if existing_models:
        print(f"[arena] Loaded existing leaderboard with {len(existing_models)} models: {existing_models}")
    else:
        print(f"[arena] Creating new leaderboard")
    
    # Add all models to database (new models will be added, existing ones preserved)
    for model in arena_config['models']:
        leaderboard_db.add_model(model)
    
    # Set experiment name
    if args.experiment_name:
        config_dict['experiment_name'] = args.experiment_name
    else:
        config_dict['experiment_name'] = f"arena_leaderboard_{args.game}"
    
    # Create evaluator (pass leaderboard_db for fair assignment)
    run_single_game_fn = create_arena_evaluator(args.game, leaderboard_db)
    
    # Show current game count balance
    balance_stats = leaderboard_db.get_game_count_balance()
    if balance_stats['max'] > 0:
        print(f"[arena] Current game count balance: min={balance_stats['min']}, "
              f"max={balance_stats['max']}, mean={balance_stats['mean']:.1f}, "
              f"balance_ratio={balance_stats['balance_ratio']:.3f}")
        if balance_stats['balance_ratio'] < 0.8:
            print(f"[arena] Warning: Game counts are unbalanced. New games will prioritize "
                  f"models with fewer games to improve fairness.")
    
    # Run evaluation with leaderboard updates
    try:
        print(f"[arena] Starting {args.game} arena evaluation with models: {arena_config['models']}")
        print(f"[arena] Running {args.num_games} games with {args.max_workers} workers")
        
        results = run_arena_with_db_update(
            config_dict=config_dict,
            num_games=args.num_games,
            max_workers=args.max_workers,
            leaderboard_db=leaderboard_db,
            run_single_game_fn=run_single_game_fn,
        )
        
        if not results:
            print("Error: All games failed", file=sys.stderr)
            sys.exit(1)
        
        print(f"\n[arena] Completed {len(results)} games successfully")
        
        # Generate and display leaderboard
        leaderboard_data = leaderboard_db.get_leaderboard_data()
        leaderboard = generate_leaderboard_from_db(leaderboard_data, arena_config, args.game)
        print(leaderboard)
        
        print(f"[arena] Leaderboard saved to: {args.leaderboard_db}")
        
    except Exception as e:
        print(f"Error during arena evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

