#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified evaluation script for all games.

This script provides a unified entry point for evaluating different games.
It handles game selection, config loading, and result aggregation.

Basic Usage:
    python games/evaluation/run_eval.py \
        --game avalon \
        --config games/games/avalon/configs/task_config.yaml \
        --num-games 3 \
        --max-workers 5 \
        --experiment-name "my_experiment"

Using Local VLLM Models:
    To use local models with VLLM, you need to start the VLLM server separately first:
    
    Terminal 1 (start VLLM server):
        python games/evaluation/start_vllm.py --model-path /path/to/model --port 8000 --model-name local_model
    
    Terminal 2 (run evaluation):
        python games/evaluation/run_eval.py --game avalon --config games/games/avalon/configs/task_config.yaml --num-games 10
    
    Make sure your config file has the correct URL and model_name:
        default_model:
          url: http://localhost:8000/v1
          model_name: local_model
"""
import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from games.utils import load_config
from games.evaluation.eval_base import run_evaluation


# Game registry: maps game names to factory functions (lazy-loaded)
GAME_REGISTRY: Dict[str, Callable[[], Callable]] = {}


def register_game(name: str):
    """Decorator to register a game evaluator factory."""
    def decorator(factory_func: Callable[[], Callable]):
        GAME_REGISTRY[name] = factory_func
        return factory_func
    return decorator


def _create_evaluator(workflow_class, game_name: str):
    """Create evaluator function for a game."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def run_single_game(config_dict: Dict[str, Any], game_id: int) -> Optional[Dict[str, Any]]:
        try:
            config_dict = config_dict.copy()
            config_dict['game_id'] = game_id
            config_dict['evaluation_timestamp'] = timestamp
            workflow = workflow_class(config_dict=config_dict)
            return workflow.execute()
        except Exception as e:
            print(f"[{game_name}] Game {game_id} failed: {e}", file=sys.stderr)
            traceback.print_exc()
            return None
    
    return run_single_game


@register_game("avalon")
def get_avalon_evaluator():
    """Get Avalon evaluator (lazy-loaded)."""
    from games.games.avalon.workflows.eval_workflow import EvalAvalonWorkflow
    return _create_evaluator(EvalAvalonWorkflow, "avalon")


@register_game("diplomacy")
def get_diplomacy_evaluator():
    """Get Diplomacy evaluator (lazy-loaded)."""
    from games.games.diplomacy.workflows.eval_workflow import EvalDiplomacyWorkflow
    return _create_evaluator(EvalDiplomacyWorkflow, "diplomacy")


def display_results(aggregated: Dict[str, Any], game_name: str = "Game", num_games: int = None):
    """Display aggregated statistics in a formatted table layout."""
    W = 90
    
    def calc_width(values):
        return max(len(str(v)) for v in values) + 1
    
    def print_table(title, headers, rows, alignments):
        columns = [[h] + [r[i] for r in rows] for i, h in enumerate(headers)]
        widths = [calc_width(col) for col in columns]
        
        print(f"\n┌─ {title} " + "─" * (W - len(title) - 5) + "┐")
        print(f"│{' ' * (W - 2)}│")
        print("│  " + " │ ".join(f'{h:<{w}}' for h, w in zip(headers, widths)) + "  │")
        print("│  " + "─┼─".join('─' * w for w in widths) + "  │")
        for row in rows:
            parts = [f'{v:<{w}}' if a == 'l' else f'{v:>{w}}' for v, w, a in zip(row, widths, alignments)]
            print("│  " + " │ ".join(parts) + "  │")
        print(f"└{'─' * (W - 2)}┘")
    
    def format_stats(stats):
        fmt = lambda v: str(v) if isinstance(v, int) else f"{v:.2f}"
        return (f"{stats['mean']:.2f}", fmt(stats['max']), fmt(stats['min']))
    
    # Extract data
    game_rows = [(k, *format_stats(v)) for k, v in aggregated.get("game_result", {}).items() 
                 if isinstance(v, dict) and "mean" in v]
    
    role_data = [(role, metric, stats) for role, metrics in aggregated.get("roles", {}).items()
                 for metric, stats in metrics.items() if isinstance(stats, dict) and "mean" in stats]
    role_rows = []
    prev = None
    for role, metric, stats in role_data:
        role_rows.append((role if role != prev else "", metric, *format_stats(stats)))
        prev = role
    
    if not (game_rows or role_rows):
        print("No data to display")
        return
    
    # Print header
    print("\n" + "═" * W)
    title = f"📊 {game_name.upper()} - RESULTS" + (f" (Total Games: {num_games})" if num_games else "")
    print(title.center(W))
    print("═" * W)
    
    # Print tables
    if game_rows:
        print_table("🏆 Game Results", ["Metric", "Mean", "Max", "Min"], game_rows, "lrrr")
    if role_rows:
        print_table("👥 Role Statistics", ["Role", "Metric", "Mean", "Max", "Min"], role_rows, "llrrr")
    
    print("\n" + "═" * W + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for all games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 Avalon games
  python games/evaluation/run_eval.py --game avalon --config games/games/avalon/configs/task_config.yaml --num-games 10
  
  # Run with custom experiment name
  python games/evaluation/run_eval.py --game avalon --config configs/task_config.yaml --num-games 5 --experiment-name "test_run"
        """
    )
    
    parser.add_argument(
        "--game",
        "-g",
        type=str,
        required=True,
        choices=list(GAME_REGISTRY.keys()),
        help=f"Game to evaluate. Choices: {', '.join(GAME_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to game config YAML file",
    )
    parser.add_argument(
        "--num-games",
        "-n",
        type=int,
        default=1,
        help="Number of games to run (default: 1)",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=10,
        help="Maximum number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for organizing logs",
    )
    
    # Game-specific arguments (will be passed as kwargs)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model length for formatter",
    )
    parser.add_argument(
        "--response-length",
        type=int,
        default=None,
        help="Response length for formatter",
    )
    
    args = parser.parse_args()
    
    # Resolve config file path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try relative to current directory first
        if not config_path.exists():
            # Try relative to game directory
            game_config_path = Path(__file__).parent.parent / "games" / args.game / args.config
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
    
    # Get game evaluator (lazy-loaded: only imports the selected game)
    if args.game not in GAME_REGISTRY:
        print(f"Error: Game '{args.game}' not found. Available: {', '.join(GAME_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)
    
    run_single_game_fn = GAME_REGISTRY[args.game]()
    
    # Prepare kwargs for additional config overrides
    kwargs = {}
    if args.max_model_len is not None:
        kwargs['formatter.max_model_len'] = args.max_model_len
    if args.response_length is not None:
        kwargs['formatter.response_length'] = args.response_length
    
    # Run evaluation
    try:
        aggregated = run_evaluation(
            game_name=args.game,
            config_dict=config_dict,
            num_games=args.num_games,
            max_workers=args.max_workers,
            experiment_name=args.experiment_name,
            run_single_game_fn=run_single_game_fn,
            **kwargs
        )
        display_results(aggregated, args.game, args.num_games)
        
        if aggregated.get("error") == "All games failed":
            sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

