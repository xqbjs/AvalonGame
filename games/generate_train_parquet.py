# -*- coding: utf-8 -*-
"""Generate training task Parquet file for games.

This script generates Parquet records for training tasks by reading game-specific
train_config.yaml files. It supports multiple games and can generate multiple task
copies from a single configuration.

Usage:
    python games/generate_train_parquet.py \
        --game avalon \
        --config games/games/avalon/configs/train_config.yaml \
        --output games/avalon/train_tasks.parquet \
        --num_tasks 10
"""
import argparse
import uuid
from pathlib import Path
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from games.utils import load_config


def generate_train_tasks_parquet(
    config_path: str,
    output_path: str,
    num_tasks: int = 10,
    env_type: str = None,
    task_id_prefix: str = None
):
    """Generate training task Parquet file from configuration.
    
    Args:
        config_path: Path to train_config.yaml
        output_path: Path to output Parquet file
        num_tasks: Number of tasks to generate (each task uses the same config)
        env_type: Environment type for tasks (auto-detected from config if not provided)
        task_id_prefix: Prefix for task IDs (auto-generated from game name if not provided)
    """
    # Load config (supports inheritance)
    base_config = load_config(config_path)
    
    # Auto-detect env_type from config if not provided
    if env_type is None:
        game_name = base_config.get('game', {}).get('name', 'unknown')
        env_type = game_name.lower()
    
    # Auto-generate task_id_prefix if not provided
    if task_id_prefix is None:
        task_id_prefix = f"{env_type}_train"
    
    # Extract game configuration for metadata
    # Structure should match what's expected by the training pipeline
    game_config = {
        'game': base_config.get('game', {}),
    }
    
    # Extract default_role configuration (used as default for all roles)
    if 'default_role' in base_config:
        game_config['default_role'] = base_config.get('default_role', {})
    
    # Extract roles with trainable flag (Avalon-style configuration)
    if 'roles' in base_config:
        roles_config = {}
        for role_name, role_cfg in base_config.get('roles', {}).items():
            if isinstance(role_cfg, dict) and role_cfg.get('trainable', False):
                roles_config[role_name] = role_cfg
        if roles_config:
            game_config['roles'] = roles_config
    
    # Generate records directly
    records = []
    for i in range(num_tasks):
        task_id = f"{task_id_prefix}_{i:04d}"
        record = {
            "data_source": env_type,
            "prompt": [{"content": task_id, "role": "user"}],
            "reward_model": {"ground_truth": [], "style": "rule"},
            "uuid": str(uuid.uuid4()),
            "extras": {
                "task_id": task_id,
                "open_query": False,
                "new_query": None,
                "evaluator": "env",
                "ground_truth": None,
                "metadata": {"config": game_config},
            },
        }
        records.append(record)
    
    # Save to Parquet
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_file, index=False)
    
    print(f"Generated {num_tasks} tasks -> {output_file.absolute()}")
    print(f"Config used: {config_path}")
    print(f"Environment type: {env_type}")
    print(f"Task ID prefix: {task_id_prefix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training task Parquet file for games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 tasks for Avalon
  python games/generate_train_parquet.py \\
      --game avalon \\
      --config games/games/avalon/configs/train_config.yaml \\
      --output games/avalon/train_tasks.parquet \\
      --num_tasks 10

  # Generate 20 tasks for Diplomacy
  python games/generate_train_parquet.py \\
      --game diplomacy \\
      --config games/games/diplomacy/configs/train_config.yaml \\
      --output games/diplomacy/train_tasks.parquet \\
      --num_tasks 20
        """
    )
    parser.add_argument(
        "--game",
        type=str,
        required=True,
        choices=["avalon", "diplomacy"],
        help="Game name (avalon or diplomacy)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to train_config.yaml (default: games/games/{game}/configs/train_config.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output Parquet file (default: games/{game}/train_tasks.parquet)"
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=10,
        help="Number of tasks to generate (default: 10)"
    )
    parser.add_argument(
        "--env_type",
        type=str,
        default=None,
        help="Environment type (auto-detected from config if not provided)"
    )
    parser.add_argument(
        "--task_id_prefix",
        type=str,
        default=None,
        help="Prefix for task IDs (default: {game}_train)"
    )
    
    args = parser.parse_args()
    
    # Set default config path if not provided
    if args.config is None:
        args.config = f"games/games/{args.game}/configs/train_config.yaml"
    
    # Set default output path if not provided
    if args.output is None:
        args.output = f"games/{args.game}/train_tasks.parquet"
    
    # Set default task_id_prefix if not provided
    if args.task_id_prefix is None:
        args.task_id_prefix = f"{args.game}_train"
    
    generate_train_tasks_parquet(
        config_path=args.config,
        output_path=args.output,
        num_tasks=args.num_tasks,
        env_type=args.env_type,
        task_id_prefix=args.task_id_prefix
    )

