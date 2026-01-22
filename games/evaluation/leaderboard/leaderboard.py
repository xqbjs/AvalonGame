# -*- coding: utf-8 -*-
"""Leaderboard calculation and reporting for arena."""
from typing import Dict, Any, Tuple


def calculate_elo(elo_a: float, elo_b: float, score_a: float, k: int = 32) -> Tuple[float, float]:
    """Calculate new Elo ratings after a game.
    
    Args:
        elo_a: Current Elo rating of player A
        elo_b: Current Elo rating of player B
        score_a: Score for player A (1.0 for win, 0.5 for draw, 0.0 for loss)
        k: K-factor for Elo calculation (default: 32)
    
    Returns:
        Tuple of (new_elo_a, new_elo_b)
    """
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    expected_b = 1 - expected_a
    
    new_elo_a = elo_a + k * (score_a - expected_a)
    new_elo_b = elo_b + k * ((1 - score_a) - expected_b)
    
    return new_elo_a, new_elo_b


def generate_leaderboard_from_db(leaderboard_data: Dict[str, Any], arena_config: Dict[str, Any], game_name: str = "avalon") -> str:
    """Generate leaderboard from database data.
    
    Args:
        leaderboard_data: Data from LeaderboardDB.get_leaderboard_data()
        arena_config: Arena configuration with models list
        game_name: Name of the game (e.g., 'avalon', 'diplomacy')
    
    Returns:
        Formatted leaderboard string
    """
    models = arena_config.get('models', [])
    model_stats = leaderboard_data.get('models', {})
    balance_stats = leaderboard_data.get('balance', {})
    
    # Get role names based on game type
    if game_name == "avalon":
        role_names = ['Merlin', 'Servant', 'Assassin', 'Minion']
        game_display_name = "AVALON"
    elif game_name == "diplomacy":
        # For Diplomacy, extract power names from role_stats
        all_roles = set()
        for model_stat in model_stats.values():
            all_roles.update(model_stat.get('role_stats', {}).keys())
        role_names = sorted(list(all_roles)) if all_roles else []
        game_display_name = "DIPLOMACY"
    else:
        # Generic: extract from role_stats
        all_roles = set()
        for model_stat in model_stats.values():
            all_roles.update(model_stat.get('role_stats', {}).keys())
        role_names = sorted(list(all_roles)) if all_roles else []
        game_display_name = game_name.upper()
    
    # Sort models by Elo
    sorted_models = sorted(models, 
                          key=lambda m: model_stats.get(m, {}).get('elo', 0),
                          reverse=True)
    
    # Calculate role averages
    role_averages = {}
    for role in role_names:
        role_win_rates = []
        role_games = []
        for model in sorted_models:
            if model in model_stats:
                role_stat = model_stats[model]['role_stats'].get(role, {})
                if role_stat.get('games', 0) > 0:
                    role_win_rates.append(role_stat.get('win_rate', 0))
                    role_games.append(role_stat.get('games', 0))
        if role_win_rates:
            total_games = sum(role_games)
            if total_games > 0:
                weighted_avg = sum(wr * games for wr, games in zip(role_win_rates, role_games)) / total_games
                role_averages[role] = weighted_avg
            else:
                role_averages[role] = sum(role_win_rates) / len(role_win_rates)
        else:
            role_averages[role] = 0
    
    lines = []
    W = 120  # Wider to accommodate Games column
    
    # Header
    lines.append("╔" + "═" * (W - 2) + "╗")
    title = f"{game_display_name} ARENA LEADERBOARD"
    lines.append("║" + " " * ((W - len(title) - 2) // 2) + title + " " * ((W - len(title) - 2) // 2) + "║")
    total_games = leaderboard_data.get('total_games', 0)
    updated_at = leaderboard_data.get('updated_at', '')
    if updated_at:
        lines.append("║" + f" Total Games: {total_games} | Updated: {updated_at}".center(W - 2) + "║")
    
    # Show balance statistics
    if balance_stats:
        balance_ratio = balance_stats.get('balance_ratio', 1.0)
        min_games = balance_stats.get('min', 0)
        max_games = balance_stats.get('max', 0)
        if balance_ratio < 0.8:
            lines.append("║" + f" ⚠️  Game Count Balance: {balance_ratio:.1%} (min={min_games}, max={max_games})".center(W - 2) + "║")
        else:
            lines.append("║" + f" ✓ Game Count Balance: {balance_ratio:.1%} (min={min_games}, max={max_games})".center(W - 2) + "║")
    
    lines.append("╚" + "═" * (W - 2) + "╝")
    lines.append("")
    
    # Compact table with essential metrics
    # Columns: Model | Elo | Overall | Games | Merlin | Servant | Assassin | Minion | Avg
    headers = ["Model", "Elo", "Overall", "Games"] + role_names + ["Avg"]
    col_widths = [16, 8, 9, 8] + [9] * len(role_names) + [9]
    
    # Build header
    header_parts = []
    for h, w in zip(headers, col_widths):
        if h == "Model":
            header_parts.append(f"{h:<{w}}")
        else:
            header_parts.append(f"{h:>{w}}")
    
    lines.append("┌" + "─" * (col_widths[0] - 1) + "┬" + 
                "─┬".join("─" * (w - 1) for w in col_widths[1:]) + "─┐")
    lines.append("│ " + " │ ".join(header_parts) + " │")
    lines.append("├" + "─" * (col_widths[0] - 1) + "┼" + 
                "─┼".join("─" * (w - 1) for w in col_widths[1:]) + "─┤")
    
    # Data rows
    for model in sorted_models:
        if model not in model_stats:
            continue
        
        stats = model_stats[model]
        elo = int(stats.get('elo', 1500))
        win_rate = stats.get('win_rate', 0)
        total_games = stats.get('total_games', 0)
        role_stats = stats.get('role_stats', {})
        
        # Calculate row average (across roles)
        role_wrs = [role_stats.get(role, {}).get('win_rate', 0) 
                   for role in role_names 
                   if role_stats.get(role, {}).get('games', 0) > 0]
        row_avg = sum(role_wrs) / len(role_wrs) if role_wrs else 0
        
        # Mark models with insufficient games
        min_games = balance_stats.get('min', 0) if balance_stats else 0
        max_games = balance_stats.get('max', 0) if balance_stats else 0
        is_insufficient = max_games > 0 and total_games < max_games * 0.8
        
        model_display = model[:14]
        if is_insufficient:
            model_display = f"{model_display}*"  # Mark insufficient games
        
        row_parts = []
        row_parts.append(f"{model_display:<{col_widths[0]}}")
        row_parts.append(f"{elo:>{col_widths[1]}}")
        row_parts.append(f"{win_rate:>5.1f}%")
        row_parts.append(f"{total_games:>{col_widths[3]}}")
        
        for role in role_names:
            if role in role_stats and role_stats[role].get('games', 0) > 0:
                wr = role_stats[role].get('win_rate', 0)
                row_parts.append(f"{wr:>5.1f}%")
            else:
                row_parts.append(f"{'N/A':>{col_widths[4]}}")
        
        row_parts.append(f"{row_avg:>5.1f}%")
        
        lines.append("│ " + " │ ".join(row_parts) + " │")
    
    # Average row
    lines.append("├" + "─" * (col_widths[0] - 1) + "┼" + 
                "─┼".join("─" * (w - 1) for w in col_widths[1:]) + "─┤")
    
    if sorted_models:
        avg_elo = sum(model_stats.get(m, {}).get('elo', 1500) for m in sorted_models) / len(sorted_models)
        avg_overall = sum(model_stats.get(m, {}).get('win_rate', 0) for m in sorted_models) / len(sorted_models)
        avg_games = sum(model_stats.get(m, {}).get('total_games', 0) for m in sorted_models) / len(sorted_models)
    else:
        avg_elo = 0
        avg_overall = 0
        avg_games = 0
    
    avg_row_parts = []
    avg_row_parts.append(f"{'Average':<{col_widths[0]}}")
    avg_row_parts.append(f"{int(avg_elo):>{col_widths[1]}}")
    avg_row_parts.append(f"{avg_overall:>5.1f}%")
    avg_row_parts.append(f"{int(avg_games):>{col_widths[3]}}")
    
    for role in role_names:
        avg_wr = role_averages.get(role, 0)
        avg_row_parts.append(f"{avg_wr:>5.1f}%")
    
    overall_avg = sum(role_averages.values()) / len(role_averages) if role_averages else 0
    avg_row_parts.append(f"{overall_avg:>5.1f}%")
    
    lines.append("│ " + " │ ".join(avg_row_parts) + " │")
    lines.append("└" + "─" * (col_widths[0] - 1) + "┴" + 
                "─┴".join("─" * (w - 1) for w in col_widths[1:]) + "─┘")
    
    # Add footnote for insufficient games
    if any(model_stats.get(m, {}).get('total_games', 0) < max_games * 0.8 
           for m in sorted_models if max_games > 0):
        lines.append("")
        lines.append("* Models with insufficient games (< 80% of max)")
    
    lines.append("")
    
    return "\n".join(lines)

