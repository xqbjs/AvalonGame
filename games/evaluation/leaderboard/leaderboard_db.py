# -*- coding: utf-8 -*-
"""Leaderboard database for persistent storage and incremental updates."""
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class LeaderboardDB:
    """Persistent leaderboard database."""
    
    def __init__(self, db_path: str = "games/evaluation/leaderboard/leaderboard.json"):
        """Initialize leaderboard database.
        
        Args:
            db_path: Path to JSON database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
        self._lock = threading.Lock()  # Thread-safe operations
    
    def _load(self) -> Dict[str, Any]:
        """Load leaderboard data from file."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load leaderboard: {e}, starting fresh")
        
        return {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'models': {},
            'games_history': [],
            'elo_initial': 1500,
            'elo_k': 32
        }
    
    def save(self):
        """Save leaderboard data to file (thread-safe)."""
        with self._lock:
            self.data['updated_at'] = datetime.now().isoformat()
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def get_model_stats(self, model: str) -> Dict[str, Any]:
        """Get statistics for a model."""
        return self.data['models'].get(model, {
            'elo': self.data.get('elo_initial', 1500),
            'total_games': 0,
            'total_wins': 0,
            'role_stats': {},
            'first_seen': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        })
    
    def add_model(self, model: str, initial_elo: Optional[int] = None):
        """Add a new model to the leaderboard.
        
        Args:
            model: Model name
            initial_elo: Initial Elo rating (defaults to elo_initial)
        """
        with self._lock:
            if model not in self.data['models']:
                self.data['models'][model] = {
                    'elo': initial_elo or self.data.get('elo_initial', 1500),
                    'total_games': 0,
                    'total_wins': 0,
                    'role_stats': {},
                    'first_seen': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }
                self.data['updated_at'] = datetime.now().isoformat()
                with open(self.db_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, indent=2, ensure_ascii=False)
                print(f"Added new model to leaderboard: {model} (Elo: {self.data['models'][model]['elo']})")
    
    def update_from_game_results(self, results: List[Dict[str, Any]]):
        """Update leaderboard from game results (thread-safe).
        
        Args:
            results: List of game results, each containing 'roles' with model_name and score
        """
        from games.evaluation.leaderboard.leaderboard import calculate_elo
        
        # Thread-safe update
        with self._lock:
            elo_k = self.data.get('elo_k', 32)
            
            for result in results:
                if 'roles' not in result:
                    continue
                
                # Extract models and their scores in this game
                game_models = {}
                for role_info in result['roles']:
                    if 'model_name' in role_info and 'score' in role_info:
                        model = role_info['model_name']
                        score = role_info['score']
                        role_name = role_info.get('role_name', 'unknown').split('_')[0]  # Extract base role
                        
                        # Ensure model exists
                        if model not in self.data['models']:
                            # Create model entry without saving (will save at end)
                            self.data['models'][model] = {
                                'elo': self.data.get('elo_initial', 1500),
                                'total_games': 0,
                                'total_wins': 0,
                                'role_stats': {},
                                'first_seen': datetime.now().isoformat(),
                                'last_updated': datetime.now().isoformat()
                            }
                        
                        game_models[model] = {
                            'score': score,
                            'role': role_name
                        }
                
                # Update statistics
                for model, info in game_models.items():
                    stats = self.data['models'][model]
                    stats['total_games'] += 1
                    stats['total_wins'] += info['score']
                    
                    # Update role stats
                    role = info['role']
                    if role not in stats['role_stats']:
                        stats['role_stats'][role] = {'wins': 0, 'games': 0}
                    stats['role_stats'][role]['games'] += 1
                    stats['role_stats'][role]['wins'] += info['score']
                    stats['last_updated'] = datetime.now().isoformat()
                
                # Update Elo scores (pairwise comparison)
                # Normalize scores to 0-1 range for Elo calculation
                scores = [info['score'] for info in game_models.values()]
                if scores:
                    min_score = min(scores)
                    max_score = max(scores)
                    score_range = max_score - min_score if max_score > min_score else 1
                    
                    model_list = list(game_models.keys())
                    for i, model_a in enumerate(model_list):
                        for model_b in model_list[i+1:]:
                            score_a_raw = game_models[model_a]['score']
                            score_b_raw = game_models[model_b]['score']
                            
                            # Normalize to 0-1: (score - min) / range
                            # For binary scores (0/1), this gives 0 or 1
                            # For continuous scores, this gives relative performance
                            score_a = (score_a_raw - min_score) / score_range if score_range > 0 else 0.5
                            
                            elo_a = self.data['models'][model_a]['elo']
                            elo_b = self.data['models'][model_b]['elo']
                            
                            new_elo_a, new_elo_b = calculate_elo(elo_a, elo_b, score_a, elo_k)
                            
                            self.data['models'][model_a]['elo'] = new_elo_a
                            self.data['models'][model_b]['elo'] = new_elo_b
                
                    # Record game in history
                    history_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'models': list(game_models.keys()),
                        'results': {m: info['score'] for m, info in game_models.items()}
                    }
                    # Add language if available in result
                    if 'language' in result:
                        history_entry['language'] = result['language']
                    self.data['games_history'].append(history_entry)
            
            # Save after all updates (inside lock)
            self.data['updated_at'] = datetime.now().isoformat()
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def get_all_models(self) -> List[str]:
        """Get list of all models in leaderboard - thread-safe."""
        with self._lock:
            return list(self.data['models'].keys())
    
    def get_model_game_counts(self, model_list: Optional[List[str]] = None) -> Dict[str, int]:
        """Get game count for models (for fair assignment) - thread-safe.
        
        Args:
            model_list: Optional list of specific models to get counts for.
                       If None, returns counts for all models in database.
                       Models not in database will have count 0.
        
        Returns:
            Dictionary mapping model names to their game counts
        """
        with self._lock:
            if model_list is None:
                # Return all models in database
                return {model: self.data['models'][model]['total_games'] 
                        for model in self.data['models'].keys()}
            else:
                # Return counts for specific models, ensuring all are included
                counts = {}
                for model in model_list:
                    if model in self.data['models']:
                        counts[model] = self.data['models'][model]['total_games']
                    else:
                        # Model not in database yet, return 0
                        counts[model] = 0
                return counts
    
    def get_min_game_count(self) -> int:
        """Get minimum game count among all models."""
        counts = self.get_model_game_counts().values()
        return min(counts) if counts else 0
    
    def get_max_game_count(self) -> int:
        """Get maximum game count among all models."""
        counts = self.get_model_game_counts().values()
        return max(counts) if counts else 0
    
    def get_game_count_balance(self) -> Dict[str, Any]:
        """Get statistics about game count balance - thread-safe."""
        with self._lock:
            counts = [self.data['models'][model]['total_games'] 
                     for model in self.data['models'].keys()]
            if not counts:
                return {
                    'min': 0,
                    'max': 0,
                    'mean': 0,
                    'std': 0,
                    'balance_ratio': 1.0
                }
            
            import statistics
            mean_count = statistics.mean(counts)
            std_count = statistics.stdev(counts) if len(counts) > 1 else 0
            min_count = min(counts)
            max_count = max(counts)
            
            # Balance ratio: min/max (1.0 = perfectly balanced, 0.0 = completely unbalanced)
            balance_ratio = min_count / max_count if max_count > 0 else 1.0
            
            return {
                'min': min_count,
                'max': max_count,
                'mean': round(mean_count, 1),
                'std': round(std_count, 1),
                'balance_ratio': round(balance_ratio, 3)
            }
    
    def get_leaderboard_data(self) -> Dict[str, Any]:
        """Get formatted leaderboard data for display - thread-safe."""
        with self._lock:
            models = list(self.data['models'].keys())
            model_stats = {}
            
            for model in models:
                stats = self.data['models'][model]
                total_games = stats['total_games']
                total_wins = stats['total_wins']
                win_rate = (total_wins / total_games * 100) if total_games > 0 else 0
                
                # Calculate role win rates
                role_stats = {}
                for role, role_data in stats['role_stats'].items():
                    role_games = role_data['games']
                    role_wins = role_data['wins']
                    if role_games > 0:
                        role_stats[role] = {
                            'win_rate': (role_wins / role_games * 100),
                            'games': role_games
                        }
                
                model_stats[model] = {
                    'elo': stats['elo'],
                    'win_rate': win_rate,
                    'total_games': total_games,
                    'total_wins': total_wins,
                    'role_stats': role_stats
                }
            
            # Calculate balance (outside lock to avoid nested lock)
            counts = [stats['total_games'] for stats in self.data['models'].values()]
            if counts:
                import statistics
                mean_count = statistics.mean(counts)
                std_count = statistics.stdev(counts) if len(counts) > 1 else 0
                min_count = min(counts)
                max_count = max(counts)
                balance_ratio = min_count / max_count if max_count > 0 else 1.0
                balance_stats = {
                    'min': min_count,
                    'max': max_count,
                    'mean': round(mean_count, 1),
                    'std': round(std_count, 1),
                    'balance_ratio': round(balance_ratio, 3)
                }
            else:
                balance_stats = {
                    'min': 0,
                    'max': 0,
                    'mean': 0,
                    'std': 0,
                    'balance_ratio': 1.0
                }
            
            return {
                'models': model_stats,
                'total_games': len(self.data['games_history']),
                'updated_at': self.data['updated_at'],
                'balance': balance_stats
            }

