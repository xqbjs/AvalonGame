# -*- coding: utf-8 -*-
"""Arena workflows that randomly assign models to roles for different games."""
import random
import copy
from typing import Dict, Any, Type, List, Union
from abc import ABC, abstractmethod


# Arena workflow registry
ARENA_WORKFLOW_REGISTRY: Dict[str, Type] = {}


def register_arena_workflow(game_name: str):
    """Decorator to register an arena workflow class."""
    def decorator(workflow_class: Type):
        ARENA_WORKFLOW_REGISTRY[game_name] = workflow_class
        return workflow_class
    return decorator


class BaseArenaWorkflow(ABC):
    """Base class for arena workflows with model assignment logic."""
    
    def _initialize_arena(self, config_dict: Dict[str, Any]):
        """Initialize arena configuration and assign models."""
        arena_config = config_dict.get('arena', {})
        self.arena_models = arena_config.get('models', [])
        if not self.arena_models:
            raise ValueError("arena.models must be specified in config")
        
        if seed := arena_config.get('seed'):
            random.seed(seed)
        
        self.original_config = copy.deepcopy(config_dict)
        self._assign_models_to_roles(config_dict)
    
    def _assign_models_to_roles(self, config_dict: Dict[str, Any]):
        """Assign models to roles with fairness and diversity."""
        role_names = self._get_role_names(config_dict)
        game_counts = config_dict.get('_model_game_counts', {})
        
        # Calculate weights: fewer games = higher weight (ensures fairness)
        weights = [1.0 / (game_counts.get(m, 0) + 1) for m in self.arena_models]
        total = sum(weights)
        weights = [w / total for w in weights] if total > 0 else [1.0 / len(self.arena_models)] * len(self.arena_models)
        
        assigned_models = self._select_models(len(role_names), weights)
        
        # Update config
        config_dict['_arena_model_assignment'] = self._create_model_assignment(role_names, assigned_models)
        # Ensure 'roles' is a dict (handle case where it might be None)
        if 'roles' not in config_dict or config_dict['roles'] is None:
            config_dict['roles'] = {}
        config_dict['roles'].update(
            {name: {'model_name': model} for name, model in zip(role_names, assigned_models)}
        )
    
    def _select_models(self, num_roles: int, weights: List[float]) -> List[str]:
        """Select models with weighted sampling without replacement.
        
        Ensures:
        1. Diversity: No duplicate models in a single game (when possible)
        2. Fairness: Models with fewer games have higher selection probability
        3. Balance: Long-term game count distribution is balanced
        """
        if len(self.arena_models) >= num_roles:
            # Weighted sampling without replacement for maximum diversity
            # Use weighted random selection to ensure both diversity and fairness
            assigned = []
            remaining_models = list(self.arena_models)
            remaining_weights = list(weights)
            
            for _ in range(num_roles):
                if not remaining_models:
                    break
                
                # Normalize remaining weights
                total_weight = sum(remaining_weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in remaining_weights]
                else:
                    normalized_weights = [1.0 / len(remaining_models)] * len(remaining_models)
                
                # Select one model based on weights
                selected_idx = random.choices(range(len(remaining_models)), weights=normalized_weights, k=1)[0]
                assigned.append(remaining_models.pop(selected_idx))
                remaining_weights.pop(selected_idx)
            
            # Shuffle to randomize positions
            random.shuffle(assigned)
        else:
            # Not enough models: use all models + weighted selection for remaining
            assigned = list(self.arena_models)
            remaining = num_roles - len(assigned)
            if remaining > 0:
                # Use weighted random selection for remaining slots (allows duplicates)
                assigned.extend(random.choices(self.arena_models, weights=weights, k=remaining))
            random.shuffle(assigned)
        
        return assigned
    
    async def _execute_async(self) -> Dict[str, Any]:
        """Execute game and add model information to results."""
        result = await super()._execute_async()
        if model_assignment := self.config_dict.get('_arena_model_assignment'):
            self._add_models_to_results(result, model_assignment)
        
        # Add language information to result
        game_config = self.config_dict.get('game', {})
        result['language'] = game_config.get('language', 'en')
        
        return result
    
    @abstractmethod
    def _get_role_names(self, config_dict: Dict[str, Any]) -> List[str]:
        """Get list of role names for this game."""
        pass
    
    @abstractmethod
    def _create_model_assignment(self, role_names: List[str], assigned_models: List[str]) -> Union[List[str], Dict[str, str]]:
        """Create model assignment structure."""
        pass
    
    @abstractmethod
    def _add_models_to_results(self, result: Dict[str, Any], model_assignment: Union[List[str], Dict[str, str]]):
        """Add model information to game results."""
        pass


def _register_avalon_workflow():
    """Lazy-load Avalon arena workflow."""
    from games.games.avalon.workflows.eval_workflow import EvalAvalonWorkflow
    
    @register_arena_workflow("avalon")
    class ArenaAvalonWorkflow(BaseArenaWorkflow, EvalAvalonWorkflow):
        """Arena workflow for Avalon with random model assignment."""
        
        def __init__(self, config_dict: Dict[str, Any]):
            self._initialize_arena(config_dict)
            super().__init__(config_dict)
        
        def _get_role_names(self, config_dict: Dict[str, Any]) -> List[str]:
            num_players = config_dict.get('game', {}).get('num_players', 5)
            return [f'player_{i}' for i in range(num_players)]
        
        def _create_model_assignment(self, role_names: List[str], assigned_models: List[str]) -> List[str]:
            return assigned_models
        
        def _get_model_config(self, indexed_role: str, base_role: str) -> Dict[str, Any]:
            """Map runtime role back to assigned model."""
            assigned_models = self.config_dict.get('_arena_model_assignment', [])
            if assigned_models and self.role_manager:
                for i, model_name in enumerate(assigned_models):
                    if self.role_manager.get_indexed_role(i) == indexed_role:
                        default_role = self.config_dict.get('default_role', {})
                        config = copy.deepcopy(default_role.get('model', {}) if isinstance(default_role, dict) else {})
                        config['model_name'] = model_name
                        return config
            return super()._get_model_config(indexed_role, base_role)
        
        def _add_models_to_results(self, result: Dict[str, Any], model_assignment: List[str]):
            if 'roles' in result:
                for i, role_info in enumerate(result['roles']):
                    if i < len(model_assignment):
                        role_info['model_name'] = model_assignment[i]
    
    return ArenaAvalonWorkflow


def _register_diplomacy_workflow():
    """Lazy-load Diplomacy arena workflow."""
    from games.games.diplomacy.workflows.eval_workflow import EvalDiplomacyWorkflow
    
    @register_arena_workflow("diplomacy")
    class ArenaDiplomacyWorkflow(BaseArenaWorkflow, EvalDiplomacyWorkflow):
        """Arena workflow for Diplomacy with random model assignment."""
        
        def __init__(self, config_dict: Dict[str, Any]):
            self._initialize_arena(config_dict)
            super().__init__(config_dict)
        
        def _get_role_names(self, config_dict: Dict[str, Any]) -> List[str]:
            return config_dict.get('game', {}).get('power_names', 
                ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"])
        
        def _create_model_assignment(self, role_names: List[str], assigned_models: List[str]) -> Dict[str, str]:
            return dict(zip(role_names, assigned_models))
        
        def _get_model_config(self, power_name: str) -> Dict[str, Any]:
            """Override to get model from arena assignment."""
            model_assignment = self.config_dict.get('_arena_model_assignment', {})
            if isinstance(model_assignment, dict) and power_name in model_assignment:
                model_name = model_assignment[power_name]
                # Get base model config from default_role.model
                default_role = self.config_dict.get('default_role', {})
                if not isinstance(default_role, dict):
                    default_role = {}
                base_model_config = default_role.get('model', {})
                # Deep copy and update with arena-assigned model name
                config = copy.deepcopy(base_model_config)
                config['model_name'] = model_name
                return config
            # Fallback to parent method
            return super()._get_model_config(power_name)
        
        def _add_models_to_results(self, result: Dict[str, Any], model_assignment: Dict[str, str]):
            if 'roles' in result:
                for role_info in result['roles']:
                    if (power_name := role_info.get('role_name')) in model_assignment:
                        role_info['model_name'] = model_assignment[power_name]
    
    return ArenaDiplomacyWorkflow


# Registry of lazy-load functions
_LAZY_LOADERS = {
    "avalon": _register_avalon_workflow,
    "diplomacy": _register_diplomacy_workflow,
}


def create_arena_workflow(game_name: str, config_dict: Dict[str, Any]):
    """Factory function to create arena workflow for a game."""
    # Lazy-load workflow if not registered
    if game_name not in ARENA_WORKFLOW_REGISTRY and game_name in _LAZY_LOADERS:
        _LAZY_LOADERS[game_name]()
    
    if game_name not in ARENA_WORKFLOW_REGISTRY:
        available = ', '.join(list(ARENA_WORKFLOW_REGISTRY.keys()) + list(_LAZY_LOADERS.keys()))
        raise ValueError(f"Game '{game_name}' not found in arena registry. Available: {available}")
    
    return ARENA_WORKFLOW_REGISTRY[game_name](config_dict=config_dict)
