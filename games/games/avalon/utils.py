# -*- coding: utf-8 -*-
"""Utility functions and classes for the Avalon game."""
import json
import os
import re
from datetime import datetime
from typing import Any

import numpy as np
from agentscope.agent import AgentBase
from loguru import logger


# ============================================================================
# Parser
# ============================================================================

class Parser:
    """Parser class for parsing agent responses."""
    
    APPROVE_KEYWORDS = ['yes', 'approve', 'accept', '是', '同意', '通过']
    REJECT_KEYWORDS = ['no', 'reject', '否', '拒绝', '不同意']
        
    @staticmethod
    def extract_text_from_content(content: str | list) -> str:
        """Extract text string from agentscope message content."""
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content", "")))
                else:
                    parts.append(str(item))
            return " ".join(parts)
        
        return str(content)
    
    @staticmethod
    def parse_team_from_response(response: str | list) -> list[int]:
        """Parse team list from agent response."""
        text = Parser.extract_text_from_content(response)
        
        # Try to find list pattern like [0, 1, 2]
        list_match = re.search(r'$$[\s]*\d+[\s]*(?:,[\s]*\d+[\s]*)*$$', text)
        if list_match:
            return [int(n) for n in re.findall(r'\d+', list_match.group())]
        
        # Fallback: extract all numbers (limit to 10 players)
        return [int(n) for n in re.findall(r'\d+', text)[:10]]
    
    @staticmethod
    def parse_vote_from_response(response: str | list) -> int:
        """Parse vote (0 or 1) from agent response."""
        text = Parser.extract_text_from_content(response).lower().strip()
        
        if any(kw in text for kw in Parser.APPROVE_KEYWORDS):
            return 1
        return 0  # Default to reject
    
    @staticmethod
    def parse_player_id_from_response(response: str | list, max_id: int) -> int:
        """Parse player ID from agent response."""
        text = Parser.extract_text_from_content(response)
        numbers = re.findall(r'\d+', text)
        
        if numbers:
            return max(0, min(int(numbers[-1]), max_id))
        return 0


# ============================================================================
# Logger
# ============================================================================

class GameLogger:
    """Logger class for game logging functionality."""
    
    def __init__(self):
        self.game_log = {
            "initialization": {},
            "missions": [],
            "assassination": None,
            "game_end": None,
        }
        self.game_log_dir = None
    
    def initialize_game_log(self, roles: list[tuple], num_players: int) -> None:
        """Initialize game log with roles and player count."""
        self.game_log["initialization"] = {
            "roles": [(role_id, role_name, side) for role_id, role_name, side in roles],
            "num_players": num_players,
        }
    
    def create_game_log_dir(self, log_dir: str | None, timestamp: str | None = None) -> str | None:
        """Create game log directory and return the path.
        
        Args:
            log_dir: Base directory for logs. If None, returns None.
            timestamp: Optional timestamp string. If None, generates a new one.
        """
        if not log_dir:
            return None
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.game_log_dir = os.path.join(log_dir, f"game_{timestamp}")
        os.makedirs(self.game_log_dir, exist_ok=True)
        logger.info(f"Game logs will be saved to: {self.game_log_dir}")
        return self.game_log_dir
    
    def add_mission(self, mission_id: int, round_id: int, leader: int) -> None:
        """Add a new mission entry to the game log."""
        if not self.game_log_dir:
            return
        
        self.game_log["missions"].append({
            "mission_id": mission_id,
            "round_id": round_id,
            "leader": leader,
            "discussion": [],
            "team_proposed": [],
        })
    
    def add_discussion_messages(self, discussion_msgs: list[dict]) -> None:
        """Add discussion messages to the current mission."""
        if self.game_log_dir and self.game_log["missions"]:
            self.game_log["missions"][-1]["discussion"] = discussion_msgs
    
    def add_team_proposal(self, team: list[int]) -> None:
        """Add team proposal to the current mission."""
        if self.game_log_dir and self.game_log["missions"]:
            self.game_log["missions"][-1]["team_proposed"] = team
    
    def add_team_voting(self, team: list[int], votes: list[int], approved: bool) -> None:
        """Add team voting results to the current mission."""
        if self.game_log_dir and self.game_log["missions"]:
            self.game_log["missions"][-1]["team_voting"] = {
                "team": team,
                "votes": votes,
                "approved": approved,
            }
    
    def add_quest_voting(self, team: list[int], votes: list[int], num_fails: int, succeeded: bool) -> None:
        """Add quest voting results to the current mission."""
        if self.game_log_dir and self.game_log["missions"]:
            self.game_log["missions"][-1]["quest_voting"] = {
                "team": team,
                "votes": votes,
                "num_fails": num_fails,
                "succeeded": succeeded,
            }
    
    def add_assassination(self, assassin_id: int, target: int, good_wins: bool) -> None:
        """Add assassination results to the game log."""
        if self.game_log_dir:
            self.game_log["assassination"] = {
                "assassin_id": assassin_id,
                "target": target,
                "good_wins": good_wins,
            }
    
    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """Convert numpy types to Python native types."""
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: GameLogger._convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [GameLogger._convert_to_serializable(item) for item in obj]
        return obj
    
    async def save_game_logs(self, agents: list[AgentBase], env: Any, roles: list[tuple]) -> None:
        """Save game logs including agent memories and game log."""
        if not self.game_log_dir:
            return
        
        self._save_game_log_json(env, roles, agents)
        await self._save_agent_memories(agents, roles)
    
    def _save_game_log_json(self, env: Any, roles: list[tuple], agents: list[AgentBase] = None) -> None:
        """Save game log to JSON file.
        
        Args:
            env: Game environment
            roles: List of roles as tuples (role_id, role_name, is_good)
            agents: Optional list of agents to extract model names from
        """
        self.game_log["game_end"] = {
            "good_victory": env.good_victory,
            "quest_results": env.quest_results,
        }
        
        # Extract model names from agents if available
        model_names = []
        if agents is not None:
            for i, agent in enumerate(agents):
                model_name = "Unknown"
                try:
                    # Try to get model name from agent.model
                    if hasattr(agent, 'model') and agent.model is not None:
                        if hasattr(agent.model, 'model_name'):
                            model_name = agent.model.model_name
                        elif hasattr(agent.model, 'name'):
                            model_name = agent.model.name
                except Exception:
                    pass
                model_names.append(model_name)
        
        game_log_data = {
            "roles": [(int(r), n, bool(s)) for r, n, s in roles],
            "game_result": {
                "good_victory": bool(env.good_victory),
                "quest_results": [bool(r) for r in env.quest_results],
            },
            "game_log": self._convert_to_serializable(self.game_log),
        }
        
        # Add model names if available
        if model_names:
            game_log_data["model_names"] = model_names
        
        path = os.path.join(self.game_log_dir, "game_log.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(game_log_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Game log saved to {path}")
    
    async def _save_agent_memories(self, agents: list[AgentBase], roles: list[tuple]) -> None:
        """Save each agent's memory and model call history to separate JSON files."""
        for i, agent in enumerate(agents):
            try:
                agent_data = {
                    "agent_name": agent.name,
                    "agent_index": i,
                    "role": roles[i][1] if i < len(roles) else "Unknown",
                }
                
                # Save memory if available
                if hasattr(agent, 'memory') and agent.memory is not None:
                    agent_memory = await agent.memory.get_memory()
                    agent_data["memory_count"] = len(agent_memory)
                    agent_data["memory"] = [msg.to_dict() for msg in agent_memory]
                
                # Save model call history if available (for ThinkingReActAgent)
                if hasattr(agent, 'model_call_history'):
                    # Convert model call history to serializable format
                    serializable_history = []
                    for call_record in agent.model_call_history:
                        serializable_record = {
                            "prompt": call_record.get("prompt", ""),
                            "response": call_record.get("response", ""),
                            "response_msg": self._convert_to_serializable(call_record.get("response_msg", {})),
                        }
                        serializable_history.append(serializable_record)
                    agent_data["model_call_history"] = serializable_history
                    agent_data["model_call_count"] = len(serializable_history)
                    
                    # Log model call history to logger
                    logger.info(f"Agent {agent.name} model call history: {agent_data['model_call_count']} calls")
                    for idx, call_record in enumerate(serializable_history):
                        logger.debug(f"Agent {agent.name} call {idx + 1}: prompt length={len(call_record.get('prompt', ''))}, response length={len(call_record.get('response', ''))}")
                
                # Only save if we have data
                if "memory" in agent_data or "model_call_history" in agent_data:
                    path = os.path.join(self.game_log_dir, f"{agent.name}_memory.json")
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(agent_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Agent {agent.name} memory and model calls saved to {path}")
            except Exception as e:
                logger.warning(f"Failed to save memory for agent {agent.name}: {e}")


# ============================================================================
# Language Formatter
# ============================================================================

class LanguageFormatter:
    """Language formatter helper to handle language-specific formatting."""
    
    LANGUAGE_CONFIG = {
        "zh": {
            "role_names": {
                "Merlin": "梅林", "Servant": "忠臣", "Assassin": "刺客", "Minion": "爪牙",
                "Percival": "派西维尔", "Morgana": "莫甘娜", "Mordred": "莫德雷德", "Oberon": "奥伯伦",
            },
            "side_names": {"Good": "好人", "Evil": "坏人"},
            "player_prefix": "玩家",
            "separator": "和",
            "vote_approve": "批准",
            "vote_reject": "拒绝",
            "is_text": "是",
        },
        "en": {
            "role_names": {},
            "side_names": {"Good": "Good", "Evil": "Evil"},
            "player_prefix": "Player",
            "separator": "and",
            "vote_approve": "Approve",
            "vote_reject": "Reject",
            "is_text": "is",
        }
    }
    
    def __init__(self, language: str = "en"):
        """Initialize language formatter with language code."""
        self.is_zh = language.lower() in ["zh", "cn", "chinese"]
        config = self.LANGUAGE_CONFIG["zh" if self.is_zh else "en"]
        
        self.role_names = config["role_names"]
        self.side_names = config["side_names"]
        self.player_prefix = config["player_prefix"]
        self.separator = config["separator"]
        self.vote_approve = config["vote_approve"]
        self.vote_reject = config["vote_reject"]
        self.is_text = config["is_text"]
    
    def format_player_name(self, agent_name: str) -> str:
        """Format player name (Player0 -> 玩家 0)."""
        if not self.is_zh or not agent_name.startswith("Player"):
            return agent_name
        
        return f"{self.player_prefix} {agent_name.replace('Player', '')}"
    
    def format_player_id(self, player_id: int) -> str:
        """Format player ID (0 -> '玩家 0' or 'Player 0')."""
        return f"{self.player_prefix} {player_id}"
    
    def format_role_name(self, role_name: str) -> str:
        """Format role name (Merlin -> 梅林)."""
        return self.role_names.get(role_name, role_name)
    
    def format_side_name(self, side: bool) -> str:
        """Format side name (True -> '好人' or 'Good')."""
        return self.side_names["Good" if side else "Evil"]
    
    def format_agents_names(self, agents: list[AgentBase]) -> str:
        """Format list of agent names for display."""
        if not agents:
            return ""
        
        names = [self.format_player_name(a.name) for a in agents]
        if len(names) == 1:
            return names[0]
        
        return f"{', '.join(names[:-1])} {self.separator} {names[-1]}"
    
    def format_vote_details(self, votes: list[int], approved: bool) -> tuple[str, str, str]:
        """Format vote details for display. Returns (votes_detail, result_text, outcome_text)."""
        approve_text = self.vote_approve
        reject_text = self.vote_reject
        result_text = approve_text if approved else reject_text
        
        votes_detail = ", ".join([
            f"{self.format_player_id(i)}: {result_text if v == approved else (reject_text if approved else approve_text)}"
            for i, v in enumerate(votes)
        ])
        
        outcome_text = result_text if self.is_zh else result_text.lower() + "d"
        
        return votes_detail, result_text, outcome_text
    
    def format_sides_info(self, roles: list[tuple]) -> list[str]:
        """Format sides information for visibility."""
        return [
            f"{self.format_player_id(j)} {self.is_text} {self.format_side_name(s)}"
            for j, (_, _, s) in enumerate(roles)
        ]
    
    def calculate_role_counts(self, config: Any) -> dict[str, Any]:
        """Calculate role counts for system prompt."""
        merlin_count = 1 if config.merlin else 0
        percival_count = 1 if config.percival else 0
        
        return {
            "num_players": config.num_players,
            "max_player_id": config.num_players - 1,
            "num_good": config.num_good,
            "merlin_count": merlin_count,
            "servant_count": config.num_good - merlin_count - percival_count,
            "num_evil": config.num_evil,
            "assassin_count": 1,
            "minion_count": config.num_evil - 1,
            "percival_count": percival_count,
        }
    
    def format_system_prompt(self, config: Any, prompts_class: Any) -> str:
        """Format system prompt with role counts."""
        return prompts_class.system_prompt_template.format(**self.calculate_role_counts(config))
    
    def format_true_roles(self, roles: list[tuple]) -> str:
        """Format true roles for game end display."""
        return ", ".join([
            f"{self.format_player_id(i)}: {self.format_role_name(role_name)}"
            for i, (_, role_name, _) in enumerate(roles)
        ])
    
    def format_game_end_message(self, good_victory: bool, roles: list[tuple], prompts_class: Any) -> str:
        """Format game end message with result and true roles."""
        result = prompts_class.to_all_good_wins if good_victory else prompts_class.to_all_evil_wins
        true_roles_str = self.format_true_roles(roles)
        return prompts_class.to_all_game_end.format(result=result, true_roles=true_roles_str)

