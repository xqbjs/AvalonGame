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
        self.stream_file_path = None  # [NEW] 新增：实时日志路径
        
    
    def _append_stream_log(self, event_type: str, data: dict) -> None:
        """Append a single log entry to the stream file immediately."""
        if not self.game_log_dir:
            return
            
        # 懒加载
        if not self.stream_file_path:
            self.stream_file_path = os.path.join(self.game_log_dir, "game_stream.jsonl")
            
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": event_type,
            "data": self._convert_to_serializable(data)
        }
        
        try:
            with open(self.stream_file_path, 'a', encoding='utf-8') as f:
                # [关键修改] indent=4 让 JSON 格式化输出，ensure_ascii=False 显示中文
                f.write(json.dumps(entry, ensure_ascii=False, indent=4))
                # 额外写入一个分割线，方便在极长的日志中区分不同的条目
                f.write("\n" + "-"*50 + "\n") 
        except Exception as e:
            logger.warning(f"Failed to append stream log: {e}")
    
    
    def initialize_game_log(self, roles: list[tuple], num_players: int) -> None:
        """Initialize game log with roles and player count."""
        # 1. 准备总日志数据
        init_data = {
            "roles": [(role_id, role_name, side) for role_id, role_name, side in roles],
            "num_players": num_players,
        }
        
        # 2. 存入内存总日志
        self.game_log["initialization"] = init_data
        
        # 3. [实时] 写入总流水 (game_stream.jsonl)
        self._append_stream_log("initialization", init_data)
        
        # 4. [NEW] 核心修复：立刻为所有 Player 创建独立日志文件！
        # 即使他们还没说话，文件也会在游戏开始的瞬间被创建
        for i in range(num_players):
            # 获取该玩家的角色信息
            my_role = "Unknown"
            if i < len(roles):
                my_role = roles[i][1] # role_name
            
            agent_name = f"Player{i}"
            
            # 强制调用 log_agent_event 写入一条初始日志
            # 这会触发 open(..., 'a')，从而在硬盘上立刻创建文件
            self.log_agent_event(i, agent_name, {
                "type": "initialization", 
                "role": my_role,
                "message": "Session started, log file initialized."
            })
    
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
        # [NEW] 立即写入任务开始事件
        self._append_stream_log("mission_start", {
            "mission_id": mission_id, "round_id": round_id, "leader": leader
        })
    
    def add_discussion_messages(self, discussion_msgs: list[dict]) -> None:
        """Add discussion messages to the current mission."""
        if self.game_log_dir and self.game_log["missions"]:
            self.game_log["missions"][-1]["discussion"] = discussion_msgs
        # [NEW] 立即写入讨论内容
            self._append_stream_log("discussion", {"messages": discussion_msgs})
    
    # [NEW] 核心功能：单句实时写入 (Game.py 中调用此方法)
    def add_single_dialogue(self, message: dict) -> None:
        """Log a single dialogue message immediately to the stream."""
        # 1. 写入硬盘流式日志
        self._append_stream_log("dialogue", message)

        # 2. 同时维护内存里的完整性
        try:
            if self.game_log["missions"]:
                current_mission = self.game_log["missions"][-1]
                # 确保 discussion 是个列表
                if "discussion" not in current_mission or not isinstance(current_mission["discussion"], list):
                    current_mission["discussion"] = []
                current_mission["discussion"].append(message)
        except Exception:
            pass 

    
    # [修改] 增加 indent=4 参数，使输出格式化换行，方便阅读
    def log_agent_event(self, agent_index: int, agent_name: str, event_data: dict) -> None:
        """Append a log entry to a specific agent's log file immediately."""
        if not self.game_log_dir:
            return
            
        filename = f"{agent_name}_stream.jsonl"
        filepath = os.path.join(self.game_log_dir, filename)
        
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": self._convert_to_serializable(event_data)
        }
        
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                # [关键修改] indent=4 让 JSON 格式化输出
                f.write(json.dumps(entry, ensure_ascii=False, indent=4))
                # 额外写入一个分割线
                f.write("\n" + "-"*50 + "\n")
        except Exception as e:
            logger.warning(f"Failed to append agent log: {e}")
            
    # [NEW] 广播写入：把一条消息同时写入所有玩家的日志文件
    # 用于 Moderator 的公告、公开讨论等所有人都能听到的内容
    def log_broadcast_event(self, message: dict, num_players: int) -> None:
        """Log a public message to ALL players' stream files."""
        if not self.game_log_dir:
            return
            
        # 遍历所有玩家，给每个人都记上一笔
        for i in range(num_players):
            agent_name = f"Player{i}"
            self.log_agent_event(i, agent_name, message)      
    
    
    def add_team_proposal(self, team: list[int]) -> None:
        """Add team proposal to the current mission."""
        if self.game_log_dir and self.game_log["missions"]:
            self.game_log["missions"][-1]["team_proposed"] = team
            # [NEW] 立即写入提议
            self._append_stream_log("team_proposal", {"team": team})
    
    
    def add_team_voting(self, team: list[int], votes: list[int], approved: bool) -> None:
        """Add team voting results to the current mission."""
        if self.game_log_dir and self.game_log["missions"]:
            # [FIX] 必须先定义变量，下面才能用
            vote_data = {
                "team": team,
                "votes": votes,
                "approved": approved,
            }
            
            # 存入内存
            self.game_log["missions"][-1]["team_voting"] = vote_data
            
            # [NEW] 立即写入流式日志
            self._append_stream_log("team_voting", vote_data)
    
    
    def add_quest_voting(self, team: list[int], votes: list[int], num_fails: int, succeeded: bool) -> None:
        """Add quest voting results to the current mission."""
        if self.game_log_dir and self.game_log["missions"]:
            # [FIX] 必须先定义变量
            quest_data = {
                "team": team,
                "votes": votes,
                "num_fails": num_fails,
                "succeeded": succeeded,
            }
            
            # 存入内存
            self.game_log["missions"][-1]["quest_voting"] = quest_data
            
            # [NEW] 立即写入流式日志
            self._append_stream_log("quest_voting", quest_data)
    
    
    def add_assassination(self, assassin_id: int, target: int, good_wins: bool) -> None:
        """Add assassination results to the game log."""
        if self.game_log_dir:
            # [FIX] 必须先定义变量
            assassin_data = {
                "assassin_id": assassin_id,
                "target": target,
                "good_wins": good_wins,
            }
            
            # 存入内存
            self.game_log["assassination"] = assassin_data
            
            # [NEW] 立即写入流式日志
            self._append_stream_log("assassination", assassin_data)
    
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
        # [NEW] 记录游戏结束
        self._append_stream_log("game_end", self.game_log["game_end"])
        
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
    
    
    # [FIX] 核心修复：更正角色数量统计逻辑
    def calculate_role_counts(self, config: Any) -> dict[str, Any]:
        """Calculate role counts for system prompt."""
        # 统计好人阵营
        merlin_count = 1 if config.merlin else 0
        percival_count = 1 if config.percival else 0
        servant_count = config.num_good - merlin_count - percival_count
        
        # 统计坏人阵营
        assassin_count = 1 # 默认总有刺客
        morgana_count = 1 if config.morgana else 0
        mordred_count = 1 if config.mordred else 0
        oberon_count = 1 if config.oberon else 0
        
        # [FIX] 爪牙数量 = 总坏人 - 所有特殊坏人
        minion_count = config.num_evil - (assassin_count + morgana_count + mordred_count + oberon_count)
        
        # 确保不会出现负数
        minion_count = max(0, minion_count)

        return {
            "num_players": config.num_players,
            "max_player_id": config.num_players - 1,
            "num_good": config.num_good,
            "num_evil": config.num_evil,
            
            "merlin_count": merlin_count,
            "percival_count": percival_count,
            "servant_count": servant_count,
            
            "assassin_count": assassin_count,
            "morgana_count": morgana_count,
            "mordred_count": mordred_count,
            "oberon_count": oberon_count,
            "minion_count": minion_count,
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

