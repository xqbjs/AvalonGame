# -*- coding: utf-8 -*-
"""Game state manager for unified web (avalon + diplomacy)."""
import asyncio
import queue
from typing import Dict, Optional, Any
from datetime import datetime


class GameStateManager:
    """Manages game state, message queues, and WebSocket connections."""
    
    def __init__(self):
        self.input_queues: Dict[str, queue.Queue] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.websocket_connections: Dict[str, Any] = {}
        self.player_sessions: Dict[int, str] = {} # player_id -> sid
        self.game_state: Dict[str, Any] = {
            "game": None,
            "phase": None,
            "mission_id": None,
            "round_id": None,
            "leader": None,
            "status": "waiting",  # waiting, running, finished, stopped, error
            "round": None,
            "map_svg": None,
            "obs_log_entry": None,
            "logs": None,
        }
        self.mode: Optional[str] = None
        self.user_agent_id: Optional[str] = None
        self.should_stop: bool = False
        self.game_thread: Optional[Any] = None
        self.history: list[Dict[str, Any]] = []
    
    def set_mode(self, mode: str, user_agent_id: Optional[str] = None, game: Optional[str] = None):
        """Set the game mode and game name."""
        self.game_state["game"] = game
        self.mode = mode
        self.user_agent_id = user_agent_id
    
    def stop_game(self):
        """Stop the current game."""
        self.should_stop = True
        self.update_game_state(status="stopped")
        if hasattr(self, '_game_task') and self._game_task:
            try:
                self._game_task.cancel()
            except Exception:
                pass
    
    def reset(self):
        """Reset the game state manager."""
        self.should_stop = False
        self.game_thread = None
        current_game = self.game_state.get("game")
        self.game_state = {
            "game": current_game,
            "phase": None,
            "mission_id": None,
            "round_id": None,
            "leader": None,
            "status": "waiting",
            "round": None,
            "map_svg": None,
            "obs_log_entry": None,
            "logs": None,
        }
        self.history = []
    
    def set_game_thread(self, thread: Any):
        """Set the game thread reference."""
        self.game_thread = thread
    
    async def put_user_input(self, agent_id: str, content: str):
        if agent_id not in self.input_queues:
            self.input_queues[agent_id] = queue.Queue()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.input_queues[agent_id].put, content)
    
    async def get_user_input(self, agent_id: str, timeout: Optional[float] = None) -> str:
        if agent_id not in self.input_queues:
            self.input_queues[agent_id] = queue.Queue()
        
        try:
            loop = asyncio.get_event_loop()
            
            def get_from_queue():
                if timeout:
                    try:
                        return self.input_queues[agent_id].get(timeout=timeout)
                    except queue.Empty:
                        raise TimeoutError(f"Timeout waiting for user input from agent {agent_id}")
                return self.input_queues[agent_id].get()
            
            return await loop.run_in_executor(None, get_from_queue)
        except TimeoutError:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
    
    async def broadcast_message(self, message: Dict[str, Any]):
        if self.should_stop:
            return
        
        await self.message_queue.put(message)
        
        disconnected = []
        for conn_id, websocket in self.websocket_connections.items():
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(conn_id)
        
        for conn_id in disconnected:
            self.websocket_connections.pop(conn_id, None)
    
    def add_websocket_connection(self, connection_id: str, websocket: Any):
        """Add a WebSocket connection."""
        self.websocket_connections[connection_id] = websocket
    
    def remove_websocket_connection(self, connection_id: str):
        """Remove a WebSocket connection."""
        self.websocket_connections.pop(connection_id, None)
    
    def update_game_state(self, **kwargs):
        self.game_state.update(kwargs)
        if self.game_state.get("game") == "diplomacy":
            snapshot_keys = ["phase", "round", "status", "map_svg", "obs_log_entry", "logs", "mission_id", "round_id", "leader"]
            snapshot = {k: self.game_state.get(k) for k in snapshot_keys}
            snapshot["timestamp"] = datetime.now().isoformat()
            snapshot["kind"] = "state"
            self.history.append(snapshot)
    
    def save_history_snapshot(self, kind: str = "state"):
        if self.game_state.get("game") != "diplomacy":
            return
        snapshot_keys = ["phase", "round", "status", "map_svg", "obs_log_entry", "logs", "mission_id", "round_id", "leader"]
        snapshot = {k: self.game_state.get(k) for k in snapshot_keys}
        snapshot["timestamp"] = datetime.now().isoformat()
        snapshot["kind"] = kind
        self.history.append(snapshot)
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state."""
        return self.game_state.copy()
    
    def format_message(self, sender: str, content: str, role: str = "assistant") -> Dict[str, Any]:
        return {
            "type": "message",
            "sender": sender,
            "content": content,
            "role": role,
            "timestamp": datetime.now().isoformat(),
        }
    
    def format_game_state(self) -> Dict[str, Any]:
        return {
            "type": "game_state",
            **self.game_state,
        }
    
    def format_user_input_request(self, agent_id: str, prompt: str) -> Dict[str, Any]:
        return {
            "type": "user_input_request",
            "agent_id": agent_id,
            "prompt": prompt,
        }

