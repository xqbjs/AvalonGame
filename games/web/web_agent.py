# -*- coding: utf-8 -*-
"""Unified web agents for Avalon + Diplomacy."""
from typing import Any
from agentscope.agent import AgentBase, UserAgent
from agentscope.message import Msg

from games.web.game_state_manager import GameStateManager
from games.web.web_user_input import WebUserInput
from games.games.avalon.utils import Parser as AvalonParser


class WebUserAgent(UserAgent):
    
# games/web/web_agent.py

    def __init__(self, name: str, state_manager: GameStateManager, player_id: int = 0):
        super().__init__(name=name)
        self.state_manager = state_manager
        self.agent_id = self.id
        self.player_id = player_id 
        
        # [修改这里] 传入 player_id
        web_input = WebUserInput(state_manager, player_id=self.player_id)
        self.override_instance_input_method(web_input)
        
        # WebUserInput needs to know which player to wait for?
        # The put_user_input in server.py stores by agent_id.
        # We assume self.agent_id matches what the frontend sends (which maps to player_id in our logic)
        web_input = WebUserInput(state_manager, player_id=self.player_id)
        self.override_instance_input_method(web_input)
    
    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        await super().observe(msg)
        if self.state_manager.mode != "participate" or msg is None:
            return
            
        messages = msg if isinstance(msg, list) else [msg]
        for m in messages:
            if isinstance(m, Msg):
                # Try to clean content using parser if it's Avalon
                content = m.content
                if hasattr(AvalonParser, 'extract_text_from_content'):
                     content = AvalonParser.extract_text_from_content(m.content)
                
                sender = m.name
                role = m.role
            else:
                content = str(m)
                sender = "System"
                role = "assistant"
            
            # CRITICAL MULTIPLAYER LOGIC:
            # We want to send this observation specifically to THIS user's socket.
            # If state_manager supports it, use send_personal_message.
            # Otherwise fall back to broadcast (legacy behavior).
            
            formatted_msg = self.state_manager.format_message(sender=sender, content=content, role=role)
            
            if hasattr(self.state_manager, "send_personal_message"):
                # Send ONLY to this player
                await self.state_manager.send_personal_message(self.player_id, formatted_msg)
            else:
                # Fallback for single player or if method missing
                await self.state_manager.broadcast_message(formatted_msg)
    
    async def reply(self, msg: Msg | list[Msg] | None = None, structured_model: Any = None) -> Msg:
        if msg is not None:
            await self.observe(msg)
        
        # When waiting for reply, we might need to notify frontend to "Enable Input"
        # The WebUserInput inside calls `state_manager.wait_for_user_input(self.agent_id)`
        # And GameStateManager usually broadcasts "user_input_request".
        # We need to make sure that request is ALSO targeted.
        
        # Note: We can't easily hook into `web_input` here without modifying `web_user_input.py`
        # OR `game_state_manager.py`.
        # Assuming GameStateManager handles the broadcast of 'user_input_request', 
        # we hope it broadcasts to ALL or matches ID.
        # Ideally, GameStateManager logic should be updated to target the request too.
        
        return await super().reply(msg=msg, structured_model=structured_model)


class ObserveAgent(AgentBase):
    
    def __init__(self, name: str, state_manager: GameStateManager, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.state_manager = state_manager
    
    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        messages = msg if isinstance(msg, list) else [msg]
        for m in messages:
            if isinstance(m, Msg):
                content = m.content
                if hasattr(AvalonParser, 'extract_text_from_content'):
                     content = AvalonParser.extract_text_from_content(m.content)
                sender = m.name
            else:
                content = str(m)
                sender = "Unknown"
            
            # Observers see everything via broadcast
            await self.state_manager.broadcast_message(
                {"type": "message", "sender": sender, "content": content, "role": "assistant"}
            )
    
    def reply(self, x: dict = None) -> Msg:
        return Msg(self.name, content="", role="assistant")